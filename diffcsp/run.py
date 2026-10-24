from pathlib import Path
from typing import List
import sys
sys.path.append('.')
import hydra
import numpy as np
import torch
import omegaconf
import pytorch_lightning as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything, Callback
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger

from diffcsp.common.utils import log_hyperparameters, PROJECT_ROOT

import wandb

import warnings
# Suppress unnecessary warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*version_base.*')
warnings.filterwarnings('ignore', message='.*DataLoader.*deprecated.*')
warnings.filterwarnings('ignore', message='.*Creating a tensor from a list.*')
warnings.filterwarnings('ignore', message='.*To copy construct from a tensor.*')
# Suppress noisy but non-critical warnings
warnings.filterwarnings(
    "ignore",
    message="The version_base parameter is not specified.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    module="pymatgen.analysis.local_env",
    category=UserWarning,
)
torch.set_float32_matmul_precision('high')

def build_callbacks(cfg: DictConfig) -> List[Callback]:
    callbacks: List[Callback] = []

    if "lr_monitor" in cfg.logging:
        hydra.utils.log.info("Adding callback <LearningRateMonitor>")
        callbacks.append(
            LearningRateMonitor(
                logging_interval=cfg.logging.lr_monitor.logging_interval,
                log_momentum=cfg.logging.lr_monitor.log_momentum,
            )
        )

    if "early_stopping" in cfg.train:
        hydra.utils.log.info("Adding callback <EarlyStopping>")
        callbacks.append(
            EarlyStopping(
                monitor=cfg.train.monitor_metric,
                mode=cfg.train.monitor_metric_mode,
                patience=cfg.train.early_stopping.patience,
                verbose=cfg.train.early_stopping.verbose,
            )
        )

    if "model_checkpoints" in cfg.train:
        hydra.utils.log.info("Adding callback <ModelCheckpoint>")
        callbacks.append(
            ModelCheckpoint(
                dirpath=Path(HydraConfig.get().run.dir),
                monitor=cfg.train.monitor_metric,
                mode=cfg.train.monitor_metric_mode,
                save_top_k=cfg.train.model_checkpoints.save_top_k,
                verbose=cfg.train.model_checkpoints.verbose,
                save_last=cfg.train.model_checkpoints.save_last,
            )
        )

    # train.save_resume_ckpt: also keep the latest epoch in <run dir>/resume/, so
    # short (e.g. 2 h debug-partition) continuation jobs don't roll back to the best
    # epoch. The subdir keeps it out of the *.ckpt glob used for evaluation.
    if OmegaConf.select(cfg, "train.save_resume_ckpt", default=False):
        callbacks.append(
            ModelCheckpoint(
                dirpath=Path(HydraConfig.get().run.dir) / "resume",
                filename="latest",
                save_top_k=1,
                every_n_epochs=1,
            )
        )

    return callbacks


def run(cfg: DictConfig) -> None:
    """
    Generic train loop

    :param cfg: run configuration, defined by Hydra in /conf
    """
    if cfg.train.deterministic:
        seed_everything(cfg.train.random_seed)

    if cfg.train.pl_trainer.fast_dev_run:
        hydra.utils.log.info(
            f"Debug mode <{cfg.train.pl_trainer.fast_dev_run=}>. "
            f"Forcing debugger friendly configuration!"
        )
        # Debuggers don't like GPUs nor multiprocessing
        cfg.train.pl_trainer.gpus = 0
        cfg.data.datamodule.num_workers.train = 0
        cfg.data.datamodule.num_workers.val = 0
        cfg.data.datamodule.num_workers.test = 0

        # Switch wandb mode to offline to prevent online logging
        cfg.logging.wandb.mode = "offline"

    # Hydra run directory
    hydra_dir = Path(HydraConfig.get().run.dir)
    if OmegaConf.select(cfg, "logging.wandb") is not None:
        OmegaConf.update(cfg, "logging.wandb.save_dir", str(hydra_dir / "wandb"))

    # Instantiate datamodule
    hydra.utils.log.info(f"Instantiating <{cfg.data.datamodule._target_}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False
    )

    # Instantiate model (from scratch or from pretrained checkpoint)
    pretrained_ckpt = OmegaConf.select(cfg, "train.pretrained_ckpt")
    if pretrained_ckpt:
        ckpt_path = Path(pretrained_ckpt)
        if ckpt_path.is_dir():
            ckpts = list(ckpt_path.glob("*.ckpt"))
            if not ckpts:
                raise FileNotFoundError(f"No .ckpt found in {ckpt_path}")
            last_ckpts = [c for c in ckpts if "last" in c.name]
            ckpt_path = last_ckpts[0] if last_ckpts else ckpts[np.array([int(c.name.split("-")[0].split("=")[1]) for c in ckpts]).argsort()[-1]]
        pretrained_ckpt = str(ckpt_path)
        hydra.utils.log.info(f"Loading pretrained model from <{pretrained_ckpt}>")
        ModelClass = hydra.utils.get_class(cfg.model._target_)
        # Pass decoder=cfg.model.decoder so xrd_pretrain etc. use current run config.
        # Otherwise pretrain ckpt (xrd_pretrain=True) keeps XRD disabled during fine-tune.
        model: pl.LightningModule = ModelClass.load_from_checkpoint(
            pretrained_ckpt,
            decoder=cfg.model.decoder,
            optim=cfg.optim,
            data=cfg.data,
            logging=cfg.logging,
            strict=False,
        )
    else:
        hydra.utils.log.info(f"Instantiating <{cfg.model._target_}>")
        model: pl.LightningModule = hydra.utils.instantiate(
            cfg.model,
            optim=cfg.optim,
            data=cfg.data,
            logging=cfg.logging,
            _recursive_=False,
        )

    # Pass scaler from datamodule to model
    hydra.utils.log.info(f"Passing scaler from datamodule to model <{datamodule.scaler}>")
    if datamodule.scaler is not None:
        model.lattice_scaler = datamodule.lattice_scaler.copy()
        model.scaler = datamodule.scaler.copy()
    torch.save(datamodule.lattice_scaler, hydra_dir / 'lattice_scaler.pt')
    torch.save(datamodule.scaler, hydra_dir / 'prop_scaler.pt')
    # Instantiate the callbacks
    callbacks: List[Callback] = build_callbacks(cfg=cfg)

    # Logger instantiation/configuration
    wandb_logger = None
    if "wandb" in cfg.logging:
        hydra.utils.log.info("Instantiating <WandbLogger>")
        # Ensure wandb directory exists (save_dir is set in config via ${hydra:run.dir}/wandb)
        wandb_dir = Path(cfg.logging.wandb.save_dir)
        wandb_dir.mkdir(parents=True, exist_ok=True)
        wandb_config = cfg.logging.wandb
        wandb_logger = WandbLogger(
            **wandb_config,
            settings=wandb.Settings(start_method="fork"),
            tags=cfg.core.tags,
        )
        hydra.utils.log.info("W&B is now watching <{cfg.logging.wandb_watch.log}>!")
        wandb_logger.watch(
            model,
            log=cfg.logging.wandb_watch.log,
            log_freq=cfg.logging.wandb_watch.log_freq,
        )

    # Store the YaML config separately into the wandb dir
    # When xrd_type is peaks, saved config should show xrd_bins: null
    if OmegaConf.select(cfg, "data.xrd_type") == "peaks":
        OmegaConf.update(cfg, "data.xrd_bins", None)
    yaml_conf: str = OmegaConf.to_yaml(cfg=cfg)
    (hydra_dir / "hparams.yaml").write_text(yaml_conf)
    # Also update Hydra's config in the run dir so it stays consistent
    hydra_config = hydra_dir / ".hydra" / "config.yaml"
    if hydra_config.exists():
        hydra_config.write_text(yaml_conf)

    # Load checkpoint (if exist)
    ckpts = list(hydra_dir.glob('*.ckpt'))
    if len(ckpts) > 0:
        ckpt_epochs = np.array([int(ckpt.parts[-1].split('-')[0].split('=')[1]) for ckpt in ckpts])
        ckpt = str(ckpts[ckpt_epochs.argsort()[-1]])
        hydra.utils.log.info(f"found checkpoint: {ckpt}")
    else:
        ckpt = None
          
    hydra.utils.log.info("Instantiating the Trainer")
    trainer = pl.Trainer(
        default_root_dir=hydra_dir,
        logger=wandb_logger,
        callbacks=callbacks,
        deterministic=cfg.train.deterministic,
        check_val_every_n_epoch=cfg.logging.val_check_interval,
        **cfg.train.pl_trainer,
    )

    log_hyperparameters(trainer=trainer, model=model, cfg=cfg)

    hydra.utils.log.info("Starting training!")
    # train.resume: continue from the checkpoint found in this run dir (used by
    # continuation jobs that re-enter the same hydra.run.dir after a SLURM time limit).
    resume_from = ckpt if OmegaConf.select(cfg, "train.resume", default=False) else None
    resume_ckpts = sorted((hydra_dir / "resume").glob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    if (OmegaConf.select(cfg, "train.resume", default=False)
            and OmegaConf.select(cfg, "train.save_resume_ckpt", default=False) and resume_ckpts):
        resume_from = str(resume_ckpts[-1])
    if resume_from:
        hydra.utils.log.info(f"Resuming training from <{resume_from}>")
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=resume_from)
    # fit() returned on its own (early stopping or max_epochs), i.e. not killed by a
    # time limit: continuation jobs check this marker and exit.
    (hydra_dir / "TRAINING_DONE").write_text(f"epoch {trainer.current_epoch}\n")

    hydra.utils.log.info("Starting testing!")
    trainer.test(datamodule=datamodule)

    # Logger closing to release resources/avoid multi-run conflicts
    if wandb_logger is not None:
        wandb_logger.experiment.finish()


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
