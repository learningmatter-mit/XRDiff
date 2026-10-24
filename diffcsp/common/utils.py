import os
from pathlib import Path
from typing import Optional

import dotenv
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf


def get_env(env_name: str, default: Optional[str] = None) -> str:
    """
    Safely read an environment variable.
    Raises errors if it is not defined or it is empty.

    :param env_name: the name of the environment variable
    :param default: the default (optional) value for the environment variable

    :return: the value of the environment variable
    """
    if env_name not in os.environ:
        if default is None:
            raise KeyError(
                f"{env_name} not defined and no default value is present!")
        return default

    env_value: str = os.environ[env_name]
    if not env_value:
        if default is None:
            raise ValueError(
                f"{env_name} has yet to be configured and no default value is present!"
            )
        return default

    return env_value


def load_envs(env_file: Optional[str] = None) -> None:
    """
    Load all the environment variables defined in the `env_file`.
    This is equivalent to `. env_file` in bash.

    It is possible to define all the system specific variables in the `env_file`.

    :param env_file: the file that defines the environment variables to use. If None
                     it searches for a `.env` file in the project.
    """
    dotenv.load_dotenv(dotenv_path=env_file, override=True)


STATS_KEY: str = "stats"


# Adapted from https://github.com/hobogalaxy/lightning-hydra-template/blob/6bf03035107e12568e3e576e82f83da0f91d6a11/src/utils/template_utils.py#L125
def log_hyperparameters(
    cfg: DictConfig,
    model: pl.LightningModule,
    trainer: pl.Trainer,
) -> None:
    """This method controls which parameters from Hydra config are saved by Lightning loggers.
    Additionally saves:
        - sizes of train, val, test dataset
        - number of trainable model parameters
    Args:
        cfg (DictConfig): [description]
        model (pl.LightningModule): [description]
        trainer (pl.Trainer): [description]
    """
    hparams = OmegaConf.to_container(cfg, resolve=True)

    # save number of model parameters
    hparams[f"{STATS_KEY}/params_total"] = sum(p.numel()
                                               for p in model.parameters())
    hparams[f"{STATS_KEY}/params_trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams[f"{STATS_KEY}/params_not_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)

    # disable logging any more hyperparameters for all loggers
    # (this is just a trick to prevent trainer from logging hparams of model, since we already did that above)
    trainer.logger.log_hyperparams = lambda params: None


# Load environment variables
load_envs()

# Set the cwd to the project root
PROJECT_ROOT: Path = Path(get_env("PROJECT_ROOT"))
assert (
    PROJECT_ROOT.exists()
), "You must configure the PROJECT_ROOT environment variable in a .env file!"

os.chdir(PROJECT_ROOT)


# Register custom OmegaConf resolvers
def _xrd_save_path(xrd_type: str, root_path: str, xrd_bins: str, split: str, for_exp: str = "false", augmented: str = "false", w_uni: str = "false", aug_tag: str = "", peak_tag: str = "") -> str:
    """Resolver for conditional XRD save path based on xrd_type.

    Usage: ${xrd_save_path:${data.xrd_type},${data.root_path},${data.xrd_bins},train,${data.for_exp},${data.xrd_augmented},${data.w_uni}}
    Append _for_exp when for_exp=true (rruff-filtered training sets).
    Append _aug when augmented=true (statically augmented training sets from xrd_scripts/augment.py).
    aug_tag / peak_tag: optional extra file-name suffixes for the _aug / peaks files (default: none).
    Append _w_uni when w_uni=true (poly _for_exp + uni sets merged).
    """
    if str(w_uni).lower() == "true":
        suffix = "_w_uni"
    else:
        suffix = "_for_exp" if str(for_exp).lower() == "true" else ""
        if str(augmented).lower() == "true":
            suffix += "_aug"
            if str(aug_tag) not in ("", "None", "none", "null"):
                suffix += str(aug_tag)
    if xrd_type == "full":
        return f"{root_path}/{split}_xrd_{xrd_bins}{suffix}.pt"
    elif xrd_type == "peaks":
        if str(peak_tag) not in ("", "None", "none", "null"):
            suffix += str(peak_tag)
        return f"{root_path}/{split}_peaks{suffix}.pt"
    else:
        raise ValueError(f"Unknown xrd_type: {xrd_type}. Must be 'full' or 'peaks'.")


OmegaConf.register_new_resolver("xrd_save_path", _xrd_save_path)
