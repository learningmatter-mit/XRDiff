# XRDiff

An XRD-conditioned crystal structure generator, built on top of [DiffCSP](https://github.com/jiaor17/DiffCSP).

### Setup

```
python==3.8.13
torch==1.9.0
torch-geometric==1.7.2
pytorch_lightning==1.3.8
pymatgen==2023.8.10
```

Rename `.env.template` to `.env` and set:

```
PROJECT_ROOT: the absolute path of this repo
HYDRA_JOBS: the absolute path to save hydra outputs
WABDB_DIR: the absolute path to save wandb outputs
```

### Training

`<dataset>` is `alex_mp_20_poly` or `oqmd`. Common options for all XRDiff models:

```
COMMON="data=<dataset> data.xrd_bins=0.1 data.for_exp=true model.decoder.xrd_cond=true model.decoder.xrd_cond_type=film model.decoder.film_mode=mp"

# full-spectrum conditioning
python diffcsp/run.py $COMMON data.xrd_type=full expname=xrdiff_film

# peak-list conditioning
python diffcsp/run.py $COMMON data.xrd_type=peaks expname=xrdiff_film_peaks

# full spectrum + RRUFF-calibrated augmentation
python xrd_scripts/augment.py --pt_path data/<dataset>/train_xrd_0.1_for_exp.pt   # static: peak shapes / intensities
python diffcsp/run.py $COMMON data.xrd_type=full data.xrd_augmented=true model.xrd_noise_augment=true expname=xrdiff_film_aug
```

The augmentation constants are fitted on the RRUFF calibration set by
`scripts/rruff_eval/calibrate.py` and hard-coded in `xrd_scripts/augment.py`.

### Data

- Alex MP 20 Poly
- OQMD Poly
- RRUFF

### Evaluation

Test sets (Alex-MP-20-Poly, OQMD-Poly): see `scripts/evaluation/README.md`.

```
python scripts/evaluation/evaluate.py --model_path=<run_dir> --eval_split=test --num_evals=3
python scripts/evaluation/compute_metrics.py --root_path=<run_dir> --eval_split=test --label=n_evals_3 --multi_eval
```

RRUFF-148 (20 candidates per sample, ranked by Rietveld refinement against the measured
pattern; see `scripts/rietveld_ranking/README.md`):

```
bash scripts/rruff_eval/rank20.sh <run_dir> rruff_pv <name>              # full-spectrum models
bash scripts/rruff_eval/rank20.sh <run_dir> rruff_pv_peaks <name>   # peaks models
```

The `rruff_pv_peaks` peak lists are built by `scripts/rruff_eval/build_rruff_pv_peaks.py`.

### Acknowledgments

Built on [DiffCSP](https://github.com/jiaor17/DiffCSP).
