# Evaluation pipeline

Everything needed to take a trained XRDiff checkpoint and produce the numbers and
figures we report. The five scripts in this directory run in order; steps 3–5 are
independent of each other and all consume the `.pt` written in step 1.

| Step | Script | Produces |
| --- | --- | --- |
| 1 | `evaluate.py` | `eval_diff_<name>.pt` — generated structures conditioned on test XRD |
| 2 | `compute_metrics.py` | `eval_metrics_<name>.json` — match rate + RMSD vs. ground truth |
| 3 | `failures_analysis.py` | `failures_analysis/fails_<name>.{csv,png,_sg.png,_summary.json}` |
| 4 | `amd_calc.py` | `eval_amd_<name>.txt` — AMD/PDD earth-mover distances |
| 5 | `lattice_parity.py` | `lattice_parity/lattice_parity_<...>.png` + `_stats.txt` |

All outputs land inside the run directory itself, so a run folder is self-contained.

## Before you start

* Run everything from the repo root with `PROJECT_ROOT` set (see `.env.template`).
* `--model_path` / `--root_path` is always a Hydra run directory, e.g.
  `out/hydra/singlerun/xrdiff_film_peaks/alex_mp_20_poly/2026-04-26_14-52-32`.
  It must contain a `*.ckpt`, `hparams.yaml`, `lattice_scaler.pt` and (usually)
  `prop_scaler.pt`. `eval_utils.load_model` picks `last.ckpt` if present, otherwise the
  checkpoint with the highest epoch number.
* Only step 1 needs a GPU. Steps 2–5 are CPU-bound and parallelize over all cores
  (`p_map`), so give them a fat CPU allocation and plenty of RAM.

## Step 1 — Generate structures (`evaluate.py`)

Loads the checkpoint, builds the data loader from the run's own `hparams.yaml`, and runs
the reverse diffusion (`model.sample`) once per test example, conditioned on that
example's XRD pattern. Ground-truth structures travel along in `input_data_batch`, which
is what every later step compares against.

```bash
python scripts/evaluation/evaluate.py \
    --model_path=out/hydra/singlerun/xrdiff_film_peaks/alex_mp_20_poly/2026-04-26_14-52-32 \
    --eval_split=test
```

Best-of-N (sample each condition N times):

```bash
python scripts/evaluation/evaluate.py --model_path=<run_dir> --eval_split=test --num_evals=3
```

Key arguments:

| Flag | Default | Notes |
| --- | --- | --- |
| `--eval_split` | `val` | `test`, `val` or `train`. **Pass `test` explicitly** for reported numbers. |
| `--num_evals` | `1` | >1 gives N samples per condition (best-of-N in step 2). |
| `--eval_dataset` | `None` | Evaluate on a different dataset than the model was trained on; `rruff_20`, `rruff_25`, `rruff_pv`, … map to `data/rruff/<name>_data.csv` and force the test split. |
| `--step_lr` | `-1` | `-1` picks the per-dataset value from `eval_utils.recommand_step_lr` (`1e-5` for `alex_mp_20_poly`, keyed by `csp` vs. `csp_multi` depending on `--num_evals`). |
| `--label` | `''` | Free-form tag that goes into the output filename. |
| `--seed` | `42` | Seeds `random`, `numpy` and `torch`. |

Output name (the rule that later steps mirror — see *File naming* below):

| Case | File |
| --- | --- |
| plain | `eval_diff_<split>.pt` |
| `--num_evals N>1` | `eval_diff_n_evals_<N>_<split>.pt` |
| `--label L` | `eval_diff_<L>_<split>.pt` |
| `--eval_dataset rruff_*` | `eval_diff_<dataset>.pt` (no split; `_n_evals_<N>` appended if N>1) |

The saved dict holds `frac_coords`, `atom_types`, `lattices`, `lengths`, `angles`,
`num_atoms`, the `input_data_batch` (ground truth), the argparse namespace and wall-clock
`time`. RRUFF runs additionally store `material_ids`.

## Step 2 — Match rate (`compute_metrics.py`)

Rebuilds pymatgen `Structure`s for predictions and ground truth, then compares each
prediction to its own ground truth with `StructureMatcher`. A sample counts as matched
if `get_rms_dist` returns a value; the reported `rms_dist` is the mean over matched
samples only.

```bash
# single sample per condition
python scripts/evaluation/compute_metrics.py --root_path=<run_dir> --eval_split=test

# best-of-3: matched if ANY of the 3 samples matches
python scripts/evaluation/compute_metrics.py --root_path=<run_dir> --eval_split=test \
    --label=n_evals_3 --multi_eval
```

* `--label` and `--multi_eval` must agree with how step 1 was run, otherwise the `.pt`
  will not be found (or will be read with the wrong batch layout). `--label=multi` without
  `--multi_eval` is rejected by an assertion.
* Matcher tolerances default to `stol=0.3, angle_tol=5, ltol=0.2`. Override with
  `--structurematcher_kwargs '{"stol":0.5,"angle_tol":10,"ltol":0.3}'`.
* Results are written to `eval_metrics_<name>.json` and **merged** into any existing file,
  keyed by the matcher settings (`"structure matcher default"` for the defaults). Re-running
  with different tolerances adds an entry instead of overwriting.
* `--return_match_indices` records which indices matched. For RRUFF labels this is on
  automatically and adds `no_dup_match_rate`, which counts a mineral as solved if any of
  its duplicate patterns matched.
* `--gt_file <csv>` compares against CIFs in a CSV instead of the ground truth stored in
  the `.pt` (used for RRUFF and the `--tasks gen` path).
* The module also defines METRe variants (`RecEvalMetre`, `RecEvalBatchMetre`) that accept
  a match against *any* ground-truth structure of the same composition — the fair criterion
  for composition-only baselines such as vanilla DiffCSP. The CLI here always uses the
  strict one-to-one matcher; METRe rates are produced through `failures_analysis.py --root_path2`.

## Step 3 — Where it fails (`failures_analysis.py`)

Re-runs the same matching criterion per sample and records, for each ground-truth
structure, the number of sites, the unit-cell volume and the space-group number. Produces
two-row figures: training-set distribution on top, match rate in that bin below — so you
can read off whether failures track rarity in training, cell size or symmetry.

```bash
python scripts/evaluation/failures_analysis.py \
    --root_path=<run_dir> --eval_split=test \
    --train_pt=data/alex_mp_20_poly/train_peaks_for_exp.pt
```

* `--train_pt` is **required** and supplies the top (training count) panels. Use the cached
  dataset `.pt` that matches the run's `xrd_type` / `for_exp` settings, e.g.
  `train_peaks_for_exp.pt` for a `peaks` run with `for_exp: true`, `train_xrd_0.1_for_exp.pt`
  for a `full` run.
* Outputs go to `<root_path>/failures_analysis/`: `fails_<name>.csv` (per-sample table),
  `fails_<name>.png` (sites / volume / crystal system), `fails_<name>_sg.png` (space group)
  and `fails_<name>_summary.json`.
* The per-sample CSV is cached — a second run reuses it and only redraws the figures. Pass
  `--force` after changing matcher settings.
* `--root_path2 <other_run>` adds a third row showing the METRe match rate for that second
  run, which is how the DiffCSP-baseline comparison panels are made.
* `--num_workers` (default `-1` = all cores), `--symprec` (default `0.01`) and
  `--max_sites_x` (inferred as 20 for Alex-MP-20-style data, 40 for OQMD) are the other
  knobs.

## Step 4 — AMD distances (`amd_calc.py`)

Structure-similarity measure that does not depend on `StructureMatcher` tolerances:
builds a PDD for prediction and ground truth and reports the earth-mover distance between
them, so near-misses get a number instead of a binary fail.

```bash
python scripts/evaluation/amd_calc.py --root_path=<run_dir> --eval_split=test
```

* Needs the `amd` package (`pip install average-minimum-distance`), or a source checkout
  passed with `--amd_src`. It imports `numba`, so the environment needs that too.
* `--n_neighbors` (default `100`) sets the PDD neighbour count.
* Values above `AMD_ALERT_THRESHOLD = 100` are treated as blow-ups: they are listed
  individually in the report but excluded from mean/std/percentiles.
* Writes `eval_amd_<name>.txt` with count, mean, std, min, p05/p25/median/p75/p95, max.
  With `--multi_eval` it reports both all GT–prediction pairs and the best-of-N per
  ground truth.

## Step 5 — Lattice parity (`lattice_parity.py`)

Parity plots of predicted vs. ground-truth cell lengths *a*, *b*, *c* (log-scale hexbin),
plus a text file of means, variances, medians and the ten largest values per axis — the
quickest way to spot runaway cells.

```bash
python scripts/evaluation/lattice_parity.py --root_path=<run_dir> --eval_split=test
```

* `--niggli` applies Niggli reduction and sorts `a ≤ b ≤ c` before comparing, which
  removes axis-permutation ambiguity. Without it, raw predicted lengths are used.
* Dataset name and `xrd_type` are read from the run's `hparams.yaml` and baked into the
  filename, e.g. `lattice_parity_alex_mp_20_poly_peaks_test_logscale.png`.
* Output directory defaults to `<root_path>/lattice_parity/`; override with `--out_dir`.
* With `--multi_eval` only the first sample per condition is plotted.

## File naming

Steps 2–5 locate the step-1 tensor with the same rule (`compute_metrics.get_file_paths`):

```
label == ''                          ->  eval_<task>_<split>.pt
label in LABELS_OMIT_SPLIT
  or label.startswith('rruff')       ->  eval_<task>_<label>.pt
otherwise                            ->  eval_<task>_<label>_<split>.pt
```

`LABELS_OMIT_SPLIT` currently holds `rruff_20`, `rruff_25`, `LiFePO4`, `SrTiO3`,
`SrTiO3_strain`, `GaN`, `NdFe2`, `multi_NdFe2`. Add new split-less labels there (and to the
copy in `lattice_parity.py`) rather than special-casing at the call site.

## Worked example

`out/hydra/singlerun/xrdiff_film_peaks/alex_mp_20_poly/2026-04-26_14-52-32` — peak-conditioned
XRDiff on Alex-MP-20-poly, 14,883 test structures:

```bash
RUN=out/hydra/singlerun/xrdiff_film_peaks/alex_mp_20_poly/2026-04-26_14-52-32

python scripts/evaluation/evaluate.py       --model_path=$RUN --eval_split=test
python scripts/evaluation/evaluate.py       --model_path=$RUN --eval_split=test --num_evals=3
python scripts/evaluation/compute_metrics.py --root_path=$RUN --eval_split=test
python scripts/evaluation/compute_metrics.py --root_path=$RUN --eval_split=test --label=n_evals_3 --multi_eval
python scripts/evaluation/failures_analysis.py --root_path=$RUN --eval_split=test \
    --train_pt=data/alex_mp_20_poly/train_peaks_for_exp.pt
python scripts/evaluation/amd_calc.py       --root_path=$RUN --eval_split=test
python scripts/evaluation/lattice_parity.py --root_path=$RUN --eval_split=test
```

leaves the run directory holding:

```
eval_diff_test.pt                     eval_metrics_test.json              -> match rate 0.525, RMSD 0.0119
eval_diff_n_evals_3_test.pt           eval_metrics_n_evals_3_test.json    -> match rate 0.679, RMSD 0.0106
eval_amd_test.txt                                                         -> median AMD 0.154 (14,879/14,883 valid)
failures_analysis/fails_test.csv, fails_test.png, fails_test_sg.png, fails_test_summary.json
lattice_parity/lattice_parity_alex_mp_20_poly_peaks_test_logscale.png, ..._stats.txt
```

## Related scripts (outside this directory)

* `type_decoding/` — re-decodes the atom types of an existing `eval_diff_*.pt` so every element
  of the conditioning set appears (coverage decoding), and re-runs the step-2 match evaluation:
  `sbatch scripts/evaluation/type_decoding/run_type_decoding.sh <run_dir> [label ...]`.
* `scripts/rruff_eval/` — the RRUFF-148 evaluation: 20 candidates per sample, ranked by Rietveld
  refinement against the measured pattern (`rank20.sh`, see `scripts/rietveld_ranking/README.md`).
