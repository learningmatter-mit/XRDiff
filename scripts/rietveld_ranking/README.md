# Reference-free ranking of XRDiff candidates by Rietveld refinement (RRUFF)

Oracle coverage@k counts a sample as solved if *any* of its k candidates matches
the ground truth, which needs a reference to pick the right one. This pipeline
ranks the candidates of each sample using only the observed pattern: it refines
every candidate against the RRUFF pattern the model was conditioned on, scores the
fit, and sorts by it. It then reports how often a matching candidate is among the
n best-ranked (success@n) next to coverage@k.

The ground truth is used only (i) to compute `matched_to_gt` for evaluation and
(ii) in the separate ground-truth sanity-check refinements. Nothing that
filters, refines or ranks candidates reads it.

## Pipeline

| Step | Script | Env | Produces (in `<run_dir>/rietveld_ranking/`) |
| --- | --- | --- | --- |
| 1 | `prepare.py` | repo env | `manifest.csv`, `cifs/`, `observed/`, `coverage_check.json` |
| 2 | `refine.py` → `g2_worker.py` | repo env → GSAS-II env | `align_k20/work/<job_id>/{result.json, fit_main.npz, fit_po.npz, refine.gpx, refine.lst}` |
| 3 | `analyze.py` | repo env | `align_k20/{per_candidate.csv, summary.json, ranking_figure.{pdf,png}}` |
| 4 | `scripts/rruff_eval/report.py` | repo env | `align_k20/rank20_report.txt`: success@1 / @3 / @10 / @20 |

The paper evaluation (20 candidates per sample, pre-aligned protocol) runs all steps
through one command:

```bash
bash scripts/rruff_eval/rank20.sh MODEL_DIR EVAL_DATASET VARIANT
```

`EVAL_DATASET` is `rruff_pv` for full-spectrum models and `rruff_pv_peaks` for peaks
models. `rank20.sh` generates the 20 candidates (`evaluate.py --num_evals=20`,
`compute_metrics.py`), writes `scripts/rruff_eval/configs/VARIANT.yaml`, and runs the steps
above. Each step can also be run by hand:

```bash
python scripts/rietveld_ranking/prepare.py --config CFG --variant V
python scripts/rietveld_ranking/refine.py  --config CFG --variant V --force
python scripts/rietveld_ranking/analyze.py --config CFG --variant V
python scripts/rruff_eval/report.py        --config CFG --variant V
```

All protocol settings live in `config.yaml`. `config_align.yaml` inherits it (`base:`)
and adds the lattice pre-alignment (below); the per-model configs written by `rank20.sh`
inherit `config_align.yaml` and set only `run_dirs`, `gen_files`, `metrics_files` and
`output_subdir`.

### GSAS-II

GSAS-II 2.0.0 (build 5841, git `f75620dc`), built from source with its own pixi
workspace (`pixi run -e py311 install`). `refine.py` runs `g2_worker.py` with the
interpreter in `$G2_PYTHON`, i.e. `<GSAS-II>/pixi/.pixi/envs/py311/bin/python`.

## Inputs

* **Candidates**: `eval_diff_<EVAL_DATASET>_n_evals_20.pt` in the model directory, 20
  independent samples per RRUFF pattern.
* **Observed pattern**: `data/rruff/test_pv_xrd.pt`, 8,500 points at 0.01° over
  5–89.99° 2θ. Each is exactly a RRUFF *Processed* XY profile, min-max scaled
  (background-subtracted, floor 0). Raw counts are not available.
* **Ground truth**: `input_data_batch` in the generation file, the same source
  `compute_metrics.py` uses.

## Settings

| Setting | Value | Why / source |
| --- | --- | --- |
| Wavelength | 1.54184 Å, single line (no Kα2) | pymatgen `XRDCalculator(wavelength='CuKa')` in `diffcsp/common/xrd_utils.py`, the simulation used for training. RRUFF XY headers give no wavelength; all 148 are Cu Kα. |
| 2θ range | 5.00–89.99°, step 0.01° (native grid) | the conditioning range; same for every candidate of a sample |
| Intensities | observed y (max 1) × 10,000 | numeric conditioning only; Rwp is scale-invariant |
| Weights | **unit** (esd = 1 for every point) | raw counts unavailable, so w = 1/y_obs is not possible. GoF is therefore only a relative quantity, not ≈ 1 for a perfect fit. |
| Validity filter | min interatomic distance (incl. periodic images) < 0.5 Å, volume per atom > 1000 Å³, or composition ≠ conditioning composition → invalid, worst score, kept in the set | |
| Symmetry | `SpacegroupAnalyzer(symprec=0.1).get_refined_structure()` (standard conventional cell, idealized coordinates), written with `CifWriter(symprec=0.1)` | |
| Atoms | coordinates, occupancies fixed; Uiso = 0.01 Å² for every atom, fixed | |
| Instrument (`instrument/generic_lab_CuKa.instprm`) | Type PXC, Lam 1.54184, Polariz. 0.5, Azimuth 0, Zero 0, U 2.0, V −2.0, W 5.0, X 0, Y 0, Z 0, SH/L 0.002 | profile values are GSAS-II's own `CuKa lab data` defaults (`GSASII/defaultIparms.py`); Polariz. 0.5 = unpolarized, the Lorentz-polarization model pymatgen uses. Not tuned on RRUFF. |
| Background | shifted Chebyshev (`chebyschev-1`), 6 terms | |
| Lattice pre-alignment | before stage a, a coarse (0.005) then fine (0.0005) grid over an isotropic factor f ∈ [0.90, 1.10] on a, b, c. The candidate's own GSAS-II reflection list (d, Fc²·Icorr) is rendered with a fixed 0.15° Gaussian at each f and scored by Pearson correlation with the observed pattern | predicted lattices are often off by more than a peak width, which otherwise collapses the scale at stage a |
| Start values | one 0-cycle evaluation, then scale s and constant background c from linear LS y_obs ≈ s·(y_calc − bkg) + c | reference-free; same for every structure |
| Stage a | histogram scale + background | |
| Stage b | + zero shift | |
| Stage c | + lattice parameters (GSAS-II constrains them by the space group) | |
| Stage d | + U, V, W, X, Y | |
| Stage e (sensitivity pass only) | + March–Dollase along [001], ratio starts at 1.0 | run from the stage-d state as a separate pass; not used for ranking |
| Refine calls | one `refine()` per stage, up to 50 LS cycles (`max cyc`), GSAS-II `min dM/M` = 0.001, analytic Hessian LM | |
| Convergence | a stage that does not reach ΔM/M < 0.001 within 50 cycles → failure (worst score) | |
| "Stuck" (GSAS-II `Ouch#4`) | not a failure: best state kept, sequence continues, recorded in `stuck_stages` | |
| Other failures | exception, non-finite Rwp/χ², refine returning no Rvals → worst score, reason logged | Rvals are cleared before each stage, since GSAS-II otherwise keeps the previous stage's values after a failed `refine()` |
| Timeout | 120 **CPU** s per pass (main a–d; PO e separately), `SIGPROF`; hard wall-clock kill at 300 s | CPU time so outcomes don't depend on node load. `Timeout` derives from `BaseException`, because GSAS-II's internal `except Exception` would swallow it. |
| Matcher | `StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)`, `get_rms_dist`, through `compute_metrics.RecEvalBatch` incl. its `Crystal.valid` gate | identical to `compute_metrics.py`; `prepare.py` checks coverage against the stored metrics |
| Seeds | bootstrap `numpy.default_rng(42)`, 1000 resamples over samples | |
| Parallelism | 44 worker processes on one 48-core node, each refinement a separate subprocess in its own directory | |

### Scores

* **Rwp** (primary): GSAS-II `Rvals['Rwp']` after stage d. Lowest = rank 1.
* **Bragg R** (`R_Bragg` column): GSAS-II's per-phase `ph:H:Rf` = 100·Σ|Fo − Fc| / Σ|Fo|,
  where Fo = √(Rietveld-partitioned Fo²). This is **R_F on |F|**. GSAS-II also reports
  `Rf^2` (on F²), kept as the `R_F2` column.
* **AIC** = N·ln(RSS/N) + 2p, with N = points in range, RSS = Σw(y_o − y_c)² (unit weights),
  and p = number of refined parameters.
* **GoF** = √(χ²/(N − p)) as GSAS-II reports it. With unit weights it is not normalized.
* Worst score = +∞. Ties, including all-worst, are broken by lowest candidate index.
  Candidate order is the generation order, so this is equivalent to an arbitrary pick.

### Evaluation

`report.py`: success@n for n = 1, 3, 10, 20 — the fraction of the 148 samples with a
matching candidate among the n best-ranked by Rwp. success@20 equals oracle coverage@20.
95% CIs: percentile bootstrap over samples (1000 resamples).

`analyze.py` additionally reports ranked@1 by Bragg R and AIC, the random-pick expectation
(mean over samples of the fraction of candidates that match), the preferred-orientation
sensitivity pass, failures per stage, and a ground-truth sanity check: (i) the fraction of
samples where the GT Rwp is below every non-matching candidate's Rwp (samples where all
candidates match are excluded and counted), and (ii) the GT's rank among the candidates
plus GT.

## Known caveats

* Four ground-truth CIFs are missing atoms (Anglesite, Baryte, Celestine, Bornite).
  Their "matches" and GT refinements are against a wrong structure.
* Unit weights, a background-subtracted observed profile and a single wavelength without
  Kα2 all make absolute Rwp values high (27–45% even for the ground truth). Only the
  ordering within a sample is used.
* The ranking can only reward a candidate whose lattice the local refinement (after the
  pre-alignment) can pull onto the observed peaks.
* The validity filter drops candidates whose composition differs from the conditioning
  composition. For CSP models (composition given) this is a no-op.
