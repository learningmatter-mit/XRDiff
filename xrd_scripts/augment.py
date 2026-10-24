"""
augment.py — XRD spectrum augmentation for crystal structure diffusion models.

STATIC augmentation (command-line):
  Re-simulates each spectrum from its stored CIF with four Caglioti (U,V,W)
  parameter sets.  For all variants except (0.05, -0.01, 0.01) each peak
  intensity is independently scaled by a Gamma-distributed factor fitted by
  MLE to observed/simulated peak-intensity ratios from RRUFF PXRD data.
  All four variants are written to one <stem>_aug.pt file.

DYNAMIC augmentation (import):
  XRDAugmentor — adds per-sample background noise during training.  At every
  call it samples per-spectrum Gamma(a_i, scale_i) parameters from a
  bivariate log-normal, clips extreme draws, and adds noise to the batch.

Usage:
  python xrd_scripts/augment.py --dataset mp_20 --split train --bins 0.1
  python xrd_scripts/augment.py --pt_path /path/to/cached.pt --bins 0.1
"""

import argparse
import os
import sys
from typing import Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

# Allow imports from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from diffcsp.common.xrd_utils import (
    xrd_calculator,
    simulate_pv_xrd_for_row,
)
from pymatgen.core import Structure

# ── Empirical constants (fitted to RRUFF) ────────────────
# Both are fitted on the RRUFF calibration set by scripts/rruff_eval/calibrate.py:
# intensity = one Gamma on observed/simulated peak-intensity ratios; noise =
# per-pattern background Gamma fits, partially pooled (lambda = 0.75) toward one
# Gamma fitted on all background bins.
INTENSITY_GAMMA_A     = 1.8341
INTENSITY_GAMMA_SCALE = 0.4917
NOISE_LOGNORMAL_MU = np.array([0.4249, -5.8853])
NOISE_LOGNORMAL_COV = np.array([
    [0.020108, -0.012364],
    [-0.012364, 0.065216],
])

# Clip thresholds in log space: draws outside these bounds are resampled
LOG_SCALE_MAX = np.log(0.05)
LOG_A_MAX = np.log(10.0)

# ── UVW parameter sets ─────────────────────────────────────────────────────────
# (U, V, W) or None for the random-W variant;  bool = apply intensity scaling
UVW_CONFIGS = [
    ((0.05, -0.06,  0.07), True),
    ((0.05, -0.01,  0.01), False),   # reference variant — no intensity scaling
    ((0.0,   0.0,   0.1),  True),
    (None,                 True),    # W ~ Uniform(0.001, 0.1), U = V = 0
]

DEFAULT_ETA = 0.1
DEFAULT_BIN = 0.1


# ── Helpers ────────────────────────────────────────────────────────────────────

def _draw_uvw(uvw, rng: np.random.Generator):
    """Return concrete (U, V, W) triple, sampling W for the random variant."""
    if uvw is None:
        W = rng.uniform(0.001, 0.1)
        return (0.0, 0.0, W)
    return uvw


def _scale_peaks(xy_merge: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Multiply each peak intensity by an independent Gamma(a, scale) sample.
    xy_merge: (n_peaks, 2) array — column 0 is 2θ, column 1 is intensity.
    Returns a copy with scaled intensities.
    """
    scaled = xy_merge.copy()
    n = len(scaled)
    factors = rng.gamma(INTENSITY_GAMMA_A, INTENSITY_GAMMA_SCALE, size=n)
    scaled[:, 1] *= factors
    return scaled


def simulate_augmented_xrd(
    cif_str: str,
    uvw,
    apply_intensity: bool,
    eta: float = DEFAULT_ETA,
    bin: float = DEFAULT_BIN,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Simulate a pseudo-Voigt XRD pattern from a CIF string.

    Parameters
    ----------
    cif_str : str
        CIF-format string of the crystal structure.
    uvw : tuple(float, float, float) or None
        Caglioti parameters (U, V, W).  Pass None to sample W uniformly.
    apply_intensity : bool
        Whether to apply per-peak Gamma intensity scaling.
    eta : float
        Lorentzian fraction for the pseudo-Voigt profile.
    bin : float
        2θ step size in degrees.
    rng : numpy Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    np.ndarray  shape (n_bins,), values in [0, 1]
    """
    if rng is None:
        rng = np.random.default_rng()

    structure = Structure.from_str(cif_str, fmt="cif")
    pattern = xrd_calculator.get_pattern(structure)

    x = np.array(pattern.x)
    y = np.array(pattern.y)
    xy_merge = np.column_stack((x, y))

    if apply_intensity:
        xy_merge = _scale_peaks(xy_merge, rng)

    U, V, W = _draw_uvw(uvw, rng)
    return simulate_pv_xrd_for_row(xy_merge, U, V, W, eta, bin=bin)


# ── Static augmentation ────────────────────────────────────────────────────────

def _augment_one(job):
    cif, uvw, apply_intensity, eta, bins, seed_key = job
    try:
        rng = np.random.default_rng(list(seed_key))
        return simulate_augmented_xrd(cif, uvw=uvw, apply_intensity=apply_intensity, eta=eta,
                                      bin=bins, rng=rng).astype(np.float32)
    except Exception:
        return None


def augment_dataset(
    pt_path: str,
    bins: float = DEFAULT_BIN,
    eta: float = DEFAULT_ETA,
    seed: int = 42,
    workers: int = 1,
) -> None:
    """
    Augment every entry in a pre-processed .pt dataset file.

    All four UVW variants are concatenated into a single output file named
    <stem>_aug.pt, placed in the same directory as the input.

    Parameters
    ----------
    pt_path : str
        Path to the cached dataset .pt file (list of dicts with 'cif', 'xrd', …).
    bins : float
        2θ bin size for the output spectra (0.01 or 0.1).
    eta : float
        Lorentzian fraction for the pseudo-Voigt profile.
    seed : int
        Base random seed; each UVW variant gets seed + variant_index.
    """
    print(f"Loading {pt_path}")
    data = torch.load(pt_path, map_location="cpu")
    out_dir = os.path.dirname(pt_path)
    stem = os.path.splitext(os.path.basename(pt_path))[0]
    out_path = os.path.join(out_dir, f"{stem}_aug.pt")

    all_augmented = []

    if workers > 1:
        # Parallel path: each entry gets its own generator, seeded by
        # (seed, variant, entry index), so the output does not depend on the
        # number of workers.  The serial path below keeps one generator per
        # variant.
        from multiprocessing import Pool
        cifs = [entry["cif"] for entry in data]
        for variant_idx, (uvw, apply_intensity) in enumerate(UVW_CONFIGS):
            tag_v = "uvw_rand" if uvw is None else "uvw_{:.2f}_{:.2f}_{:.2f}".format(*uvw).replace("-", "n")
            print(f"\nVariant {variant_idx+1}/4: {tag_v}  intensity_scaling={apply_intensity}  ({workers} workers)")
            jobs = [(cif, uvw, apply_intensity, eta, bins, (seed, variant_idx, i))
                    for i, cif in enumerate(cifs)]
            with Pool(workers) as pool:
                xrds = list(tqdm(pool.imap(_augment_one, jobs, chunksize=256), total=len(jobs), desc=tag_v))
            for entry, xrd in zip(data, xrds):
                new_entry = {k: v for k, v in entry.items()}
                if xrd is not None:
                    new_entry["xrd"] = xrd
                all_augmented.append(new_entry)
        torch.save(all_augmented, out_path)
        print(f"\nSaved {len(all_augmented)} entries ({len(data)} × {len(UVW_CONFIGS)} variants) -> {out_path}")
        return

    for variant_idx, (uvw, apply_intensity) in enumerate(UVW_CONFIGS):
        rng = np.random.default_rng(seed + variant_idx)
        tag = "uvw_rand" if uvw is None else "uvw_{:.2f}_{:.2f}_{:.2f}".format(*uvw).replace("-", "n")
        print(f"\nVariant {variant_idx+1}/4: {tag}  intensity_scaling={apply_intensity}")

        for entry in tqdm(data, desc=tag):
            new_entry = {k: v for k, v in entry.items()}
            try:
                xrd = simulate_augmented_xrd(
                    entry["cif"],
                    uvw=uvw,
                    apply_intensity=apply_intensity,
                    eta=eta,
                    bin=bins,
                    rng=rng,
                )
                new_entry["xrd"] = xrd.astype(np.float32)
            except Exception as e:
                print(f"  Warning: {entry.get('mp_id', '?')}: {e}")
            all_augmented.append(new_entry)

    torch.save(all_augmented, out_path)
    print(f"\nSaved {len(all_augmented)} entries ({len(data)} × {len(UVW_CONFIGS)} variants) -> {out_path}")


# ── Dynamic background noise augmentor ────────────────────────────────────────

class XRDAugmentor:
    """
    Adds empirically-calibrated background noise to a batch of XRD patterns.

    At each call, per-sample Gamma(a_i, scale_i) parameters are drawn from a
    bivariate log-normal distribution fitted to RRUFF background residuals.
    Draws where log(scale) > log(0.05) or log(a) > log(10) are resampled.
    Noise is added to the pattern and clipped at the 99th percentile of the
    noise distribution to suppress extreme artefacts.

    Parameters
    ----------
    max_resample : int
        Maximum resampling attempts before clamping the draw.
    seed : int or None
        Optional fixed seed for reproducibility.
    """

    def __init__(self, max_resample: int = 20, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self.max_resample = max_resample
        self.mu, self.cov = NOISE_LOGNORMAL_MU, NOISE_LOGNORMAL_COV

    def _sample_gamma_params(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sample n valid (a, scale) pairs from the bivariate log-normal."""
        log_params = np.full((n, 2), np.nan)
        unfilled = np.ones(n, dtype=bool)

        for _ in range(self.max_resample):
            need = unfilled.sum()
            if need == 0:
                break
            draws = self.rng.multivariate_normal(
                self.mu, self.cov, size=need
            )
            valid = (draws[:, 1] <= LOG_SCALE_MAX) & (draws[:, 0] <= LOG_A_MAX)
            idx = np.where(unfilled)[0][valid]
            log_params[idx] = draws[valid]
            unfilled[idx] = False

        # Clamp any remaining unfilled to mean params
        if unfilled.any():
            log_params[unfilled] = self.mu

        a = np.exp(log_params[:, 0])
        scale = np.exp(log_params[:, 1])
        return a, scale

    def __call__(
        self, xrd_batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply noise augmentation to a batch of XRD patterns.

        Parameters
        ----------
        xrd_batch : torch.Tensor  shape (n, L)
            Batch of normalised XRD patterns.

        Returns
        -------
        torch.Tensor  shape (n, L), same device as input.
        """
        n, L = xrd_batch.shape
        device = xrd_batch.device

        a, scale = self._sample_gamma_params(n)

        noise_np = np.stack(
            [self.rng.gamma(a[i], scale[i], size=L) for i in range(n)],
            axis=0,
        ).astype(np.float32)

        # Clip at 99th percentile of per-sample noise to suppress outliers
        clip_vals = np.percentile(noise_np, 99, axis=1, keepdims=True)
        noise_np = np.clip(noise_np, 0.0, clip_vals)

        noise = torch.from_numpy(noise_np).to(device)
        return (xrd_batch + noise).clamp(0.0, None)


# ── CLI ────────────────────────────────────────────────────────────────────────

def _resolve_pt_path(dataset: str, split: str, bins: float, base_dir: str) -> str:
    candidates = [
        os.path.join(base_dir, "data", dataset, f"{split}_xrd_{bins}.pt"),
        os.path.join(base_dir, "data", dataset, f"{split}_xrd_{bins:g}.pt"),
        os.path.join(base_dir, "data", dataset, f"{split}.pt"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        f"Could not find cached .pt for dataset={dataset} split={split} bins={bins}. "
        f"Tried: {candidates}"
    )


def main():
    parser = argparse.ArgumentParser(description="XRD augmentation script")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--dataset", type=str,
        help="Dataset name (subfolder under data/), e.g. mp_20"
    )
    group.add_argument(
        "--pt_path", type=str,
        help="Direct path to a cached .pt dataset file"
    )
    parser.add_argument("--split", type=str, default="train",
                        help="Data split: train / val / test  (used with --dataset)")
    parser.add_argument("--bins", type=float, default=DEFAULT_BIN,
                        help="2θ bin size for simulated spectra (default 0.1)")
    parser.add_argument("--eta", type=float, default=DEFAULT_ETA,
                        help="Lorentzian fraction η for pseudo-Voigt (default 0.1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed")
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel processes (per-entry seeds; default 1 = serial path)")
    args = parser.parse_args()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    if args.pt_path:
        pt_path = args.pt_path
    else:
        pt_path = _resolve_pt_path(args.dataset, args.split, args.bins, base_dir)

    augment_dataset(pt_path, bins=args.bins, eta=args.eta, seed=args.seed,
                    workers=args.workers)


if __name__ == "__main__":
    main()
