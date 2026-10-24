"""
build_rruff_pv_peaks.py — the peak lists fed to the peak-conditioned XRDiff
models for the RRUFF-148 patterns (eval dataset `rruff_pv_peaks`).

Training peak lists come from patterns rendered by diffcsp.common.xrd_utils
with Caglioti U, V, W = 0.1, 0.01, 0.1 (FWHM 0.33 deg at 30 deg, 0.46 deg at
90 deg).  Measured RRUFF peaks are ~0.05-0.1 deg wide, so reflections the
training render merges (close reflections, Ka1/Ka2 doublets) come out as
separate peaks, and noise bumps pass as peaks too.  Per pattern:

  1. resolution match: convolve the 0.01-deg pattern with a Gaussian whose FWHM
     at each 2-theta is sqrt(FWHM_train(2theta)^2 - FWHM_obs^2), the broadening
     that turns the observed resolution into the training one.  FWHM_obs is the
     median width of the pattern's most prominent peaks.
  2. every 10th point (0.1-deg grid), / max, find_peaks(prominence=0).
  3. keep peaks whose prominence is >= 5 sigma of the noise propagated through
     the kernel (sigma = MAD of first differences of the raw pattern).
  4. detection floor: drop peaks lower than 2 % of the strongest peak.
  5. positions = bin / 849, widths / max width.  The model keeps the 20 most
     intense peaks (data_utils.load_peaks_features).

Nothing here is fitted to the 148 test patterns: the 5 sigma threshold was
chosen on non-test calibration patterns, and the rest is fixed by the training
simulator.  Reads data/rruff/rruff_pv_data.csv and test_pv_xrd.pt; writes
data/rruff/rruff_pv_peaks_data.csv (a copy of rruff_pv_data.csv) and its
_peak_info.csv companion.

Usage:  python scripts/rruff_eval/build_rruff_pv_peaks.py
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[2]
RRUFF = ROOT / "data" / "rruff"
N_BINS, TOP_K = 850, 20
K_SIGMA, FLOOR = 5.0, 0.02
U, V, W = 0.1, 0.01, 0.1          # diffcsp/common/xrd_utils.py main()
FINE = 0.01
X_FINE = 5.0 + FINE * np.arange(8500)


def fwhm_train(tth):
    t = np.tan(np.radians(tth / 2))
    return np.sqrt(U * t ** 2 + V * t + W)


def raw_sigma(f):
    d = np.diff(f)
    return 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)


def observed_fwhm(f, sigma, n_top=10):
    """Median FWHM (deg) of the pattern's most prominent, clearly real peaks."""
    p, props = find_peaks(f, prominence=10 * sigma if sigma > 0 else 0, width=0)
    if len(p) == 0:
        return 0.0
    top = np.argsort(props["prominences"])[::-1][:n_top]
    return float(np.median(props["widths"][top]) * FINE)


def broaden(f, fwhm_obs):
    """Variable-width Gaussian convolution on the 0.01-deg grid.
    Returns the smoothed pattern and, per point, sqrt(sum w^2) of its kernel
    (the factor by which white noise is scaled)."""
    fk = np.sqrt(np.clip(fwhm_train(X_FINE) ** 2 - fwhm_obs ** 2, 0, None))
    sig = fk / 2.3548 / FINE                      # in grid points
    out = np.empty_like(f, dtype=float)
    gain = np.ones_like(f, dtype=float)
    for i in range(len(f)):
        s = sig[i]
        if s < 0.3:                               # no broadening needed
            out[i] = f[i]
            continue
        h = int(np.ceil(4 * s))
        lo, hi = max(0, i - h), min(len(f), i + h + 1)
        w = np.exp(-0.5 * ((np.arange(lo, hi) - i) / s) ** 2)
        w /= w.sum()
        out[i] = np.dot(w, f[lo:hi])
        gain[i] = np.sqrt(np.sum(w ** 2))
    return out, gain


def extract(full):
    """full: (8500,) measured pattern on the 0.01-deg grid (any scale).
    Returns (intensity, bin index on the 0.1 grid, prominence, width_norm)."""
    f = np.asarray(full, dtype=np.float64)
    f = (f - f.min()) / (f.max() - f.min())
    sigma = raw_sigma(f)
    sm, gain = broaden(f, observed_fwhm(f, sigma))
    x = sm[::10].astype(np.float32)
    x = x / x.max()
    p, props = find_peaks(x, height=0, width=0, prominence=0)
    prom, wid = props["prominences"], props["widths"]
    if len(p):
        noise = sigma * gain[::10][p] / sm.max()  # same scale as x
        keep = (prom >= K_SIGMA * noise) & (x[p] >= FLOOR)
        p, prom, wid = p[keep], prom[keep], wid[keep]
    wn = wid / wid.max() if len(wid) and wid.max() > 0 else wid
    return x[p], p, prom, wn


def pack(inten, bins, prom, widths_norm):
    return {"peak_intensity": json.dumps(np.asarray(inten, float).tolist()),
            "peak_position": json.dumps((np.asarray(bins, float) / (N_BINS - 1)).tolist()),
            "peak_prominence": json.dumps(np.asarray(prom, float).tolist()),
            "peak_width": json.dumps(np.asarray(widths_norm, float).tolist()),
            "num_peaks": int(len(bins))}


def main():
    base = pd.read_csv(RRUFF / "rruff_pv_data.csv")
    pv = torch.load(RRUFF / "test_pv_xrd.pt", map_location="cpu")
    rows = [{"material_id": mid, **pack(*extract(pv[mid][0].numpy()))} for mid in base.material_id]
    pk = pd.DataFrame(rows)
    shutil.copyfile(RRUFF / "rruff_pv_data.csv", RRUFF / "rruff_pv_peaks_data.csv")
    pk.to_csv(RRUFF / "rruff_pv_peaks_data_peak_info.csv", index=False)
    n = pk.num_peaks.values
    print(f"peaks/pattern median {np.median(n):.0f} [p10 {np.percentile(n, 10):.0f}, "
          f"p90 {np.percentile(n, 90):.0f}], <{TOP_K} peaks: {np.mean(n < TOP_K):.2f}")


if __name__ == "__main__":
    main()
