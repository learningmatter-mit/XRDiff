"""calibrate.py — fit the aug constants hard-coded in xrd_scripts/augment.py on
the RRUFF calibration set.

Intensity: one Gamma (MLE) on observed/simulated
  intensity ratios of every simulated peak >= 5 % (observed = max of the measured
  pattern within +/- 0.25 deg).
Noise: per pattern, a Gamma on its background bins (> 0 and <= its 5th
  percentile) -> L_i = (log a_i, log scale_i).  One Gamma on all background bins
  -> L_pool.  Partial pooling: L_i' = (1 - LAM) L_i + LAM L_pool, LAM = 0.75, then
  NOISE_LOGNORMAL_MU / COV = mean / covariance of the L_i'.

    python scripts/rruff_eval/calibrate.py --out data/rruff/aug_constants.json
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Structure
from scipy.stats import gamma

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from diffcsp.common.xrd_utils import xrd_calculator  # noqa: E402

warnings.filterwarnings("ignore")
GRID = np.arange(5.0, 90.0, 0.1)          # the 850-bin model grid
LAM = 0.75                                # partial-pooling weight of the noise fit


def intensity_ratios(cif, obs, min_height=0.05, window=0.25):
    p = xrd_calculator.get_pattern(Structure.from_str(cif, fmt="cif"))
    out = []
    for t, y in zip(p.x, np.array(p.y) / 100):
        if y >= min_height and 5 <= t < 90:
            yo = obs[np.abs(GRID - t) <= window].max()
            if yo > 0:
                out.append(yo / y)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default=str(ROOT / "data/rruff/rruff_calibration_data.csv"))
    ap.add_argument("--out", default=str(ROOT / "data/rruff/aug_constants.json"))
    a = ap.parse_args()
    cal = pd.read_csv(a.calib)
    patterns = [np.array(json.loads(s)) for s in cal["xrd_0.1"]]

    r = np.concatenate([intensity_ratios(c, x) for c, x in zip(cal.cif, patterns)])
    ia, _, iscale = gamma.fit(r[r <= np.percentile(r, 99)], floc=0)

    L, bins = [], []
    for x in patterns:
        b = x[(x > 0) & (x <= np.percentile(x, 5))]
        if len(b) >= 5:
            ga, _, gs = gamma.fit(b, floc=0)
            L.append([np.log(ga), np.log(gs)]); bins.append(b)
    L = np.array(L)
    pa, _, ps = gamma.fit(np.concatenate(bins), floc=0)
    Ls = (1 - LAM) * L + LAM * np.array([np.log(pa), np.log(ps)])
    cov = np.cov(Ls.T)

    c = {"INTENSITY_GAMMA_A": round(float(ia), 4), "INTENSITY_GAMMA_SCALE": round(float(iscale), 4),
         "NOISE_LOGNORMAL_MU": np.round(Ls.mean(0), 4).tolist(), "NOISE_LOGNORMAL_COV": np.round(cov, 6).tolist(),
         "_provenance": f"scripts/rruff_eval/calibrate.py on {Path(a.calib).name} ({len(cal)} patterns, "
                        f"{len(L)} with a background fit)"}
    Path(a.out).write_text(json.dumps(c, indent=1))
    print(json.dumps({k: v for k, v in c.items() if k != "_provenance"}))


if __name__ == "__main__":
    main()
