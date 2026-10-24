"""report.py — the paper numbers for one rank20.sh run: success@1 / @3 / @10 of
the Rietveld-ranked list of 20 candidates per RRUFF-148 sample, plus
success@20 (= oracle coverage@20).

Ranking rule (as in rietveld_ranking/analyze.py): candidates of a sample are
sorted by refined Rwp, ties broken by candidate index; failed or invalid
candidates carry Rwp = inf.  A sample is a success at n if a candidate matching
the ground truth is among its n best-ranked.  95% CIs: percentile bootstrap over
the 148 samples (minerals), 1000 resamples, seed from the config.

    python scripts/rruff_eval/report.py --config scripts/rruff_eval/configs/VARIANT.yaml --variant VARIANT

Writes <run_dir>/<output_subdir>/rank20_report.txt and prints it.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts' / 'rietveld_ranking'))
from common import dirs, load_config  # noqa: E402

K = 20
NS = (1, 3, 10, K)
N_BOOT = 1000


def success_table(cands):
    """[n_samples, len(NS)] bool: matching candidate among the n best by Rwp."""
    rows = []
    for _, g in cands.groupby('sample_idx'):
        m = g.sort_values(['Rwp', 'cand_idx'], kind='mergesort').matched_to_gt.to_numpy()
        rows.append([m[:n].any() for n in NS])
    return np.array(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--variant', required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    _, out_dir = dirs(cfg, args.variant)
    df = pd.read_csv(out_dir / 'per_candidate.csv')
    cands = df[(df.input_variant == args.variant) & (df.gen_file_k == K)]
    assert cands.groupby('sample_idx').size().eq(K).all(), 'expected 20 candidates per sample'

    tab = success_table(cands)
    rng = np.random.default_rng(cfg['seed'])
    bs = tab[rng.integers(0, len(tab), size=(N_BOOT, len(tab)))].mean(1)
    lo, hi = np.percentile(bs, 2.5, axis=0), np.percentile(bs, 97.5, axis=0)

    lines = [f'{args.variant}: {len(tab)} samples, {K} candidates each, ranked by Rwp (pre-aligned Rietveld)']
    for j, n in enumerate(NS):
        name = f'success@{n}' + (' (= coverage)' if n == K else '')
        lines.append(f'  {name:23s}  {100 * tab[:, j].mean():4.1f}%  '
                     f'95% CI [{100 * lo[j]:.1f}, {100 * hi[j]:.1f}]')
    text = '\n'.join(lines) + '\n'
    (out_dir / 'rank20_report.txt').write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
