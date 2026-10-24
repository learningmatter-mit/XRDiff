"""
Remove training entries whose structures match any rruff/test.csv structure
(as identified by StructureMatcher). Saves filtered copies with '_for_exp' suffix.

Files processed:
  data/alex_mp_20_poly/train_peaks.pt      -> train_peaks_for_exp.pt
  data/alex_mp_20_poly/train_xrd_0.1.pt   -> train_xrd_0.1_for_exp.pt
  data/oqmd/train_peaks.pt                -> train_peaks_for_exp.pt
  data/oqmd/train_xrd_0.1.pt             -> train_xrd_0.1_for_exp.pt

Note: user originally referenced oqmd/test_xrd_0.1.pt but match indices were
derived from train.csv, so train_xrd_0.1.pt is processed here instead.
"""

import gc
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_ids(path):
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def filter_and_save(src: Path, dst: Path, exclude_ids: set, id_key: str = 'mp_id'):
    print(f"\nLoading {src} ...")
    data = torch.load(src, map_location='cpu')
    before = len(data)

    filtered = [d for d in data if str(d[id_key]) not in exclude_ids]
    after = len(filtered)

    del data
    gc.collect()

    print(f"  {before} -> {after} entries (removed {before - after})")
    print(f"  Saving to {dst} ...")
    torch.save(filtered, dst)
    del filtered
    gc.collect()
    print(f"  Done.")


def main():
    # --- alex_mp_20_poly ---
    alex_exclude = load_ids(ROOT / 'data/rruff/alex_mp_20_poly_train_rruff_matches.txt')
    print(f"Alex exclude set: {len(alex_exclude)} mp_ids")

    filter_and_save(
        ROOT / 'data/alex_mp_20_poly/train_peaks.pt',
        ROOT / 'data/alex_mp_20_poly/train_peaks_for_exp.pt',
        alex_exclude,
    )
    filter_and_save(
        ROOT / 'data/alex_mp_20_poly/train_xrd_0.1.pt',
        ROOT / 'data/alex_mp_20_poly/train_xrd_0.1_for_exp.pt',
        alex_exclude,
    )

    # --- oqmd ---
    # oqmd_ids in the .pt dicts are stored as the raw oqmd_id value (int/float).
    # The matches file contains plain integers; we compare as strings of the
    # int value to handle both int and float representations (e.g. 2019 vs 2019.0).
    oqmd_exclude_raw = load_ids(ROOT / 'data/rruff/oqmd_train_rruff_matches.txt')
    # Build a set that matches both "2019" and "2019.0"
    oqmd_exclude = set()
    for v in oqmd_exclude_raw:
        oqmd_exclude.add(v)           # "2019"
        oqmd_exclude.add(v + '.0')    # "2019.0"
        oqmd_exclude.add(str(int(float(v))))  # normalised int string
    print(f"\nOQMD exclude set: {len(oqmd_exclude_raw)} oqmd_ids")

    filter_and_save(
        ROOT / 'data/oqmd/train_peaks.pt',
        ROOT / 'data/oqmd/train_peaks_for_exp.pt',
        oqmd_exclude,
    )
    filter_and_save(
        ROOT / 'data/oqmd/train_xrd_0.1.pt',
        ROOT / 'data/oqmd/train_xrd_0.1_for_exp.pt',
        oqmd_exclude,
    )

    print("\nAll done.")


if __name__ == '__main__':
    main()
