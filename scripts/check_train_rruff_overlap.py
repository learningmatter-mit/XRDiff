"""
Check if any structures in training sets match structures in test_rruff_pv_full_0.1.pt
using StructureMatcher with high tolerances.

Uses *_for_exp.pt (non-augmented base files) because augmentation only changes XRD
spectra — crystal geometry is identical between _for_exp.pt and _for_exp_aug.pt.
"""
import torch
from pymatgen.core import Structure, Lattice
from pymatgen.analysis.structure_matcher import StructureMatcher


TRAIN_PATHS = {
    'alex_mp_20_poly': 'data/alex_mp_20_poly/train_xrd_0.1_for_exp.pt',
    'oqmd':            'data/oqmd/train_xrd_0.1_for_exp.pt',
}
RRUFF_PATH = 'data/rruff/test_rruff_pv_full_0.1.pt'

# High tolerances
MATCHER = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)


def item_to_structure(item):
    frac_coords, atom_types, lengths, angles, *_ = item['graph_arrays']
    lattice = Lattice.from_parameters(*lengths, *angles)
    return Structure(lattice, atom_types, frac_coords, coords_are_cartesian=False)


def load_rruff(path):
    data = torch.load(path, map_location='cpu')
    structs, ids = [], []
    for i, item in enumerate(data):
        s = item_to_structure(item)
        structs.append(s)
        mineral = item.get('mp_id', item.get('rruff_id', '?'))
        ids.append(f"{i}/{len(data)} {mineral}")
    print(f"  {len(structs)} RRUFF structures, "
          f"{len({s.composition.reduced_formula for s in structs})} unique formulas")
    return structs, ids


def load_train_filtered(path, target_formulas):
    """Load train .pt and return only structures whose formula overlaps with RRUFF."""
    print(f"  Loading {path} ...")
    data = torch.load(path, map_location='cpu')
    print(f"  {len(data)} total train structures — filtering by formula ...")
    candidates = []
    for i, item in enumerate(data):
        s = item_to_structure(item)
        if s.composition.reduced_formula in target_formulas:
            tid = item.get('mp_id', item.get('oqmd_id', str(i)))
            candidates.append((i, tid, s))
    del data
    print(f"  {len(candidates)} formula-matching candidates retained")
    return candidates


def find_matches(rruff_structs, rruff_ids, train_candidates, dataset_name):
    matches = []
    for rs, rid in zip(rruff_structs, rruff_ids):
        rf = rs.composition.reduced_formula
        for ti, tid, ts in train_candidates:
            if ts.composition.reduced_formula != rf:
                continue
            try:
                rms = MATCHER.get_rms_dist(rs, ts)
            except Exception:
                rms = None
            if rms is not None:
                matches.append((rid, ti, tid, rms[0], dataset_name))
    return matches


def main():
    print("=== Loading RRUFF test set ===")
    rruff_structs, rruff_ids = load_rruff(RRUFF_PATH)
    rruff_formulas = {s.composition.reduced_formula for s in rruff_structs}

    all_matches = []
    for dataset_name, train_path in TRAIN_PATHS.items():
        print(f"\n=== {dataset_name} ===")
        candidates = load_train_filtered(train_path, rruff_formulas)
        if not candidates:
            print("  No formula overlap — skipping matcher.")
            continue
        print(f"  Running StructureMatcher (stol=0.5, angle_tol=10, ltol=0.3) ...")
        matches = find_matches(rruff_structs, rruff_ids, candidates, dataset_name)
        print(f"  Matches found: {len(matches)}")
        all_matches.extend(matches)

    print("\n" + "=" * 70)
    print(f"TOTAL MATCHES: {len(all_matches)}")
    if all_matches:
        print(f"\n{'Dataset':<20} {'RRUFF (idx/148 mineral)':<35} {'Train idx':<10} {'Train id':<20} {'RMS'}")
        print("-" * 90)
        for rid, ti, tid, rms, ds in sorted(all_matches, key=lambda x: x[3]):
            print(f"{ds:<20} {str(rid):<35} {ti:<10} {str(tid):<20} {rms:.4f}")
    else:
        print("No matches found in any training set.")


if __name__ == '__main__':
    main()
