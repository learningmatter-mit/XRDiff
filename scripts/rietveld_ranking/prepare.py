"""Step 1 of the Rietveld ranking pipeline (runs in the repo env, e.g. symmcd).

For every generation file listed in config.yaml:
  * rebuild candidates and ground truth exactly as compute_metrics.py does and
    match them with the same RecEval / RecEvalBatch code, so matched_to_gt is
    the paper's criterion; abort if coverage@k differs from the stored metrics;
  * apply the ranking validity filter (min distance, composition);
  * symmetrize (SpacegroupAnalyzer, symprec from config) and write a CIF per
    candidate and per ground truth;
  * write the observed pattern of each sample as a 3-column .xye.

The ground truth is used here only to compute matched_to_gt (evaluation) and to
build the separate ground-truth sanity-check jobs. Nothing the refinement or
ranking consumes depends on it.

    python scripts/rietveld_ranking/prepare.py --variant peaks
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from p_tqdm import p_map
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts' / 'evaluation'))
os.chdir(ROOT)
from compute_metrics import Crystal, RecEval, RecEvalBatch, get_crystal_array_list  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config  # noqa: E402

N_LATTICE_PARAMS = {'triclinic': 6, 'monoclinic': 4, 'orthorhombic': 3, 'tetragonal': 2,
                    'trigonal': 2, 'hexagonal': 2, 'cubic': 1}




def validity(crys, gt_comp, min_dist, max_vol_per_atom):
    """Ranking validity filter. Returns (valid, reason)."""
    if not getattr(crys, 'constructed', False):
        return False, f"not_constructed:{getattr(crys, 'invalid_reason', 'unknown')}"
    s = crys.structure
    if s.composition != gt_comp:
        return False, 'composition_mismatch'
    # runaway cell (seen in the OQMD model, up to 1e6 A): no reflections in range, and the
    # neighbour search below stalls or needs terabytes. Typical crystals are 5-60 A^3/atom.
    if s.volume / len(s) > max_vol_per_atom:
        return False, 'runaway_cell'
    # any pair (incl. periodic self-images) closer than min_dist
    try:
        i, _, _, _ = s.get_neighbor_list(r=min_dist)
    except MemoryError:   # pathological cell: the neighbour grid needs terabytes
        return False, 'degenerate_cell'
    if len(i) > 0:
        return False, 'min_distance'
    return True, ''


def symmetrize(structure, symprec):
    """Standard conventional cell with idealized coordinates, plus metadata."""
    sga = SpacegroupAnalyzer(structure, symprec=symprec)
    refined = sga.get_refined_structure()
    ds = sga.get_symmetry_dataset()
    info = {'sg_number': int(sga.get_space_group_number()),
            'spacegroup': sga.get_space_group_symbol(),
            'hall_number': int(ds.hall_number if hasattr(ds, 'hall_number') else ds['hall_number']),
            'crystal_system': sga.get_crystal_system()}
    info['n_lattice_params'] = N_LATTICE_PARAMS[info['crystal_system']]
    return refined, info


def write_cif(structure, symprec, path):
    CifWriter(structure, symprec=symprec).write_file(path)


def cif_symmetry(path):
    """Space group of the CIF actually written (and refined). CifWriter(symprec)
    re-detects symmetry on the refined structure and, for ~4% of candidates, finds
    a higher group than the first pass and writes the structure in that group's
    standard cell; the recorded metadata must follow the file GSAS-II reads."""
    from pymatgen.symmetry.groups import SpaceGroup
    num = sym = None
    for line in open(path):
        if line.startswith('_symmetry_Int_Tables_number'):
            num = int(line.split()[1])
        elif line.startswith('_symmetry_space_group_name_H-M'):
            sym = line.split(None, 1)[1].strip().strip("'")
    cs = SpaceGroup.from_int_number(num).crystal_system
    return {'sg_number': num, 'spacegroup': sym, 'crystal_system': cs, 'n_lattice_params': N_LATTICE_PARAMS[cs]}


def process_structure(args):
    """Validity + symmetrization + CIF for one structure (candidate or GT)."""
    crys, gt_comp, cfg, cif_path = args
    valid, reason = validity(crys, gt_comp, cfg['min_interatomic_distance'], cfg['max_volume_per_atom'])
    out = {'valid': valid, 'invalid_reason': reason, 'spacegroup': '', 'sg_number': -1,
           'hall_number': -1, 'crystal_system': '', 'n_lattice_params': -1, 'cif_path': ''}
    if not valid:
        return out
    try:
        refined, info = symmetrize(crys.structure, cfg['symprec'])
        write_cif(refined, cfg['symprec'], cif_path)
        info['sg_number_first_pass'] = info['sg_number']
        info.update(cif_symmetry(cif_path))
        out.update(info)
        out['cif_path'] = str(cif_path)
    except Exception as e:  # symmetrization failure -> treated as invalid, never dropped
        out.update(valid=False, invalid_reason=f'symmetrize_failed:{type(e).__name__}')
    return out


def per_candidate_matches(pred_crys, gt_crys, k, cfg):
    """(k, n_samples) bool array of matches, via the paper's own evaluator classes."""
    kw = cfg['matcher']
    if k == 1:
        ev = RecEval(pred_crys[0], gt_crys, structurematcher_kwargs=kw, return_match_indices=True)
        res = ev.get_match_rate_and_rms()
        m = np.zeros((1, len(gt_crys)), bool)
        m[0, res['match_indices']] = True
    else:
        ev = RecEvalBatch(pred_crys, gt_crys, structurematcher_kwargs=kw, return_match_indices=True)
        res = ev.get_match_rate_and_rms()
        m = ~np.isnan(ev.all_rms_dis.astype(float))
    return m, res


def write_xye(y, cfg, path):
    x = cfg['two_theta_min'] + np.arange(len(y)) * cfg['two_theta_step']
    y = np.asarray(y, float) * cfg['intensity_scale']
    np.savetxt(path, np.column_stack([x, y, np.ones_like(y)]), fmt='%.4f %.6f %.1f')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(ROOT / 'scripts/rietveld_ranking/config.yaml'))
    ap.add_argument('--variant', default='peaks')
    ap.add_argument('--refresh_symmetry', action='store_true',
                    help='only re-read space-group metadata from the written CIFs into manifest.csv')
    args = ap.parse_args()
    cfg = load_config(args.config)
    run_dir = ROOT / cfg['run_dirs'][args.variant]
    out_dir = run_dir / 'rietveld_ranking'
    (out_dir / 'cifs').mkdir(parents=True, exist_ok=True)
    (out_dir / 'observed').mkdir(exist_ok=True)
    if args.refresh_symmetry:
        man = pd.read_csv(out_dir / 'manifest.csv')
        man['sg_number_first_pass'] = man['sg_number']
        for i in man.index[man.valid]:
            for key, val in cif_symmetry(man.at[i, 'cif_path']).items():
                man.at[i, key] = val
        man.to_csv(out_dir / 'manifest.csv', index=False)
        print((man.sg_number != man.sg_number_first_pass).sum(), 'entries updated to the CIF space group')
        return

    observed = torch.load(ROOT / cfg['observed_pt'], map_location='cpu')
    rows, coverage_check, gt_done = [], {}, False
    for k, fname in cfg['gen_files'].items():
        k = int(k)
        path = run_dir / fname
        data = torch.load(path, map_location='cpu', weights_only=False)
        sample_ids = list(data['material_ids'])
        assert data['frac_coords'].shape[0] == k, f'{fname}: expected {k} candidates'

        cand_arrays, gt_arrays = get_crystal_array_list(str(path), batch_idx=-1)
        gt_crys = p_map(lambda x: Crystal(x), gt_arrays, desc=f'gt k={k}')
        pred_crys = [p_map(lambda x: Crystal(x), cand_arrays[j], desc=f'k={k} cand {j}')
                     for j in range(k)]

        matches, res = per_candidate_matches(pred_crys, gt_crys, k, cfg)
        with open(run_dir / cfg['metrics_files'][k]) as f:
            ref = json.load(f)['matcher']['structure matcher default']
        coverage = matches.any(0).mean()
        ok = np.isclose(coverage, ref['match_rate']) and \
            sorted(np.where(matches.any(0))[0].tolist()) == sorted(ref['match_indices'])
        coverage_check[k] = {'reproduced': float(coverage), 'manuscript': ref['match_rate'], 'ok': bool(ok)}
        print(f'k={k}: coverage {coverage:.4f} vs manuscript {ref["match_rate"]:.4f} -> {"OK" if ok else "MISMATCH"}')
        if not ok:
            json.dump(coverage_check, open(out_dir / 'coverage_check.json', 'w'), indent=2)
            sys.exit(f'coverage@{k} does not reproduce the manuscript number; stopping.')

        gt_comps = [g.structure.composition for g in gt_crys]
        jobs, meta = [], []
        for j in range(k):
            for i, sid in enumerate(sample_ids):
                cif = out_dir / 'cifs' / f'{args.variant}_k{k}_s{i:03d}_c{j:02d}.cif'
                jobs.append((pred_crys[j][i], gt_comps[i], cfg, cif))
                meta.append(dict(job_id=cif.stem, sample_idx=i, sample_id=sid, input_variant=args.variant,
                                 gen_file_k=k, cand_idx=j, matched_to_gt=bool(matches[j, i])))
        if not gt_done:  # GT sanity-check jobs: same protocol, once per sample
            for i, sid in enumerate(sample_ids):
                cif = out_dir / 'cifs' / f'gt_s{i:03d}.cif'
                jobs.append((gt_crys[i], gt_comps[i], cfg, cif))
                meta.append(dict(job_id=cif.stem, sample_idx=i, sample_id=sid, input_variant='gt',
                                 gen_file_k=0, cand_idx=-1, matched_to_gt=True))
                write_xye(observed[sid][0].numpy(), cfg, out_dir / 'observed' / f's{i:03d}.xye')
            gt_done = True
        results = p_map(process_structure, jobs, desc=f'symmetrize k={k}')
        for m, r in zip(meta, results):
            m.update(r)
            m['xye_path'] = str(out_dir / 'observed' / f's{m["sample_idx"]:03d}.xye')
            rows.append(m)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'manifest.csv', index=False)
    json.dump(coverage_check, open(out_dir / 'coverage_check.json', 'w'), indent=2)
    print(f'{len(df)} jobs -> {out_dir / "manifest.csv"}')
    print(df.groupby(['input_variant', 'gen_file_k'])['valid'].agg(['sum', 'count']))
    print(df.loc[~df.valid, 'invalid_reason'].value_counts())


if __name__ == '__main__':
    main()
