"""
Parity plots for predicted vs. ground-truth lattice parameters (a, b, c).

Optionally applies Niggli reduction before comparison so that the labeling
of axes is canonical (lengths sorted a ≤ b ≤ c), removing ambiguity from
models that may predict an equivalent cell with axes permuted.

Usage:
    python scripts/evaluation/lattice_parity.py --root_path <path> [--eval_split test]
                                 [--label <label>] [--gt_file <csv>]
                                 [--niggli] [--multi_eval] [--out_dir <dir>]

The file-path conventions mirror compute_metrics.py: the script expects a
eval_diff_<name_part>.pt file produced by evaluate.py.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import gaussian_kde
import yaml
from p_tqdm import p_map

SCRIPTS_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SCRIPTS_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_ROOT))

from eval_utils import load_data, get_crystals_list
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice

import warnings
warnings.filterwarnings("ignore", module="pymatgen")


# ---------------------------------------------------------------------------
# Constants (mirrors compute_metrics.py)
# ---------------------------------------------------------------------------
def _read_run_meta(root_path):
    """Return (dataset_name, xrd_type) from hparams.yaml; fall back gracefully."""
    hparams_path = Path(root_path) / 'hparams.yaml'
    if not hparams_path.exists():
        return 'unknown', 'unknown'
    with open(hparams_path) as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg.get('data', {})
    dataset = data_cfg.get('name', 'unknown')
    xrd_type = data_cfg.get('xrd_type', 'no_xrd')
    return dataset, xrd_type


AXIS_LIMITS = {
    'oqmd':       {'a':  8.0, 'b': 10.0, 'c': 18.0},
    'alex_mp_20_poly': {'a': 10.0, 'b': 12.0, 'c': 32.0},
}

LABELS_OMIT_SPLIT = frozenset({
    'rruff_20', 'rruff_25', 'LiFePO4', 'SrTiO3', 'SrTiO3_strain',
    'GaN', 'NdFe2', 'multi_NdFe2',
})


def get_file_path(root_path, task, split='test', label='', suffix='pt'):
    if label == '':
        name_part = split
    elif label in LABELS_OMIT_SPLIT or label.startswith('rruff'):
        name_part = label
    else:
        name_part = f'{label}_{split}'
    return os.path.join(root_path, f'eval_{task}_{name_part}.{suffix}')


# ---------------------------------------------------------------------------
# Structure loading helpers
# ---------------------------------------------------------------------------

def _build_structure(crys_dict):
    """Return (Structure, valid:bool). Minimal version — no smact/fingerprints."""
    lengths = crys_dict['lengths']
    angles = crys_dict['angles']
    try:
        if np.any(np.isnan(lengths)) or np.any(np.isnan(angles)):
            return None, False
        if np.min(lengths) <= 0:
            return None, False
        s = Structure(
            lattice=Lattice.from_parameters(*(lengths.tolist() + angles.tolist())),
            species=crys_dict['atom_types'],
            coords=crys_dict['frac_coords'],
            coords_are_cartesian=False,
        )
        if s.volume < 0.1:
            return None, False
        return s, True
    except Exception:
        return None, False


def _niggli_lengths(structure):
    """Return sorted (a, b, c) after Niggli reduction."""
    lat = structure.lattice.get_niggli_reduced_lattice()
    abc = sorted(lat.abc)
    return tuple(abc)


def _raw_lengths(structure):
    return tuple(structure.lattice.abc)


def load_structures(crys_array_list, niggli=False):
    """Build structures from array list; return list of (a, b, c) or None."""
    result = []
    get_lengths = _niggli_lengths if niggli else _raw_lengths
    for d in crys_array_list:
        s, valid = _build_structure(d)
        result.append(get_lengths(s) if valid else None)
    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _setup_style():
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 18,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def plot_parity(pred_lengths, gt_lengths, niggli, out_path, dataset=''):
    """
    pred_lengths / gt_lengths: list of (a, b, c) tuples or None.
    Invalid pairs are skipped.
    """
    _setup_style()

    params = ['a', 'b', 'c']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), layout='constrained')

    hb_artists = []

    for col, (ax, param) in enumerate(zip(axes, params)):
        pred_vals, gt_vals = [], []
        for p, g in zip(pred_lengths, gt_lengths):
            if p is not None and g is not None:
                pred_vals.append(p[col])
                gt_vals.append(g[col])

        pred_vals = np.array(pred_vals)
        gt_vals = np.array(gt_vals)

        if len(pred_vals) == 0:
            ax.set_title(f'{param} (no valid pairs)')
            hb_artists.append(None)
            continue

        errors = np.abs(pred_vals - gt_vals)

        lim_lo = 0.0
        axis_caps = AXIS_LIMITS.get(dataset, {})
        lim_hi = axis_caps.get(param, min(max(gt_vals.max(), pred_vals.max()), 32.0))

        hb = ax.hexbin(
            gt_vals, pred_vals,
            gridsize=80,
            extent=(lim_lo, lim_hi, lim_lo, lim_hi),
            cmap='Blues',
            mincnt=1,
            bins='log',
        )
        hb_artists.append(hb)

        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], 'k--', lw=1, zorder=3)
        ax.set_xlim(lim_lo, lim_hi)
        ax.set_ylim(lim_lo, lim_hi)
        ax.set_aspect('equal', adjustable='box')
        ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
        ax.grid(True, alpha=0.25, zorder=0)

        mae = np.mean(errors)
        sem = np.std(errors, ddof=1) / np.sqrt(len(errors))
        ax.set_xlabel(f'Ground truth  {param}  (Å)')
        ax.set_ylabel(f'Generated  {param}  (Å)')
        ax.plot([], [], ' ', label=f'MAE = {mae:.2f} ± {sem:.2f} Å')
        ax.legend(loc='upper left', frameon=True, fontsize=16, handlelength=0)

    # Shared colorbar with unified limits across all panels
    valid_hbs = [h for h in hb_artists if h is not None]
    if valid_hbs:
        all_clims = [h.get_clim() for h in valid_hbs]
        global_vmin = min(c[0] for c in all_clims)
        global_vmax = max(c[1] for c in all_clims)
        for h in valid_hbs:
            h.set_clim(global_vmin, global_vmax)
        cb = fig.colorbar(valid_hbs[-1], ax=axes.tolist(), shrink=1.0, pad=0.02)
        cb.set_label('log₁₀(count)', fontsize=20)
        cb.ax.tick_params(labelsize=16)

    cell_label = 'Niggli cell' if niggli else 'raw cell'  # noqa: F841
    # fig.suptitle(f'Lattice parameter parity — {cell_label}', fontsize=13, y=1.01)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved → {out_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_crystal_array_list(file_path, batch_idx=0):
    """Mirrors compute_metrics.py's helper (without the batch==-1 multi_eval path)."""
    data = load_data(file_path)
    if batch_idx == -1:
        batch_size = data['frac_coords'].shape[0]
        crys_array_list = []
        for i in range(batch_size):
            crys_array_list.append(get_crystals_list(
                data['frac_coords'][i], data['atom_types'][i],
                data['lengths'][i], data['angles'][i], data['num_atoms'][i]))
    elif batch_idx == -2:
        crys_array_list = get_crystals_list(
            data['frac_coords'], data['atom_types'],
            data['lengths'], data['angles'], data['num_atoms'])
    else:
        crys_array_list = get_crystals_list(
            data['frac_coords'][batch_idx], data['atom_types'][batch_idx],
            data['lengths'][batch_idx], data['angles'][batch_idx],
            data['num_atoms'][batch_idx])

    true_crystal_array_list = None
    if 'input_data_batch' in data:
        batch = data['input_data_batch']
        if isinstance(batch, dict):
            true_crystal_array_list = get_crystals_list(
                batch['frac_coords'], batch['atom_types'],
                batch['lengths'], batch['angles'], batch['num_atoms'])
        else:
            true_crystal_array_list = get_crystals_list(
                batch.frac_coords, batch.atom_types,
                batch.lengths, batch.angles, batch.num_atoms)

    return crys_array_list, true_crystal_array_list


def save_stats(pred_lengths, gt_lengths, out_path):
    params = ['a', 'b', 'c']
    lines = []
    for col, param in enumerate(params):
        pred_vals = np.array([p[col] for p in pred_lengths if p is not None])
        gt_vals   = np.array([g[col] for g in gt_lengths   if g is not None])
        lines.append(f'=== {param} ===')
        for label, vals in [('GT', gt_vals), ('Pred', pred_vals)]:
            lines.append(f'  {label}:  mean={vals.mean():.4f}  var={vals.var():.4f}  median={np.median(vals):.4f}')
        lines.append('')

    lines.append('=== Top-10 highest c values ===')
    lines.append('  GT:')
    gt_c = np.array([g[2] for g in gt_lengths if g is not None])
    for v in np.sort(gt_c)[::-1][:10]:
        lines.append(f'    {v:.4f}')
    lines.append('  Pred:')
    pred_c = np.array([p[2] for p in pred_lengths if p is not None])
    for v in np.sort(pred_c)[::-1][:10]:
        lines.append(f'    {v:.4f}')

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'Saved stats → {out_path}')


def main(args):
    file_path = get_file_path(args.root_path, 'diff', args.eval_split, args.label)
    batch_idx = -1 if args.multi_eval else 0
    crys_array_list, true_crystal_array_list = get_crystal_array_list(file_path, batch_idx)

    # Ground truth
    if args.gt_file:
        import pandas as pd
        from pymatgen.core.structure import Structure as _S
        csv = pd.read_csv(args.gt_file)
        def _from_cif(cif):
            s = _S.from_str(cif, fmt='cif')
            return {
                'frac_coords': s.frac_coords,
                'atom_types': np.array([sp.Z for sp in s.species]),
                'lengths': np.array(s.lattice.abc),
                'angles': np.array(s.lattice.angles),
            }
        gt_array_list = list(p_map(_from_cif, csv['cif'].tolist()))
    else:
        gt_array_list = true_crystal_array_list

    # Flatten multi_eval: take first batch for comparison
    if args.multi_eval:
        pred_list = crys_array_list[0]
    else:
        pred_list = crys_array_list

    print(f'Loading {len(pred_list)} predicted structures …')
    pred_lengths = load_structures(pred_list, niggli=args.niggli)
    print(f'Loading {len(gt_array_list)} ground-truth structures …')
    gt_lengths = load_structures(gt_array_list, niggli=args.niggli)

    dataset, xrd_type = _read_run_meta(args.root_path)
    label_str = args.label if args.label else args.eval_split
    fname = f'lattice_parity_{dataset}_{xrd_type}_{label_str}_logscale.png'
    out_dir = args.out_dir if args.out_dir else os.path.join(args.root_path, 'lattice_parity')
    out_path = os.path.join(out_dir, fname)
    plot_parity(pred_lengths, gt_lengths, args.niggli, out_path, dataset=dataset)
    stats_fname = fname.replace('.png', '_stats.txt')
    save_stats(pred_lengths, gt_lengths, os.path.join(out_dir, stats_fname))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', required=True)
    parser.add_argument('--eval_split', default='test')
    parser.add_argument('--label', default='')
    parser.add_argument('--gt_file', default='')
    parser.add_argument('--multi_eval', action='store_true')
    parser.add_argument('--niggli', action='store_true',
                        help='Apply Niggli reduction before comparing (sorts a≤b≤c)')
    parser.add_argument('--out_dir', default='',
                        help='Output directory; defaults to <root_path>/lattice_parity')
    args = parser.parse_args()
    main(args)