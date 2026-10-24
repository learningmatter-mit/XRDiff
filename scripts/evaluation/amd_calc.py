import argparse
import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from p_tqdm import p_map
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure

# `pip install average-minimum-distance`, or point --amd_src at a source checkout.
if "--amd_src" in sys.argv:
    sys.path.insert(0, sys.argv[sys.argv.index("--amd_src") + 1])
import amd  # noqa: E402

script_dir = Path(__file__).resolve().parent
SCRIPTS_ROOT = script_dir.parent
PROJECT_ROOT = SCRIPTS_ROOT.parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from compute_metrics import (  # noqa: E402
    LABELS_OMIT_SPLIT,
    get_crystal_array_list,
    get_file_paths,
)

# Structures/pairs with AMD above this value are excluded from mean/std/percentiles.
AMD_ALERT_THRESHOLD = 100.0


def get_gt_crys_ori(cif):
    structure = Structure.from_str(cif, fmt="cif")
    return structure


def crystal_array_to_structure(crys_array_dict):
    atom_types = crys_array_dict["atom_types"]
    if len(atom_types.shape) > 1:
        atom_types = np.argmax(atom_types, axis=-1) + 1
    atom_types = [int(z) for z in atom_types.tolist()]
    lengths = crys_array_dict["lengths"]
    angles = crys_array_dict["angles"]
    frac_coords = crys_array_dict["frac_coords"]
    lattice = Lattice.from_parameters(*(lengths.tolist() + angles.tolist()))
    return Structure(
        lattice=lattice,
        species=atom_types,
        coords=frac_coords,
        coords_are_cartesian=False,
    )


def safe_amd_distance(gt_structure, pred_structure, n_neighbors):
    try:
        gt_pset = amd.periodicset_from_pymatgen_structure(gt_structure)
        pred_pset = amd.periodicset_from_pymatgen_structure(pred_structure)
        gt_pdd = amd.PDD(gt_pset, n_neighbors)
        pred_pdd = amd.PDD(pred_pset, n_neighbors)
        dist = amd.EMD(gt_pdd, pred_pdd)
        if dist is None or not np.isfinite(dist):
            return None
        return float(dist)
    except Exception:
        return None


def summarize(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return None
    q = np.percentile(arr, [5, 25, 50, 75, 95])
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "p05": float(q[0]),
        "p25": float(q[1]),
        "median": float(q[2]),
        "p75": float(q[3]),
        "p95": float(q[4]),
        "max": float(arr.max()),
    }


def format_summary(name, summary, total_expected):
    if summary is None:
        return [f"[{name}] no valid AMD values"]
    lines = [f"[{name}]"]
    lines.append(f"valid_count: {summary['count']} / {total_expected}")
    lines.append(f"mean:   {summary['mean']:.8f}")
    lines.append(f"std:    {summary['std']:.8f}")
    lines.append(f"min:    {summary['min']:.8f}")
    lines.append(f"p05:    {summary['p05']:.8f}")
    lines.append(f"p25:    {summary['p25']:.8f}")
    lines.append(f"median: {summary['median']:.8f}")
    lines.append(f"p75:    {summary['p75']:.8f}")
    lines.append(f"p95:    {summary['p95']:.8f}")
    lines.append(f"max:    {summary['max']:.8f}")
    return lines


def _append_high_amd_report(results_lines, lines_prefix, entries, threshold):
    """entries: list of (label_str, amd_value) for stdout + file."""
    results_lines.append("")
    results_lines.append(f"[{lines_prefix}] AMD > {threshold}")
    results_lines.append(f"count: {len(entries)}")
    for label, d in entries:
        results_lines.append(f"  {label}: {d:.8f}")


def _save_amd_scatter_single(out_png, xs, ys, threshold):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(xs, ys, s=14, alpha=0.65, edgecolors="none")
    ax.set_xlabel("Structure index")
    ax.set_ylabel("AMD (EMD)")
    ax.set_title("AMD per structure (single eval)")
    ax.set_ylim(0, threshold)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_amd_scatter_multi(out_png, pair_points, best_amd_by_index, threshold):
    """pair_points: list of (gt_idx, eidx, amd). best_amd_by_index: length n_gt, nan if none."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax0, ax1 = axes
    if pair_points:
        gi = [p[0] for p in pair_points]
        dd = [p[2] for p in pair_points]
        ei = [p[1] for p in pair_points]
        sc = ax0.scatter(
            gi, dd, c=ei, cmap="tab10", s=12, alpha=0.65, edgecolors="none"
        )
        fig.colorbar(sc, ax=ax0, label="eval index")
    ax0.set_xlabel("GT structure index")
    ax0.set_ylabel("AMD (EMD)")
    ax0.set_title("All GT–prediction pairs")
    ax0.set_ylim(0, threshold)
    ax0.grid(True, alpha=0.3)

    bi = np.arange(len(best_amd_by_index), dtype=float)
    bvals = np.asarray(best_amd_by_index, dtype=float)
    mask = np.isfinite(bvals)
    ax1.scatter(bi[mask], bvals[mask], s=14, alpha=0.65, edgecolors="none")
    ax1.set_xlabel("GT structure index")
    ax1.set_ylabel("AMD (EMD)")
    ax1.set_title("Best-of-n AMD per GT (min over evals, thresholded)")
    ax1.set_ylim(0, threshold)
    ax1.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(args):
    if args.amd_src and args.amd_src not in sys.path:
        sys.path.insert(0, args.amd_src)

    diff_file_path = get_file_paths(args.root_path, "diff", args.eval_split, args.label)
    batch_idx = -1 if args.multi_eval else 0
    pred_crys_array, true_crys_array = get_crystal_array_list(diff_file_path, batch_idx=batch_idx)

    if args.gt_file:
        csv = pd.read_csv(args.gt_file)
        gt_structures = p_map(get_gt_crys_ori, csv["cif"])
    else:
        if true_crys_array is None:
            raise ValueError(
                "No ground-truth structures found in eval file and --gt_file was not provided."
            )
        gt_structures = p_map(crystal_array_to_structure, true_crys_array)

    if args.multi_eval:
        pred_structures = [p_map(crystal_array_to_structure, p) for p in pred_crys_array]
    else:
        pred_structures = p_map(crystal_array_to_structure, pred_crys_array)

    n_gt = len(gt_structures)
    results_lines = []
    results_lines.append("AMD evaluation")
    results_lines.append(f"root_path: {args.root_path}")
    results_lines.append(f"eval_split: {args.eval_split}")
    results_lines.append(f"label: {args.label}")
    results_lines.append(f"multi_eval: {args.multi_eval}")
    results_lines.append(f"n_neighbors: {args.n_neighbors}")
    results_lines.append(f"n_ground_truth: {n_gt}")
    results_lines.append(
        f"stats_exclude_amd_above: {AMD_ALERT_THRESHOLD} (excluded from summaries below)"
    )
    results_lines.append("")

    plot_single_xs = plot_single_ys = None
    plot_pair_points = plot_best_amd = None

    if not args.multi_eval:
        if len(pred_structures) != n_gt:
            raise ValueError(
                f"Prediction/GT length mismatch: {len(pred_structures)} vs {n_gt}."
            )
        dists = [
            safe_amd_distance(gt_structures[i], pred_structures[i], args.n_neighbors)
            for i in range(n_gt)
        ]
        high_entries = []
        for i, d in enumerate(dists):
            if d is not None and not math.isnan(d) and d > AMD_ALERT_THRESHOLD:
                high_entries.append((f"index {i}", d))
                print(f"AMD > {AMD_ALERT_THRESHOLD} at index {i}: {d:.8f}")
        n_high = len(high_entries)
        print(f"Count AMD > {AMD_ALERT_THRESHOLD} (excluded from stats): {n_high}")

        dists_for_stats = [
            d
            for d in dists
            if d is not None
            and not math.isnan(d)
            and d <= AMD_ALERT_THRESHOLD
        ]
        summary = summarize(dists_for_stats)
        results_lines.extend(format_summary("single_eval", summary, n_gt))
        results_lines.append(
            f"excluded_from_stats_amd_above_{AMD_ALERT_THRESHOLD}: {n_high}"
        )
        if high_entries:
            _append_high_amd_report(results_lines, "single_eval", high_entries, AMD_ALERT_THRESHOLD)

        plot_single_xs = []
        plot_single_ys = []
        for i, d in enumerate(dists):
            if (
                d is not None
                and not math.isnan(d)
                and math.isfinite(d)
                and d <= AMD_ALERT_THRESHOLD
            ):
                plot_single_xs.append(i)
                plot_single_ys.append(d)
    else:
        num_evals = len(pred_structures)
        if num_evals == 0:
            raise ValueError("No prediction batches found for --multi_eval.")
        for eidx in range(num_evals):
            if len(pred_structures[eidx]) != n_gt:
                raise ValueError(
                    f"Eval {eidx} prediction/GT length mismatch: "
                    f"{len(pred_structures[eidx])} vs {n_gt}."
                )

        all_dists = []
        best_of_n = []
        high_entries = []
        pair_points = []
        best_amd_by_index = [math.nan] * n_gt
        for i in range(n_gt):
            per_gt = []
            for eidx in range(num_evals):
                d = safe_amd_distance(
                    gt_structures[i], pred_structures[eidx][i], args.n_neighbors
                )
                if d is not None and not math.isnan(d):
                    per_gt.append(d)
                    if d <= AMD_ALERT_THRESHOLD:
                        pair_points.append((i, eidx, d))
                        all_dists.append(d)
                    else:
                        label = f"GT index {i}, eval {eidx}"
                        high_entries.append((label, d))
                        print(
                            f"AMD > {AMD_ALERT_THRESHOLD} at GT index {i}, "
                            f"eval {eidx}: {d:.8f}"
                        )
            per_gt_f = [d for d in per_gt if d <= AMD_ALERT_THRESHOLD]
            if per_gt_f:
                m = float(np.min(per_gt_f))
                best_of_n.append(m)
                best_amd_by_index[i] = m

        plot_pair_points = pair_points
        plot_best_amd = best_amd_by_index

        n_high = len(high_entries)
        print(f"Count AMD > {AMD_ALERT_THRESHOLD} (excluded from stats): {n_high}")
        if high_entries:
            _append_high_amd_report(
                results_lines, "multi_eval_pairs", high_entries, AMD_ALERT_THRESHOLD
            )

        results_lines.append(
            f"excluded_from_stats_amd_above_{AMD_ALERT_THRESHOLD} (pairs): {n_high}"
        )

        all_summary = summarize(all_dists)
        best_summary = summarize(best_of_n)
        results_lines.extend(
            format_summary("multi_eval_all_pairs", all_summary, n_gt * num_evals)
        )
        results_lines.append("")
        results_lines.extend(
            format_summary("multi_eval_best_of_n", best_summary, n_gt)
        )

    results_lines.append("")
    out_name_base = "eval_amd"
    if args.label == "":
        out_name = f"{out_name_base}_{args.eval_split}.txt"
    elif args.label in LABELS_OMIT_SPLIT or args.label.startswith("rruff"):
        out_name = f"{out_name_base}_{args.label}.txt"
    else:
        out_name = f"{out_name_base}_{args.label}_{args.eval_split}.txt"
    out_path = Path(args.root_path) / out_name
    scatter_png = out_path.with_name(out_path.stem + "_scatter.png")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(results_lines) + "\n")

    print(f"Wrote AMD stats to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", required=True, help="Model directory path.")
    parser.add_argument("--eval_split", default="test")
    parser.add_argument(
        "--label",
        type=str,
        default="",
        help='Evaluation label (e.g. "rruff_20"). Matches compute_metrics naming.',
    )
    parser.add_argument("--gt_file", default="", help="Optional CSV containing a 'cif' column.")
    parser.add_argument("--multi_eval", action="store_true")
    parser.add_argument("--n_neighbors", type=int, default=100, help="PDD neighbors for AMD.")
    parser.add_argument(
        "--amd_src",
        default="",
        help="Optional path to AMD package source (if import fails without it).",
    )
    args = parser.parse_args()
    main(args)