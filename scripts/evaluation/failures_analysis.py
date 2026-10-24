"""
Characterize samples where predicted and ground-truth structures match,
using the same StructureMatcher criterion as compute_metrics / RecEval.

For each ground-truth structure the script records the number of sites, the unit-cell
volume, and the international space-group number (1–230). Figures show how matches
are distributed with respect to those quantities.
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter

import numpy as np
from scipy import stats as _scipy_stats
import pandas as pd
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from tqdm import tqdm

import sys

script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir))

from compute_metrics import (
    LABELS_OMIT_SPLIT,
    Crystal,
    get_file_paths,
    get_crystal_array_list,
    get_gt_crys_ori,
    _comp_key, _build_comp_to_gt_indices,
)
from p_tqdm import p_map

_COLOR_TRAIN   = "#C5CBD3"  # light gray — training-set bin counts
_COLOR_MATCH   = "#2F5D8A"  # deep steel blue — eval match rate
_COLOR_METRE   = "#8DB3D6"  # light steel blue — METRe match rate
_COLOR_GRID    = "#D9DCE0"
_BAR_EDGE      = dict(edgecolor="white", linewidth=1.0)


def _style_grid(ax):
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", color=_COLOR_GRID, linewidth=0.9)

# Crystal systems in increasing order of symmetry (by SG range)
_CRYSTAL_SYSTEMS_ORDERED = [
    "Triclinic",    # SG   1–2
    "Monoclinic",   # SG   3–15
    "Orthorhombic", # SG  16–74
    "Tetragonal",   # SG  75–142
    "Trigonal",     # SG 143–167
    "Hexagonal",    # SG 168–194
    "Cubic",        # SG 195–230
]
# Standard crystallographic abbreviations (caption: Tric.=Triclinic, Mono.=Monoclinic,
# Orth.=Orthorhombic, Tetr.=Tetragonal, Trig.=Trigonal, Hex.=Hexagonal, Cub.=Cubic)
_CRYSTAL_SYSTEM_SHORT = {
    "Triclinic":    "Tric.",
    "Monoclinic":   "Mono.",
    "Orthorhombic": "Orth.",
    "Tetragonal":   "Tetr.",
    "Trigonal":     "Trig.",
    "Hexagonal":    "Hex.",
    "Cubic":        "Cub.",
}


def _sg_to_crystal_system(sg: float) -> str:
    n = int(sg)
    if n <= 2:   return "Triclinic"
    if n <= 15:  return "Monoclinic"
    if n <= 74:  return "Orthorhombic"
    if n <= 142: return "Tetragonal"
    if n <= 167: return "Trigonal"
    if n <= 194: return "Hexagonal"
    return "Cubic"

warnings.filterwarnings(
    "ignore",
    message="The version_base parameter is not specified.*",
    category=UserWarning)


def _parse_matcher_kwargs(raw):
    if raw is None:
        return {}
    if isinstance(raw, str):
        return json.loads(raw) if raw.strip() else {}
    return raw


def _make_matcher(structurematcher_kwargs):
    k = structurematcher_kwargs or {}
    stol = k.get("stol", 0.3)
    angle_tol = k.get("angle_tol", 5)
    ltol = k.get("ltol", 0.2)
    return StructureMatcher(stol=stol, angle_tol=angle_tol, ltol=ltol)


def _rms_match_one(matcher, pred, gt, is_valid):
    """Mirror compute_metrics.RecEval.process_one: match iff RMS returned."""
    if not is_valid:
        return None
    try:
        rms = matcher.get_rms_dist(pred.structure, gt.structure)
        return None if rms is None else rms[0]
    except Exception:
        return None


def gt_descriptors(gt_cry: Crystal, symprec: float) -> dict:
    """Properties from ground-truth structure (when constructible)."""
    out = {"n_sites": np.nan, "volume": np.nan, "spacegroup_num": np.nan}
    if not gt_cry.constructed:
        return out
    s = gt_cry.structure
    out["n_sites"] = float(s.num_sites)
    out["volume"] = float(s.volume)
    try:
        n = SpacegroupAnalyzer(s, symprec=symprec).get_space_group_number()
        out["spacegroup_num"] = float(n)
    except Exception:
        pass
    return out


def _parallel_row_single(payload):
    """Module-level for pickling. Returns one dataframe row dict."""
    idx, structurematcher_kwargs, symprec, pred, gt = payload
    matcher = _make_matcher(structurematcher_kwargs)
    r = _rms_match_one(matcher, pred, gt, pred.valid and gt.valid)
    d = gt_descriptors(gt, symprec)
    d["index"] = idx
    d["matched"] = r is not None
    return d


def _parallel_row_multi(payload):
    """Multi-eval: first successful pred wins (same as RecEvalBatch)."""
    idx, structurematcher_kwargs, symprec, preds, gt = payload
    matcher = _make_matcher(structurematcher_kwargs)
    ok = False
    for pred in preds:
        r = _rms_match_one(matcher, pred, gt, pred.valid)
        if r is not None:
            ok = True
            break
    d = gt_descriptors(gt, symprec)
    d["index"] = idx
    d["matched"] = ok
    return d

def _parallel_row_single_metre(payload):
    """Metre single-eval: pred[i] matches any GT with the same composition."""
    idx, structurematcher_kwargs, symprec, pred, gt_orig, gt_comp_list = payload
    matcher = _make_matcher(structurematcher_kwargs)
    matched = False
    if pred.valid:
        for gt in gt_comp_list:
            r = _rms_match_one(matcher, pred, gt, pred.valid and gt.valid)
            if r is not None:
                matched = True
                break
    d = gt_descriptors(gt_orig, symprec)
    d["index"] = idx
    d["matched"] = matched
    return d

def _parallel_row_multi_metre(payload):
    """Metre multi-eval: any candidate pred matches any GT with the same composition."""
    idx, structurematcher_kwargs, symprec, preds, gt_orig, gt_comp_list = payload
    matcher = _make_matcher(structurematcher_kwargs)
    matched = False
    for pred in preds:
        if not pred.valid:
            continue
        for gt in gt_comp_list:
            r = _rms_match_one(matcher, pred, gt, pred.valid and gt.valid)
            if r is not None:
                matched = True
                break
        if matched:
            break
    d = gt_descriptors(gt_orig, symprec)
    d["index"] = idx
    d["matched"] = matched
    return d

def run_analysis_rows(
    pred_crys,
    gt_crys,
    pred_batches,
    multi_eval: bool,
    structurematcher_kwargs,
    symprec: float,
    num_workers: int,
    use_metre: bool = False,
):
    """Structure matching and ground-truth descriptors in a single pass over the dataset.

    num_workers: use 0 for single-threaded execution, -1 for all CPUs, or a positive integer
    for that many worker processes.
    """
    n_pool = (os.cpu_count() or 1) if num_workers < 0 else num_workers
    use_parallel = num_workers != 0

    if use_metre:
        comp_to_gt = _build_comp_to_gt_indices(gt_crys)

    if not multi_eval:
        n = len(pred_crys)
        if use_metre:
            payloads = [
                (i, structurematcher_kwargs, symprec, pred_crys[i], gt_crys[i],
                 [gt_crys[j] for j in comp_to_gt.get(_comp_key(gt_crys[i]), [])])
                for i in range(n)
            ]
            row_fn = _parallel_row_single_metre
            desc = "Structure matching (metre, one-to-one)"
        else:
            payloads = [
                (i, structurematcher_kwargs, symprec, pred_crys[i], gt_crys[i])
                for i in range(n)
            ]
            row_fn = _parallel_row_single
            desc = "Structure matching (one-to-one)"
        if not use_parallel:
            rows = []
            for p in tqdm(payloads, desc=desc):
                rows.append(row_fn(p))
            return rows
        return p_map(row_fn, payloads, num_cpus=n_pool)

    n_samples = len(pred_batches[0])
    preds_per_i = [
        tuple(pred_batches[j][i] for j in range(len(pred_batches)))
        for i in range(n_samples)
    ]
    if use_metre:
        payloads = [
            (i, structurematcher_kwargs, symprec, preds_per_i[i], gt_crys[i],
             [gt_crys[j] for j in comp_to_gt.get(_comp_key(gt_crys[i]), [])])
            for i in range(n_samples)
        ]
        row_fn = _parallel_row_multi_metre
        desc = "Structure matching (metre, multiple samples)"
    else:
        payloads = [
            (i, structurematcher_kwargs, symprec, preds_per_i[i], gt_crys[i])
            for i in range(n_samples)
        ]
        row_fn = _parallel_row_multi
        desc = "Structure matching (multiple samples)"
    if not use_parallel:
        rows = []
        for p in tqdm(payloads, desc=desc):
            rows.append(row_fn(p))
        return rows
    return p_map(row_fn, payloads, num_cpus=n_pool)


def _apply_publication_font_settings():
    """Larger type for print figures (axis labels, ticks, legend)."""
    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.labelsize": 28,
            "axes.titlesize": 28,
            "xtick.labelsize": 22,
            "ytick.labelsize": 22,
            "legend.fontsize": 30,
            "legend.title_fontsize": 28,
        }
    )


def _plot_rate_on_ax(ax, centers, match_counts, totals, log_x=False, fit=True, color=_COLOR_MATCH):
    """Scatter of per-bin match rate + optional linear fit with confidence band, plotted directly on ax."""
    ax.set_ylim(0, 1)
    ax.spines["top"].set_visible(False)
    _style_grid(ax)

    has_data = totals > 0
    if not has_data.any():
        return

    xc = centers[has_data]
    rates = match_counts[has_data] / totals[has_data]
    x_fit = np.log10(xc) if log_x else xc.copy()

    ax.scatter(xc, rates, color=color, zorder=5, s=55, edgecolors="white", linewidths=1.0)

    if fit and len(xc) >= 2:
        slope, intercept, _, _, _ = _scipy_stats.linregress(x_fit, rates)
        x_line = np.linspace(x_fit.min(), x_fit.max(), 300)
        y_line = slope * x_line + intercept

        n = len(x_fit)
        x_mean = x_fit.mean()
        ssx = np.sum((x_fit - x_mean) ** 2)
        residuals = rates - (slope * x_fit + intercept)
        s = np.sqrt(np.sum(residuals ** 2) / max(n - 2, 1))
        se_pred = s * np.sqrt(1.0 / n + (x_line - x_mean) ** 2 / max(ssx, 1e-12))

        x_plot = 10 ** x_line if log_x else x_line
        ax.plot(x_plot, np.clip(y_line, 0, 1),
                color=color, alpha=0.9, linewidth=2.2, zorder=4)
        ax.fill_between(
            x_plot,
            np.clip(y_line - se_pred, 0, 1),
            np.clip(y_line + se_pred, 0, 1),
            color=color, alpha=0.14, linewidth=0, zorder=3,
        )


def _plot_bar_rate_on_ax(ax, positions, match_counts, totals, color=_COLOR_MATCH):
    """Bar chart of per-category match rate, plotted directly on ax."""
    ax.set_ylim(0, 1)
    ax.spines["top"].set_visible(False)
    _style_grid(ax)
    has_data = totals > 0
    rates = np.where(has_data, match_counts / np.where(has_data, totals, 1.0), 0.0)
    ax.bar(positions, rates, width=0.78, align="center", color=color, **_BAR_EDGE)


def _aggregate_discrete_two_panel(
    x_eval, match_mask, ax_top, ax_bot,
    x_train=None, max_unique_x: int = 80, x_max_cap: float | None = None, fit: bool = True,
    ax_bot2=None, x_eval2=None, match_mask2=None, show_scale_label: bool = True,
):
    """Top panel: training-set bin counts. Bottom panel: eval match rate.

    Returns max training-set bar height.
    """
    x_ev = np.asarray(x_eval, dtype=float)
    match_mask = np.asarray(match_mask, dtype=bool)
    valid_ev = np.isfinite(x_ev)
    x_ev, match_mask = x_ev[valid_ev], match_mask[valid_ev]
    if x_max_cap is not None:
        cap = x_ev <= float(x_max_cap)
        x_ev, match_mask = x_ev[cap], match_mask[cap]

    # Source for bin definitions: training data if available, else eval data
    if x_train is not None:
        x_src = np.asarray(x_train, dtype=float)
        x_src = x_src[np.isfinite(x_src)]
        if x_max_cap is not None:
            x_src = x_src[x_src <= float(x_max_cap)]
    else:
        x_src = x_ev

    ax_top.set_ylabel("Training count")
    _style_grid(ax_top)
    ax_bot.set_ylabel("Match rate")

    if x_src.size == 0:
        _plot_rate_on_ax(ax_bot, np.array([]), np.array([]), np.array([]))
        return 0.0

    edges = None
    uniq = np.unique(x_src)
    if len(uniq) <= max_unique_x:
        centers = uniq
        spacing = np.min(np.diff(np.sort(centers))) if len(centers) > 1 else 1.0
        train_totals = np.array([(x_src == c).sum() for c in centers], dtype=float)
        eval_totals  = np.array([(x_ev  == c).sum() for c in centers], dtype=float)
        match_counts = np.array([(match_mask & (x_ev == c)).sum() for c in centers], dtype=float)
    else:
        nbin = min(40, max(10, int(np.sqrt(len(x_src)))))
        lo, hi = x_src.min(), x_src.max()
        if lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
        edges = np.linspace(lo, hi, nbin + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        spacing = centers[1] - centers[0] if len(centers) > 1 else 1.0
        train_totals, eval_totals, match_counts = [], [], []
        for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            is_last = (i == len(edges) - 2)
            m_tr = (x_src >= a) & (x_src <= b if is_last else x_src < b)
            m_ev = (x_ev  >= a) & (x_ev  <= b if is_last else x_ev  < b)
            train_totals.append(m_tr.sum())
            eval_totals.append(m_ev.sum())
            match_counts.append((match_mask & m_ev).sum())
        train_totals = np.array(train_totals, dtype=float)
        eval_totals  = np.array(eval_totals,  dtype=float)
        match_counts = np.array(match_counts, dtype=float)

    w = 0.9 * spacing
    ax_top.bar(centers - w / 2, train_totals / 1e3, width=w, align="edge", color=_COLOR_TRAIN, **_BAR_EDGE)
    if show_scale_label:
        ax_top.text(0, 1.02, r"$\times10^3$", transform=ax_top.transAxes, ha="left", va="bottom")
    _plot_rate_on_ax(ax_bot, centers, match_counts, eval_totals, fit=fit, color=_COLOR_MATCH)

    if ax_bot2 is not None and x_eval2 is not None and match_mask2 is not None:
        x_ev2 = np.asarray(x_eval2, dtype=float)
        mm2 = np.asarray(match_mask2, dtype=bool)
        valid2 = np.isfinite(x_ev2)
        x_ev2, mm2 = x_ev2[valid2], mm2[valid2]
        if x_max_cap is not None:
            cap2 = x_ev2 <= float(x_max_cap)
            x_ev2, mm2 = x_ev2[cap2], mm2[cap2]
        ax_bot2.set_ylabel("METRe")
        if edges is None:
            ev2_totals  = np.array([(x_ev2 == c).sum() for c in centers], dtype=float)
            mc2         = np.array([(mm2 & (x_ev2 == c)).sum() for c in centers], dtype=float)
        else:
            ev2_totals, mc2 = [], []
            for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
                is_last = (i == len(edges) - 2)
                m2 = (x_ev2 >= a) & (x_ev2 <= b if is_last else x_ev2 < b)
                ev2_totals.append(m2.sum())
                mc2.append((mm2 & m2).sum())
            ev2_totals = np.array(ev2_totals, dtype=float)
            mc2        = np.array(mc2,        dtype=float)
        _plot_rate_on_ax(ax_bot2, centers, mc2, ev2_totals, fit=fit, color=_COLOR_METRE)

    return float(np.max(train_totals)) / 1e3 if train_totals.size else 0.0


def _aggregate_log_volume_two_panel(vol_eval, match_mask, ax_top, ax_bot,
                                     vol_train=None, fit: bool = False,
                                     ax_bot2=None, vol_eval2=None, match_mask2=None,
                                     show_scale_label: bool = True):
    """Top panel: training-set volume histogram. Bottom panel: eval match rate.

    Returns max training-set bar height.
    """
    v_ev = np.asarray(vol_eval, dtype=float)
    ok_ev = np.isfinite(v_ev) & (v_ev > 0)
    v_ev, match_mask = v_ev[ok_ev], np.asarray(match_mask, dtype=bool)[ok_ev]
    logv_ev = np.log10(v_ev)

    if vol_train is not None:
        v_tr = np.asarray(vol_train, dtype=float)
        logv_src = np.log10(v_tr[np.isfinite(v_tr) & (v_tr > 0)])
    else:
        logv_src = logv_ev

    nbin = 25
    edges = np.linspace(np.percentile(logv_src, 1), np.percentile(logv_src, 99), nbin + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    vol_centers = 10 ** centers
    vol_widths  = 10 ** edges[1:] - 10 ** edges[:-1]

    train_totals, eval_totals, match_counts = [], [], []
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        is_last = (i == len(edges) - 2)
        m_tr = (logv_src >= a) & (logv_src <= b if is_last else logv_src < b)
        m_ev = (logv_ev  >= a) & (logv_ev  <= b if is_last else logv_ev  < b)
        train_totals.append(m_tr.sum())
        eval_totals.append(m_ev.sum())
        match_counts.append((match_mask & m_ev).sum())
    train_totals = np.asarray(train_totals, dtype=float)
    eval_totals  = np.asarray(eval_totals,  dtype=float)
    match_counts = np.asarray(match_counts, dtype=float)

    ax_top.bar(10 ** edges[:-1], train_totals / 1e3, width=vol_widths, align="edge",
               color=_COLOR_TRAIN, **_BAR_EDGE)
    if show_scale_label:
        ax_top.text(0, 1.02, r"$\times10^3$", transform=ax_top.transAxes, ha="left", va="bottom")
    def _apply_vol_ticks(ax):
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator([20, 50, 100, 200, 500, 1000]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_locator(LogLocator(subs=np.arange(1, 10), numticks=100))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="x", which="minor", length=3)
        pad = 0.02 * (edges[-1] - edges[0])
        ax.set_xlim(10 ** (edges[0] - pad), 10 ** (edges[-1] + pad))

    _apply_vol_ticks(ax_top)
    ax_top.set_ylabel("Training count")
    _style_grid(ax_top)

    _apply_vol_ticks(ax_bot)
    ax_bot.set_ylabel("Match rate")
    _plot_rate_on_ax(ax_bot, vol_centers, match_counts, eval_totals, log_x=True, fit=fit, color=_COLOR_MATCH)

    if ax_bot2 is not None and vol_eval2 is not None and match_mask2 is not None:
        v_ev2 = np.asarray(vol_eval2, dtype=float)
        ok2 = np.isfinite(v_ev2) & (v_ev2 > 0)
        v_ev2 = v_ev2[ok2]
        mm2 = np.asarray(match_mask2, dtype=bool)[ok2]
        logv_ev2 = np.log10(v_ev2)
        ev2_totals, mc2 = [], []
        for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            is_last = (i == len(edges) - 2)
            m2 = (logv_ev2 >= a) & (logv_ev2 <= b if is_last else logv_ev2 < b)
            ev2_totals.append(m2.sum())
            mc2.append((mm2 & m2).sum())
        ev2_totals = np.asarray(ev2_totals, dtype=float)
        mc2        = np.asarray(mc2,        dtype=float)
        _apply_vol_ticks(ax_bot2)
        ax_bot2.set_ylabel("METRe")
        _plot_rate_on_ax(ax_bot2, vol_centers, mc2, ev2_totals, log_x=True, fit=fit, color=_COLOR_METRE)

    return float(train_totals.max()) / 1e3 if train_totals.size else 0.0


def _aggregate_crystal_system_two_panel(cs_eval, match_mask, ax_top, ax_bot, cs_train=None,
                                         ax_bot2=None, cs_eval2=None, match_mask2=None,
                                         show_scale_label: bool = True):
    """Top panel: training-set counts per crystal system. Bottom panel: eval match rate.

    Returns max training-set bar height.
    """
    cs_ev = np.asarray(cs_eval, dtype=object)
    match_mask = np.asarray(match_mask, dtype=bool)
    cs_src = np.asarray(cs_train, dtype=object) if cs_train is not None else cs_ev

    positions = np.arange(len(_CRYSTAL_SYSTEMS_ORDERED))
    train_totals = np.array([(cs_src == s).sum() for s in _CRYSTAL_SYSTEMS_ORDERED], dtype=float)
    eval_totals  = np.array([(cs_ev  == s).sum() for s in _CRYSTAL_SYSTEMS_ORDERED], dtype=float)
    match_counts = np.array(
        [(match_mask & (cs_ev == s)).sum() for s in _CRYSTAL_SYSTEMS_ORDERED], dtype=float
    )
    short_labels = [_CRYSTAL_SYSTEM_SHORT[s] for s in _CRYSTAL_SYSTEMS_ORDERED]

    xlim = (positions[0] - 0.55, positions[-1] + 0.55)

    ax_top.bar(positions, train_totals / 1e3, width=0.78, align="center", color=_COLOR_TRAIN, **_BAR_EDGE)
    if show_scale_label:
        ax_top.text(0, 1.02, r"$\times10^3$", transform=ax_top.transAxes, ha="left", va="bottom")
    ax_top.set_xticks(positions)
    ax_top.set_xticklabels(short_labels, rotation=45, ha="right")
    ax_top.set_ylabel("Training count")
    ax_top.set_xlim(xlim)
    _style_grid(ax_top)

    ax_bot.set_xticks(positions)
    ax_bot.set_xticklabels(short_labels, rotation=45, ha="right")
    ax_bot.set_ylabel("Match rate")
    ax_bot.set_xlim(xlim)
    _plot_bar_rate_on_ax(ax_bot, positions, match_counts, eval_totals, color=_COLOR_MATCH)

    if ax_bot2 is not None and cs_eval2 is not None and match_mask2 is not None:
        cs_ev2 = np.asarray(cs_eval2, dtype=object)
        mm2 = np.asarray(match_mask2, dtype=bool)
        ev2_totals  = np.array([(cs_ev2 == s).sum() for s in _CRYSTAL_SYSTEMS_ORDERED], dtype=float)
        mc2         = np.array([(mm2 & (cs_ev2 == s)).sum() for s in _CRYSTAL_SYSTEMS_ORDERED], dtype=float)
        ax_bot2.set_xticks(positions)
        ax_bot2.set_xticklabels(short_labels, rotation=45, ha="right")
        ax_bot2.set_ylabel("METRe")
        ax_bot2.set_xlim(xlim)
        _plot_bar_rate_on_ax(ax_bot2, positions, mc2, ev2_totals, color=_COLOR_METRE)

    return float(train_totals.max()) / 1e3 if train_totals.size else 0.0


def _shared_count_ylim(axes, max_bar_heights: list[float]) -> None:
    """Use the same left-axis upper limit for match counts on every panel."""
    peak = max(max_bar_heights) if max_bar_heights else 0.0
    if peak <= 0:
        top = 1.0
    else:
        top = float(np.ceil(peak * 1.06))
        if top <= peak:
            top = peak + 1.0
    for ax in axes:
        ax.set_ylim(0.0, top)


def _eval_label_omits_split(label: str) -> bool:
    """Same rule as compute_metrics.get_file_paths: RRUFF / custom labels have no split in the .pt name."""
    if not label:
        return False
    if label in LABELS_OMIT_SPLIT:
        return True
    return label.startswith("rruff")


def _infer_max_sites_x(label: str, root_path: str = "", train_pt: str = "") -> float | None:
    """Optional right-hand limit for the site-count axis (None keeps data-driven limits)."""
    if label:
        low = label.lower()
        if low.startswith("rruff_"):
            try:
                return float(low.split("_", 1)[1])
            except (IndexError, ValueError):
                pass
    s = f"{label or ''} {root_path or ''} {train_pt or ''}".lower()
    if "oqmd" in s:
        return 40.0
    if "alex" in s or "mp_20" in s or "mp20" in s:
        return 20.0
    return None


def _matched_bool_series(df_col: pd.Series) -> np.ndarray:
    """Robust bool array for CSV reload (avoids bool('False') == True)."""
    if pd.api.types.is_bool_dtype(df_col):
        return df_col.to_numpy(dtype=bool)
    return df_col.astype(str).str.lower().isin(("true", "1", "1.0")).to_numpy()



def _load_train_pt_descriptors(pt_path: str, symprec: float = 0.01) -> pd.DataFrame:
    """Extract n_sites, volume, spacegroup_num from a training .pt file (list of dicts).

    Builds a pymatgen Structure from graph_arrays and uses SpacegroupAnalyzer to
    determine the space group (the stored spacegroup key can be degenerate).
    """
    from pymatgen.core.lattice import Lattice
    from pymatgen.core.structure import Structure

    data = torch.load(pt_path, map_location="cpu")
    rows = []
    for item in data:
        ga = item["graph_arrays"]
        frac_coords, atom_types = ga[0], ga[1]
        n_sites = int(ga[6])
        vol = float("nan")
        sg = float("nan")
        try:
            a, b, c = float(ga[2][0]), float(ga[2][1]), float(ga[2][2])
            al, be, gm = float(ga[3][0]), float(ga[3][1]), float(ga[3][2])
            lattice = Lattice.from_parameters(a, b, c, al, be, gm)
            vol = lattice.volume
            structure = Structure(
                lattice=lattice,
                species=atom_types,
                coords=frac_coords,
                coords_are_cartesian=False,
            )
            sg = float(SpacegroupAnalyzer(structure, symprec=symprec).get_space_group_number())
        except Exception:
            pass
        rows.append({
            "n_sites": float(n_sites),
            "volume": float(vol),
            "spacegroup_num": sg,
        })
    return pd.DataFrame(rows)


def _load_eval_df(root_path: str, args, struct_kwargs: dict, use_metre: bool = False):
    """Load (from cache or compute) the per-sample match dataframe for root_path.

    Returns (df, csv_path, from_cache).
    """
    out_dir = Path(args.out_dir) if args.out_dir else Path(root_path)
    out_dir = out_dir / "failures_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "fails"
    if args.label:
        stem += f"_{args.label}"
    if args.eval_split and not _eval_label_omits_split(args.label):
        stem += f"_{args.eval_split}"
    if use_metre:
        stem += "_metre"
    csv_path = out_dir / f"{stem}.csv"

    if csv_path.is_file() and not args.force:
        try:
            cached = pd.read_csv(csv_path)
            need = {"matched", "index", "n_sites", "volume", "spacegroup_num"}
            if need.issubset(cached.columns) and len(cached) > 0:
                df = cached.sort_values("index").reset_index(drop=True)
                print(f"Loaded existing per-sample table (skipping structure matching): {csv_path}")
                return df, csv_path, True
        except Exception as exc:
            print(f"Could not read {csv_path} ({exc}); running full analysis.")

    recon_file_path = get_file_paths(root_path, "diff", args.eval_split, args.label)
    batch_idx = -1 if args.multi_eval else 0
    crys_array_list, true_crystal_array_list = get_crystal_array_list(
        recon_file_path, batch_idx=batch_idx)

    if args.gt_file:
        csv = pd.read_csv(args.gt_file)
        gt_crys = p_map(get_gt_crys_ori, csv["cif"])
    else:
        gt_crys = p_map(lambda x: Crystal(x), true_crystal_array_list)

    pred_batches = None
    if not args.multi_eval:
        pred_crys = p_map(lambda x: Crystal(x), crys_array_list)
    else:
        pred_crys = None
        pred_batches = []
        for i in range(len(crys_array_list)):
            pred_batches.append(p_map(lambda x: Crystal(x), crys_array_list[i]))

    rows = run_analysis_rows(
        pred_crys, gt_crys, pred_batches, args.multi_eval,
        struct_kwargs, args.symprec, args.num_workers,
        use_metre=use_metre,
    )
    df = pd.DataFrame(rows).sort_values("index").reset_index(drop=True)
    df.to_csv(csv_path, index=False)
    return df, csv_path, False


def main(args):
    struct_kwargs = _parse_matcher_kwargs(args.structurematcher_kwargs)
    n_pool = (os.cpu_count() or 1) if args.num_workers < 0 else args.num_workers

    df, csv_path, from_cache = _load_eval_df(args.root_path, args, struct_kwargs)
    out_dir = csv_path.parent
    stem = csv_path.stem

    match_mask = _matched_bool_series(df["matched"])

    df2 = match_mask2 = crystal_systems2 = None
    if args.root_path2:
        df2, _, _ = _load_eval_df(args.root_path2, args, struct_kwargs, use_metre=True)
        match_mask2 = _matched_bool_series(df2["matched"])
        crystal_systems2 = df2["spacegroup_num"].apply(
            lambda sg: _sg_to_crystal_system(sg) if pd.notna(sg) and sg >= 1 else ""
        ).to_numpy()

    _apply_publication_font_settings()

    max_sites = args.max_sites_x
    if max_sites is None:
        max_sites = _infer_max_sites_x(args.label, args.root_path, args.train_pt)

    crystal_systems = df["spacegroup_num"].apply(
        lambda sg: _sg_to_crystal_system(sg) if pd.notna(sg) and sg >= 1 else ""
    ).to_numpy()

    # --- Load training descriptors for the counts (top) panel ---
    train_pt_path = Path(args.train_pt)
    if not train_pt_path.is_file():
        raise FileNotFoundError(
            f"--train_pt {args.train_pt!r} not found. "
            "Provide the training .pt file (list of dicts with graph_arrays and spacegroup keys)."
        )
    df_train = _load_train_pt_descriptors(str(train_pt_path), symprec=args.symprec)
    print(f"Loaded training descriptors: {train_pt_path} ({len(df_train)} samples)")

    n_sites_train = df_train["n_sites"].values
    vol_train     = df_train["volume"].values
    sg_train_vals = df_train["spacegroup_num"].values
    cs_train = pd.Series(sg_train_vals).apply(
        lambda sg: _sg_to_crystal_system(sg) if pd.notna(sg) and sg >= 1 else ""
    ).to_numpy()

    n_rows = 3 if df2 is not None else 2

    # --- Figure 1: sites + volume + crystal system ---
    fig_height = 9.0 if n_rows == 2 else 13.5
    fig, axes = plt.subplots(
        n_rows, 3, figsize=(19.0, fig_height),
        gridspec_kw={"width_ratios": [1.25, 1.0, 1.25]},
    )

    ax_bot2_sites = axes[2, 0] if n_rows == 3 else None
    ax_bot2_vol   = axes[2, 1] if n_rows == 3 else None
    ax_bot2_cs    = axes[2, 2] if n_rows == 3 else None

    peak_sites = _aggregate_discrete_two_panel(
        df["n_sites"].values, match_mask, axes[0, 0], axes[1, 0],
        x_train=n_sites_train, x_max_cap=max_sites, fit=True,
        ax_bot2=ax_bot2_sites,
        x_eval2=df2["n_sites"].values if df2 is not None else None,
        match_mask2=match_mask2,
    )
    axes[n_rows - 1, 0].set_xlabel("Number of atoms in unit cell")
    for row in range(n_rows):
        axes[row, 0].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    if max_sites is not None:
        for row in range(n_rows):
            axes[row, 0].set_xlim(0.5, max_sites + 0.5)

    peak_vol = _aggregate_log_volume_two_panel(
        df["volume"].values, match_mask, axes[0, 1], axes[1, 1],
        vol_train=vol_train, fit=False,
        ax_bot2=ax_bot2_vol,
        vol_eval2=df2["volume"].values if df2 is not None else None,
        match_mask2=match_mask2,
        show_scale_label=False,
    )
    axes[n_rows - 1, 1].set_xlabel(r"Unit-cell volume ($\mathrm{\AA}^3$)")

    peak_cs = _aggregate_crystal_system_two_panel(
        crystal_systems, match_mask, axes[0, 2], axes[1, 2],
        cs_train=cs_train,
        ax_bot2=ax_bot2_cs,
        cs_eval2=crystal_systems2,
        match_mask2=match_mask2,
        show_scale_label=False,
    )
    axes[n_rows - 1, 2].set_xlabel("Crystal system")

    _shared_count_ylim(list(axes[0, :]), [peak_sites, peak_vol, peak_cs])

    # Spine and tick styling
    for ax in axes.flat:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for row in range(n_rows - 1):
        for ax in axes[row, :]:
            ax.tick_params(axis="x", which="both", labelbottom=False)
    for col in (1, 2):
        for row in range(n_rows):
            axes[row, col].set_ylabel("")
            axes[row, col].tick_params(axis="y", left=False, labelleft=False)
            axes[row, col].spines["left"].set_visible(False)

    hspace = 0.18 if n_rows == 3 else 0.22
    bottom = 0.07 if n_rows == 3 else 0.10
    fig.subplots_adjust(top=0.90, bottom=bottom, left=0.09, right=0.97, wspace=0.06, hspace=hspace)
    fig.align_ylabels(list(axes[:, 0]))
    fig.align_xlabels(list(axes[n_rows - 1, :]))
    png_path = out_dir / f"{stem}.png"
    fig.savefig(png_path, dpi=220, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)

    # --- Figure 2: space group ---
    sg_height = 8.0 if n_rows == 2 else 11.0
    fig_sg, axes_sg = plt.subplots(n_rows, 1, figsize=(7.5, sg_height), sharex=True)

    peak_sg = _aggregate_discrete_two_panel(
        df["spacegroup_num"].values, match_mask, axes_sg[0], axes_sg[1],
        x_train=sg_train_vals, max_unique_x=50, fit=False,
        ax_bot2=axes_sg[2] if n_rows == 3 else None,
        x_eval2=df2["spacegroup_num"].values if df2 is not None else None,
        match_mask2=match_mask2,
    )
    axes_sg[-1].set_xlabel("Space-group number")
    _shared_count_ylim([axes_sg[0]], [peak_sg])

    for ax in axes_sg:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig_sg.subplots_adjust(top=0.90, bottom=0.12, left=0.12, right=0.90, hspace=0.40)
    fig_sg.align_ylabels(list(axes_sg))
    png_path_sg = out_dir / f"{stem}_sg.png"
    fig_sg.savefig(png_path_sg, dpi=220, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig_sg)

    n_match = int(match_mask.sum())
    summary = {
        "n_structures": int(len(df)),
        "n_matches": n_match,
        "match_rate": float(match_mask.mean()),
        "per_sample_csv": str(csv_path),
        "figure_png": str(png_path),
        "figure_sg_png": str(png_path_sg),
        "structurematcher_kwargs": struct_kwargs,
        "num_workers": int(n_pool) if args.num_workers != 0 else 0,
        "from_cached_csv": from_cache,
    }
    with open(out_dir / f"{stem}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Summarize matching structure predictions versus ground-truth "
            "crystallographic descriptors (same matcher as compute_metrics)."
        )
    )
    parser.add_argument(
        "--root_path",
        required=True,
        help="Directory containing evaluation outputs (for example, a Hydra run folder).",
    )
    parser.add_argument(
        "--root_path2",
        default="",
        help=(
            "Optional second evaluation directory. When provided, a third (bottom) panel "
            "is added to each figure showing the match rate for this path."
        ),
    )
    parser.add_argument(
        "--eval_split",
        default="test",
        help=(
            "Split segment in eval_diff_{label}_{split}.pt for standard datasets. "
            "Ignored for the output file stem when --label is RRUFF (e.g. rruff_20) or "
            "other labels that omit split; see compute_metrics.get_file_paths."
        ),
    )
    parser.add_argument(
        "--label",
        type=str,
        default="",
        help=(
            "Label segment in the evaluation .pt name (e.g. rruff_20 for eval_diff_rruff_20.pt). "
            "Required for RRUFF runs because there is no split in the filename."
        ),
    )
    parser.add_argument(
        "--gt_file",
        default="",
        help="Optional CSV with a CIF column; if empty, ground truth is read from the evaluation tensor.",
    )
    parser.add_argument(
        "--multi_eval",
        action="store_true",
        help="Use when the reconstruction file stores multiple predictions per ground-truth structure.",
    )
    parser.add_argument(
        "--structurematcher_kwargs",
        default=None,
        type=lambda x: json.loads(x) if isinstance(x, str) else (x or {}),
        help=(
            "Keyword arguments for pymatgen StructureMatcher as a JSON object, "
            'for example {"stol":0.3,"angle_tol":5,"ltol":0.2}.'
        ),
    )
    parser.add_argument(
        "--symprec",
        type=float,
        default=0.01,
        help="Symmetry precision passed to SpacegroupAnalyzer on each ground-truth structure.",
    )
    parser.add_argument(
        "--out_dir",
        default="",
        help="Base directory for outputs; defaults to root_path if not set.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=-1,
        help=(
            "Number of parallel worker processes for matching and symmetry analysis. "
            "Use 0 for single-threaded execution, or -1 to use all available CPUs."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore an existing per-sample CSV and recompute matches from the reconstruction file.",
    )
    parser.add_argument(
        "--train_pt",
        required=True,
        help=(
            "Training .pt file (list of dicts with graph_arrays and spacegroup keys). "
            "Used for the counts (top) panel."
        ),
    )
    parser.add_argument(
        "--max_sites_x",
        type=float,
        default=None,
        help=(
            "Maximum value shown on the horizontal axis for site counts. "
            "If omitted, the script infers 40 when the label or path suggests OQMD, "
            "and 20 when it suggests Alex-MP-20 or Materials Project-20-style data."
        ),
    )
    args = parser.parse_args()
    if args.structurematcher_kwargs is None:
        args.structurematcher_kwargs = {}
    assert args.label != "multi" or args.multi_eval, "label 'multi' requires --multi_eval"
    main(args)
