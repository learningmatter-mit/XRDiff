"""Re-decode atom types from recovered t=1 logits and re-run the paper's match evaluation.

Decoders (all on the saved final geometry, logits from recover_type_logits.py):
    saved     atom types stored in eval_diff_<name>.pt (reproduces eval_metrics_<name>.json)
    argmax    masked argmax of the recovered logits (what sample() does, on these logits)
    coverage  max-likelihood assignment s.t. every element of the conditioning set appears
              at least once (exact, via linear assignment)

Matching mirrors compute_metrics: Crystal() validity gating, StructureMatcher defaults
(stol=0.3, angle_tol=5, ltol=0.2), a pair matches iff get_rms_dist returns a value.
    1 attempt  (RecEval):      requires pred.valid and gt.valid
    n attempts (RecEvalBatch): sample matched if any attempt matches; requires pred.valid only
"""
import argparse
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from p_tqdm import p_map
from scipy.optimize import linear_sum_assignment

warnings.filterwarnings('ignore')
EVAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(EVAL_DIR.parent))
sys.path.insert(0, str(EVAL_DIR.parents[1]))
from compute_metrics import Crystal, get_crystal_array_list
from failures_analysis import _make_matcher, _rms_match_one
from eval_utils import load_data

METHODS = ('saved', 'argmax', 'coverage')


def _coverage(S):
    n, k = S.shape
    amax = S.argmax(1)
    if len(set(amax.tolist())) == k:
        return amax
    # k "required" columns (one per element) + n-k "free" columns scoring each atom's best element
    M = np.concatenate([S, np.repeat(S.max(1, keepdims=True), n - k, axis=1)], axis=1)
    r, c = linear_sum_assignment(M, maximize=True)
    out = amax.copy()
    req = c < k
    out[r[req]] = c[req]
    return out


def process(payload):
    attempt, idx, pred_arr, gt_arr, lp, require_gt_valid = payload
    gt = Crystal(gt_arr)
    elems = np.array(sorted(set(gt.atom_types.tolist())))
    S = np.nan_to_num(lp[:, elems - 1].astype(np.float64), neginf=-1e4)
    types = {'saved': pred_arr['atom_types'], 'argmax': elems[S.argmax(1)], 'coverage': elems[_coverage(S)]}

    matcher = _make_matcher({})
    gc = Counter(gt.atom_types.tolist())
    row, cache = {'attempt': attempt, 'index': idx, 'gt_valid': bool(gt.valid)}, {}
    for m in METHODS:
        key = tuple(types[m].tolist())
        if key not in cache:
            pred = Crystal({**pred_arr, 'atom_types': types[m].copy()})
            ok = pred.valid and (gt.valid or not require_gt_valid)
            pc = Counter(key)
            cache[key] = dict(matched=_rms_match_one(matcher, pred, gt, ok) is not None,
                              formula_exact=pc == gc, elem_set_same=set(pc) == set(gc),
                              pred_valid=bool(pred.valid))
        for f, v in cache[key].items():
            row[f'{m}_{f}'] = v
        row[f'{m}_changed'] = key != tuple(types['saved'].tolist())
    return row


def summarize(df, run_name, material_ids=None):
    """Attempt-level rows -> one summary row per decoder. Match = any attempt matches."""
    per_sample = df.groupby('index').agg(
        **{f'{m}_{f}': (f'{m}_{f}', 'max') for m in METHODS for f in ('matched', 'formula_exact')})
    rows = []
    for m in METHODS:
        matched, formula_any = per_sample[f'{m}_matched'], per_sample[f'{m}_formula_exact']
        row = dict(
            run=run_name, decoder=m, n=len(per_sample), n_evals=df['attempt'].nunique(),
            match_rate=matched.mean(),
            exact_formula_acc=df[f'{m}_formula_exact'].mean(),
            all_elements_present=df[f'{m}_elem_set_same'].mean(),
            pred_valid=df[f'{m}_pred_valid'].mean(),
            nonmatch_wrong_formula_frac=(~formula_any[~matched]).mean(),
            match_rate_given_right_formula=matched[formula_any].mean(),
            n_changed_vs_saved=int(df[f'{m}_changed'].sum()))
        if material_ids is not None:
            minerals = pd.Series([mid.split('__')[0] for mid in material_ids])
            row['no_dup_match_rate'] = matched.groupby(minerals.values[matched.index]).max().mean()
        rows.append(row)
    return pd.DataFrame(rows)


def main(args):
    run = Path(args.root_path).resolve()
    name = args.label or args.eval_split
    eval_path = str(run / f'eval_diff_{name}.pt')
    preds, gts = get_crystal_array_list(eval_path, batch_idx=-1)
    lp_all = torch.load(run / f'type_logits_{name}.pt')['log_probs'].float().numpy()
    if lp_all.ndim == 2:  # files written before multi-attempt support
        lp_all = lp_all[None]
    n_evals = len(preds)
    assert lp_all.shape[0] == n_evals
    require_gt_valid = n_evals == 1

    payloads = []
    for j in range(n_evals):
        ptr = np.concatenate([[0], np.cumsum([len(p['atom_types']) for p in preds[j]])])
        assert ptr[-1] == lp_all.shape[1]
        B = min(args.limit, len(gts)) if args.limit else len(gts)
        payloads += [(j, i, preds[j][i], gts[i], lp_all[j, ptr[i]:ptr[i + 1]], require_gt_valid)
                     for i in range(B)]
    df = pd.DataFrame(p_map(process, payloads, num_cpus=args.num_cpus))

    material_ids = load_data(eval_path).get('material_ids')
    suffix = f'_limit{args.limit}' if args.limit else ''
    df.to_csv(run / f'type_decoding_per_sample_{name}{suffix}.csv', index=False)
    summary = summarize(df, args.name or f'{run.parent.name}/{run.name}', material_ids)
    summary.to_csv(run / f'type_decoding_summary_{name}{suffix}.csv', index=False)
    print(summary.to_string(index=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root_path', required=True)
    p.add_argument('--eval_split', default='test')
    p.add_argument('--label', default='', help='e.g. rruff_pv, rruff_pv_n_evals_10 (overrides eval_split)')
    p.add_argument('--name', default='')
    p.add_argument('--num_cpus', type=int, default=16)
    p.add_argument('--limit', type=int, default=0)
    main(p.parse_args())
