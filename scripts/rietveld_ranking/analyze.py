"""Step 3: scores, ranking, evaluation, figure (runs in the repo env).

Reads manifest.csv (prepare.py) and work/*/result.json (refine.py) and writes,
into <run_dir>/rietveld_ranking/:
    per_candidate.csv, summary.json, ranking_figure.{pdf,png}

Ranking is done within each generation file (k = 1, 3, 10 are independent
draws; see README). The ground truth enters only through matched_to_gt (for
evaluation) and the separate GT sanity-check jobs.

    python scripts/rietveld_ranking/analyze.py --variant peaks [--n_samples 5]
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import dirs, load_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
WORST = np.inf
C = {'s1': '#2a78d6', 's2': '#eb6834', 's3': '#1baf7a', 'gray': '#8a8985',
     'ink': '#0b0b0b', 'ink2': '#52514e', 'grid': '#e6e5e0'}


# --------------------------------------------------------------------------- table
def load_results(out_dir, man):
    rows = []
    for r in man.itertuples():
        row = dict(sample_idx=r.sample_idx, sample_id=r.sample_id, input_variant=r.input_variant,
                   gen_file_k=r.gen_file_k, cand_idx=r.cand_idx, valid=r.valid,
                   invalid_reason=r.invalid_reason if isinstance(r.invalid_reason, str) else '',
                   spacegroup=r.spacegroup if isinstance(r.spacegroup, str) else '',
                   sg_number=r.sg_number, n_lattice_params=r.n_lattice_params,
                   matched_to_gt=r.matched_to_gt, job_id=r.job_id)
        res_path = out_dir / 'work' / r.job_id / 'result.json'
        res = json.load(open(res_path)) if (r.valid and res_path.exists()) else None
        if not r.valid:
            fail = 'invalid'
        elif res is None:
            fail = 'not_run'
        else:
            fail = res.get('fail_reason', '')
        st = (res or {}).get('stages', {})
        row['stuck_stages'] = ','.join((res or {}).get('stuck_stages', []))
        final = (res or {}).get('final_stage', 'profile')
        last = st.get(final) if not fail else None
        row['final_stage'] = final if last else ''
        row['extra_fail_reason'] = (res or {}).get('extra_fail_reason', '')
        row.update(fail_reason=fail, failed_stage=fail.split(':')[0] if fail not in ('', 'invalid', 'not_run') else '',
                   converged=bool(last and last['converged']),
                   n_refined_params=last['Nvars'] if last else np.nan,
                   Rwp=last['Rwp'] if last else WORST, GoF=last['GoF'] if last else np.nan,
                   R_Bragg=last['R_F'] if last else WORST, R_F2=last['R_F2'] if last else np.nan,
                   N=last['Nobs'] if last else np.nan, RSS=last['chisq'] if last else np.nan,
                   wall_time=(res or {}).get('main_wall_time', (res or {}).get('total_wall_time', np.nan)))
        row['AIC'] = (row['N'] * np.log(row['RSS'] / row['N']) + 2 * row['n_refined_params']) if last else WORST
        po = st.get('pref_orient')
        po_ok = last is not None and po is not None and not (res or {}).get('po_fail_reason')
        row.update(po_fail_reason='' if po_ok else ((res or {}).get('po_fail_reason') or fail or 'not_run'),
                   Rwp_po=po['Rwp'] if po_ok else WORST, R_Bragg_po=po['R_F'] if po_ok else WORST,
                   AIC_po=(po['Nobs'] * np.log(po['chisq'] / po['Nobs']) + 2 * po['Nvars']) if po_ok else WORST)
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- metrics
def pick(group, score):
    """Top-1 by lowest score; ties (incl. all-worst) broken by lowest cand_idx."""
    g = group.sort_values([score, 'cand_idx'], kind='mergesort')
    return g.iloc[0]


def per_sample_table(cands, score_cols):
    """One row per sample: coverage, random-pick expectation, ranked@1 per score."""
    out = []
    for sid, g in cands.groupby('sample_idx'):
        row = {'sample_idx': sid, 'coverage': g.matched_to_gt.any(), 'random': g.matched_to_gt.mean()}
        for s in score_cols:
            row[f'ranked_{s}'] = bool(pick(g, s).matched_to_gt)
            row[f'pick_{s}'] = int(pick(g, s).cand_idx)
        out.append(row)
    return pd.DataFrame(out).set_index('sample_idx')


def bootstrap(tab, cols, n, rng):
    idx = rng.integers(0, len(tab), size=(n, len(tab)))
    res = {}
    for c in cols:
        v = tab[c].to_numpy(float)
        bs = v[idx].mean(1)
        res[c] = {'value': float(v.mean()), 'ci95': [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    cov = tab['coverage'].to_numpy(float)[idx].mean(1)
    for c in cols:
        if c.startswith('ranked_'):
            v = tab[c].to_numpy(float)
            eff = np.divide(v[idx].mean(1), cov, out=np.full(n, np.nan), where=cov > 0)
            res[f'efficiency_{c[7:]}'] = {
                'value': float(v.mean() / tab['coverage'].mean()) if tab['coverage'].mean() > 0 else float('nan'),
                'ci95': [float(np.nanpercentile(eff, 2.5)), float(np.nanpercentile(eff, 97.5))]}
    return res


def top_n(cands, scores, ns, rng, n_boot):
    """Fraction of samples with >= 1 match among the n best-ranked candidates, per
    score, next to the random-subset expectation 1 - C(k-M, n)/C(k, n)."""
    from math import comb
    per = []
    for _, g in cands.groupby('sample_idx'):
        k, M = len(g), int(g.matched_to_gt.sum())
        row = {f'random_{n}': 1 - comb(k - M, n) / comb(k, n) for n in ns}
        for sc in scores:
            m = g.sort_values([sc, 'cand_idx'], kind='mergesort').matched_to_gt.to_numpy()
            first = int(np.argmax(m)) + 1 if M else 10 ** 9
            row.update({f'{sc}_{n}': float(first <= n) for n in ns})
        per.append(row)
    tab = pd.DataFrame(per)
    idx = rng.integers(0, len(tab), size=(n_boot, len(tab)))
    out = {}
    for c in tab.columns:
        v = tab[c].to_numpy(float)
        bs = v[idx].mean(1)
        out[c] = {'value': float(v.mean()), 'ci95': [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    return out


def gt_sanity(df, k_ref, rng, n_boot):
    cands = df[(df.input_variant != 'gt') & (df.gen_file_k == k_ref)]
    gt = df[df.input_variant == 'gt'].set_index('sample_idx')
    lower, ranks, excluded = [], [], 0
    for sid, g in cands.groupby('sample_idx'):
        g_rwp = gt.loc[sid, 'Rwp']
        nonmatch = g.loc[~g.matched_to_gt, 'Rwp'].to_numpy()
        if len(nonmatch) == 0:
            excluded += 1
        else:
            lower.append(bool(np.isfinite(g_rwp) and (g_rwp < nonmatch).all()))
        ranks.append(int(1 + (g.Rwp.to_numpy() < g_rwp).sum()) if np.isfinite(g_rwp) else len(g) + 1)
    lower, ranks = np.array(lower, float), np.array(ranks)
    bs = lower[rng.integers(0, len(lower), (n_boot, len(lower)))].mean(1)
    return {
        'k': k_ref,
        'frac_gt_below_all_nonmatching': {'value': float(lower.mean()),
                                          'ci95': [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                          'n_samples': int(len(lower)), 'n_excluded_all_candidates_match': excluded},
        'gt_rank_among_k_plus_gt': {'median': float(np.median(ranks)), 'mean': float(ranks.mean()),
                                    'frac_rank1': float((ranks == 1).mean()),
                                    'histogram': {int(r): int((ranks == r).sum()) for r in range(1, k_ref + 2)}},
        'n_gt_refinement_failed': int((~np.isfinite(gt.Rwp)).sum()),
    }


# --------------------------------------------------------------------------- figure
def style_ax(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(C['ink2'])
        ax.spines[s].set_linewidth(0.6)
    ax.tick_params(colors=C['ink2'], labelsize=7, width=0.6)
    ax.grid(axis='y', color=C['grid'], lw=0.5)
    ax.set_axisbelow(True)


def plot_fit(ax, npz, title, rwp, offset_frac=0.12):
    d = np.load(npz)
    x, yo, yc = d['x'], d['yobs'], d['ycalc']
    scale = yo.max()
    yo, yc = yo / scale, yc / scale
    off = -offset_frac - (yo - yc).max()
    ax.plot(x, yo, color=C['ink'], lw=0.5, label='observed')
    ax.plot(x, yc, color=C['s1'], lw=0.7, label='calculated')
    ax.plot(x, yo - yc + off, color=C['gray'], lw=0.5, label='difference')
    ax.set_title(title, fontsize=8, loc='left', color=C['ink'])
    ax.text(0.99, 0.92, f'$R_{{wp}}$ = {rwp:.1f}%', transform=ax.transAxes, ha='right', va='top',
            fontsize=7, color=C['ink'])
    ax.set_yticks([])
    style_ax(ax)
    ax.grid(False)


def make_figure(df, summaries, variants, out_dir, k_ref):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42})
    single = len(variants) == 1   # one input variant: (a) and (b) share the top row
    if single:
        fig = plt.figure(figsize=(7.2, 5.6))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1], hspace=0.6, wspace=0.28)
        b_slot, c_row = gs[0, 1], 1
    else:
        fig = plt.figure(figsize=(7.2, 7.6))
        gs = fig.add_gridspec(3, len(variants), height_ratios=[1, 1, 1.25], hspace=0.55, wspace=0.3)
        b_slot, c_row = gs[1, :], 2

    # (a) success rate vs k
    for vi, var in enumerate(variants):
        ax = fig.add_subplot(gs[0, vi])
        rates = summaries[var]['rates']
        ks = sorted(int(k) for k in rates)
        series = [('coverage', 'oracle coverage@k', C['s1'], '-'),
                  ('ranked_Rwp', 'ranked@1 ($R_{wp}$)', C['s2'], '-'),
                  ('ranked_R_Bragg', 'ranked@1 ($R_F$)', C['s3'], '-'),
                  ('random', 'random pick', C['gray'], '--')]
        for key, lab, col, ls in series:
            v = np.array([rates[str(k)][key]['value'] for k in ks])
            lo = np.array([rates[str(k)][key]['ci95'][0] for k in ks])
            hi = np.array([rates[str(k)][key]['ci95'][1] for k in ks])
            ax.fill_between(ks, lo, hi, color=col, alpha=0.12, lw=0)
            ax.plot(ks, v, ls, color=col, lw=2 if ls == '-' else 1.2, marker='o', ms=4, label=lab)
        ax.set_xscale('log')
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        ax.minorticks_off()
        ax.set_xlabel('candidates k', fontsize=8, color=C['ink2'])
        ax.set_ylabel('success rate', fontsize=8, color=C['ink2'])
        ax.set_ylim(0, None)
        ax.set_title(f'(a) {var} input', fontsize=9, loc='left', color=C['ink'])
        style_ax(ax)
        if vi == 0:
            ax.legend(fontsize=6.5, frameon=False, loc='upper left')

    # (b) Rwp distribution: matching vs non-matching valid candidates, GT overlaid
    ax = fig.add_subplot(b_slot)
    groups, labels, cols = [], [], []
    for var in variants:
        c = df[(df.input_variant == var) & (df.gen_file_k == k_ref) & np.isfinite(df.Rwp)]
        groups += [c.loc[c.matched_to_gt, 'Rwp'].to_numpy(), c.loc[~c.matched_to_gt, 'Rwp'].to_numpy()]
        pre = '' if single else f'{var}: '
        labels += [f'{pre}matching\n(n={c.matched_to_gt.sum()})', f'{pre}non-matching\n(n={(~c.matched_to_gt).sum()})']
        cols += [C['s1'], C['s2']]
    g = df[(df.input_variant == 'gt') & np.isfinite(df.Rwp)].Rwp.to_numpy()
    groups.append(g)
    labels.append(f'ground truth\n(n={len(g)})')
    cols.append(C['s3'])
    pos = np.arange(len(groups))
    keep = [i for i, v in enumerate(groups) if len(v) > 1]
    if keep:
        vp = ax.violinplot([groups[i] for i in keep], positions=pos[keep], widths=0.7,
                           showmedians=True, showextrema=False)
        for body, i in zip(vp['bodies'], keep):
            body.set_facecolor(cols[i])
            body.set_edgecolor(cols[i])
            body.set_alpha(0.35)
        vp['cmedians'].set_color(C['ink'])
        vp['cmedians'].set_linewidth(1)
    for i, v in enumerate(groups):
        jitter = np.random.default_rng(0).uniform(-0.12, 0.12, len(v))
        ax.scatter(pos[i] + jitter, v, s=4, color=cols[i], alpha=0.6, lw=0)
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel('refined $R_{wp}$ (%)', fontsize=8, color=C['ink2'])
    ax.set_title(f'(b) refined $R_{{wp}}$ (successful fits, k = {k_ref})' if single else
                 f'(b) refined $R_{{wp}}$, successful refinements, k = {k_ref} candidates',
                 fontsize=9, loc='left', color=C['ink'])
    style_ax(ax)

    # (c) representative sample: median GT Rwp (lower median), first variant
    var = variants[0]
    rep = summaries[var]['representative_sample']
    for j, (key, title) in enumerate([('top', 'top-ranked candidate'), ('nonmatch', 'non-matching candidate')]):
        ax = fig.add_subplot(gs[c_row, j])
        info = rep.get(key)
        if info is None:
            ax.text(0.5, 0.5, 'none available', ha='center', transform=ax.transAxes, fontsize=8)
            ax.set_axis_off()
            continue
        m = 'matches GT' if info['matched_to_gt'] else 'does not match GT'
        wd = out_dir / 'work' / info['job_id']
        plot_fit(ax, wd / 'fit_extra.npz' if (wd / 'fit_extra.npz').exists() else wd / 'fit_main.npz',
                 f"(c) {rep['sample_id']}: {title}\ncand {info['cand_idx']}, {info['spacegroup']}, {m}",
                 info['Rwp'])
        ax.set_xlabel('2θ (°)', fontsize=8, color=C['ink2'])
        if j == 0:
            ax.legend(fontsize=6, frameon=False, loc='center right')
    for ext in ('pdf', 'png'):
        fig.savefig(out_dir / f'ranking_figure.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)


def representative(df, var, k_ref):
    """Fixed rule: sample with the median GT Rwp (lower median) among samples whose
    GT refinement succeeded; show its Rwp top-1 candidate and the lowest-Rwp
    non-matching candidate other than the top-1."""
    gt = df[(df.input_variant == 'gt') & np.isfinite(df.Rwp)].sort_values(['Rwp', 'sample_idx'])
    if len(gt) == 0:
        return {}
    sid = int(gt.iloc[(len(gt) - 1) // 2].sample_idx)
    c = df[(df.input_variant == var) & (df.gen_file_k == k_ref) & (df.sample_idx == sid)]
    top = pick(c, 'Rwp')
    nm = c[(~c.matched_to_gt) & (c.cand_idx != top.cand_idx) & np.isfinite(c.Rwp)].sort_values(['Rwp', 'cand_idx'])
    fmt = lambda r: {k: (r[k].item() if hasattr(r[k], 'item') else r[k])  # noqa: E731
                     for k in ('job_id', 'cand_idx', 'spacegroup', 'Rwp', 'matched_to_gt')}
    return {'rule': 'median ground-truth Rwp (lower median) over samples with a successful GT refinement; '
                    'top = Rwp top-1 of the k=%d file; nonmatch = lowest-Rwp non-matching candidate other than top' % k_ref,
            'sample_idx': sid, 'sample_id': top.sample_id, 'gt_Rwp': float(gt.iloc[(len(gt) - 1) // 2].Rwp),
            'top': None if not np.isfinite(top.Rwp) else fmt(top),
            'nonmatch': fmt(nm.iloc[0]) if len(nm) else None}


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(ROOT / 'scripts/rietveld_ranking/config.yaml'))
    ap.add_argument('--variant', nargs='+', default=['peaks'])
    ap.add_argument('--n_samples', type=int, default=None)
    ap.add_argument('--samples', type=int, nargs='+', default=None)
    ap.add_argument('--n_boot', type=int, default=1000)
    args = ap.parse_args()
    cfg = load_config(args.config)
    k_ref = max(int(k) for k in cfg['gen_files'])
    summaries, dfs = {}, []
    for var in args.variant:
        prep_dir, out_dir = dirs(cfg, var)
        man = pd.read_csv(prep_dir / 'manifest.csv')
        if args.n_samples is not None:
            man = man[man.sample_idx < args.n_samples]
        if args.samples is not None:
            man = man[man.sample_idx.isin(args.samples)]
        df = load_results(out_dir, man)
        cols = ['sample_id', 'input_variant', 'gen_file_k', 'cand_idx', 'valid', 'invalid_reason', 'spacegroup',
                'n_lattice_params', 'n_refined_params', 'converged', 'fail_reason', 'Rwp', 'GoF', 'R_Bragg',
                'AIC', 'matched_to_gt', 'wall_time', 'sample_idx', 'sg_number', 'failed_stage', 'R_F2',
                'po_fail_reason', 'Rwp_po', 'R_Bragg_po', 'AIC_po', 'stuck_stages', 'final_stage', 'extra_fail_reason',
                'job_id']
        df[cols].replace([np.inf], np.nan).to_csv(out_dir / 'per_candidate.csv', index=False)

        rng = np.random.default_rng(cfg['seed'])
        s = {'n_samples': int(df.sample_idx.nunique()), 'rates': {}, 'aic_vs_rwp_top1_differs': {},
             'po_sensitivity': {}, 'failures': {}}
        for k in sorted(int(k) for k in cfg['gen_files']):
            c = df[(df.input_variant == var) & (df.gen_file_k == k)]
            tab = per_sample_table(c, ['Rwp', 'R_Bragg', 'AIC', 'Rwp_po', 'R_Bragg_po', 'AIC_po'])
            main_cols = ['coverage', 'random', 'ranked_Rwp', 'ranked_R_Bragg', 'ranked_AIC']
            s['rates'][str(k)] = bootstrap(tab, main_cols, args.n_boot, rng)
            s['po_sensitivity'][str(k)] = bootstrap(tab, ['coverage', 'ranked_Rwp_po', 'ranked_R_Bragg_po',
                                                          'ranked_AIC_po'], args.n_boot, rng)
            diff = (tab.pick_AIC != tab.pick_Rwp)
            s['aic_vs_rwp_top1_differs'][str(k)] = {
                'frac_samples': float(diff.mean()), 'n': int(diff.sum()),
                'of_which_changes_success': int((tab.ranked_AIC != tab.ranked_Rwp).sum())}
            s['po_sensitivity'][str(k)]['top1_changes_vs_main_Rwp'] = float((tab.pick_Rwp_po != tab.pick_Rwp).mean())
            s['failures'][str(k)] = {
                'n_candidates': int(len(c)), 'n_invalid': int((~c.valid).sum()),
                'by_stage': c.loc[c.failed_stage != '', 'failed_stage'].value_counts().to_dict(),
                'by_reason': c.loc[c.valid & (c.fail_reason != ''), 'fail_reason']
                             .str.split(':').str[:2].str.join(':').value_counts().to_dict(),
                'po_failed_after_main_ok': int(((c.fail_reason == '') & (c.po_fail_reason != '')).sum())}
            if k == 1:  # ranking a single candidate cannot change the outcome
                assert (tab.ranked_Rwp == tab.coverage).all()
        g = df[df.input_variant == 'gt']
        s['failures']['gt'] = {'n': int(len(g)),
                               'by_stage': g.loc[g.failed_stage != '', 'failed_stage'].value_counts().to_dict()}
        s['top_n'] = top_n(df[(df.input_variant == var) & (df.gen_file_k == k_ref)],
                           ['Rwp', 'R_Bragg', 'AIC'], [1, 2, 3, 4, 5], rng, args.n_boot)
        s['gt_sanity'] = gt_sanity(df, k_ref, rng, args.n_boot)
        s['representative_sample'] = representative(df, var, k_ref)
        s['manuscript_coverage_check'] = json.load(open(prep_dir / 'coverage_check.json'))
        s['protocol'] = {'config': str(args.config), 'lattice_prealign': cfg['lattice_prealign']}
        s['bragg_r_definition'] = ('GSAS-II ph:H:Rf = 100*sum|Fo-Fc|/sum|Fo|, Fo = sqrt of Rietveld-partitioned Fo^2 '
                                   '(R_F on |F|); R_F2 column = GSAS-II Rf^2 on F^2')
        summaries[var] = s
        json.dump(s, open(out_dir / 'summary.json', 'w'), indent=2)
        dfs.append(df)
        print(json.dumps({k: s[k] for k in ('rates', 'aic_vs_rwp_top1_differs', 'gt_sanity', 'failures')}, indent=1))
    make_figure(pd.concat(dfs).drop_duplicates('job_id'), summaries, args.variant,
                dirs(cfg, args.variant[0])[1], k_ref)


if __name__ == '__main__':
    main()
