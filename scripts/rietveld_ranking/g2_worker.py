"""Refine one structure against one observed pattern with GSAS-II (runs in the GSAS-II env).

Invoked by refine.py as a subprocess, one per candidate, inside that candidate's
own working directory:

    python g2_worker.py <job.json>

job.json carries the CIF, the .xye, the .instprm and the protocol settings. The
worker runs the cumulative main sequence (a scale+background, b +zero,
c +lattice, d +profile) and then, as a separate pass with its own time budget,
the preferred-orientation sensitivity stage (e +March-Dollase). It writes
result.json (per-stage statistics) and fit_main.npz / fit_po.npz (x, yobs,
ycalc, background) to the working directory. Nothing here sees the ground truth.
"""
import json
import math
import os
import signal
import sys
import time
import traceback

import numpy as np
from GSASII import GSASIIscriptable as G2sc


class Timeout(BaseException):
    """BaseException so GSAS-II's internal `except Exception` blocks cannot swallow it."""
    pass


def _alarm(signum, frame):
    raise Timeout()


def stage_stats(gpx, hist, t0):
    rv = gpx['Covariance']['data']['Rvals']
    res = hist.residuals
    rf = [v for k, v in res.items() if k.endswith(':Rf')]
    rf2 = [v for k, v in res.items() if k.endswith(':Rf^2')]
    return {
        'Rwp': float(rv['Rwp']), 'GoF': float(rv['GOF']), 'chisq': float(rv['chisq']),
        'Nobs': int(rv['Nobs']), 'Nvars': int(rv['Nvars']),
        'converged': bool(rv.get('converged')), 'aborted': bool(rv.get('Aborted', False)),
        'R_F': float(rf[0]) if rf else float('nan'),
        'R_F2': float(rf2[0]) if rf2 else float('nan'),
        'msg': str(rv.get('msg', '')), 'wall_time': time.time() - t0,
    }


def refine_stage(gpx, hist, t0):
    """One GSAS-II refine call. Clears the previous Rvals first: when a refinement
    fails GSAS-II leaves the old ones in place, which would otherwise be read as
    this stage's result."""
    gpx.data['Covariance']['data'] = {}
    gpx.refine()
    if 'Rvals' not in gpx['Covariance']['data']:
        return None
    return stage_stats(gpx, hist, t0)


def init_scale_background(gpx, hist, n_terms):
    """Reference-free starting point for stage a: evaluate the pattern once
    (0 cycles), then set the histogram scale s and constant background c from a
    linear least-squares fit yobs ~ s*(ycalc - bkg) + c. Uses only the observed
    pattern and the candidate's own calculated profile."""
    gpx.set_Controls('cycles', 0)
    gpx.data['Covariance']['data'] = {}
    gpx.refine()
    yo, yc, bk = (np.asarray(hist.getdata(k), float) for k in ('yobs', 'ycalc', 'background'))
    prof = yc - bk
    (s, c), *_ = np.linalg.lstsq(np.column_stack([prof, np.ones_like(prof)]), yo, rcond=None)
    if not (np.isfinite(s) and s > 0):      # fall back to integrated-intensity ratio
        c = float(np.median(yo))
        s = float(np.clip(yo - c, 0, None).sum() / max(prof.sum(), 1e-30))
    hist.SampleParameters['Scale'][0] *= float(s)
    hist.data['Background'][0][:] = ['chebyschev-1', False, n_terms, float(c)] + [0.0] * (n_terms - 1)
    return float(s), float(c)


def prealign_lattice(gpx, hist, phase, cfg):
    """Variant only (lattice_prealign set): reference-free isotropic lattice
    pre-alignment before stage a. The candidate's own reflection list (d,
    Fc^2 * Icorr) is re-positioned for cell lengths scaled by f, rendered with a
    fixed Gaussian width, and f is chosen to maximise the Pearson correlation
    with the observed pattern (coarse grid, then a fine grid around the best).
    Angles are unchanged; the best f is applied to a, b, c."""
    from scipy.ndimage import gaussian_filter1d
    gpx.set_Controls('cycles', 0)
    gpx.data['Covariance']['data'] = {}
    gpx.refine()
    rl = np.asarray(hist.data['Reflection Lists'][phase.name]['RefList'], float)
    d, inten = rl[:, 4], rl[:, 9] * rl[:, 11]
    x = np.asarray(hist.getdata('x'), float)
    yo = np.asarray(hist.getdata('yobs'), float)
    lam = hist.InstrumentParameters['Lam'][1]
    step = float(np.median(np.diff(x)))
    sig_bins = cfg['render_fwhm_deg'] / 2.3548 / step

    def corr(f):
        arg = lam / (2 * f * d)
        ok = arg < 1
        tth = 2 * np.degrees(np.arcsin(arg[ok]))
        idx = np.rint((tth - x[0]) / step).astype(int)
        m = (idx >= 0) & (idx < len(x))
        y = np.zeros_like(x)
        np.add.at(y, idx[m], inten[ok][m])
        y = gaussian_filter1d(y, sig_bins)
        return float(np.corrcoef(yo, y)[0, 1]) if y.std() > 0 else -1.0

    lo, hi = cfg['range']
    coarse = np.arange(lo, hi + 1e-9, cfg['coarse_step'])
    c_coarse = [corr(f) for f in coarse]
    f0 = coarse[int(np.argmax(c_coarse))]
    fine = np.arange(max(lo, f0 - cfg['coarse_step']), min(hi, f0 + cfg['coarse_step']) + 1e-9, cfg['fine_step'])
    c_fine = [corr(f) for f in fine]
    f_best = float(fine[int(np.argmax(c_fine))])
    cell = phase.data['General']['Cell']
    for i in (1, 2, 3):
        cell[i] *= f_best
    cell[7] *= f_best ** 3
    return {'f': f_best, 'corr': float(max(c_fine)), 'corr_at_1': corr(1.0)}


def check(stats, stage):
    """Divergence / non-convergence of one stage -> failure reason or ''."""
    if stats is None:
        return f'{stage}:diverged_refine_failed'
    if not all(math.isfinite(stats[k]) for k in ('Rwp', 'GoF', 'chisq')):
        return f'{stage}:diverged_nonfinite'
    if stats['aborted']:
        # GSAS-II "stuck" (Ouch#4): no step lowers chi^2 even at max Marquardt
        # lambda. Per protocol decision this is not a failure: the best state
        # found is kept and the sequence continues (recorded as stuck).
        return ''
    if not stats['converged']:
        return f'{stage}:not_converged'
    return ''


def save_fit(hist, path):
    np.savez_compressed(path, **{k: np.asarray(hist.getdata(k), np.float32)
                                 for k in ('x', 'yobs', 'ycalc', 'background')})


def run_main(job, wd, out):
    G2sc.SetPrintLevel('none')
    gpx = G2sc.G2Project(newgpx=os.path.join(wd, 'refine.gpx'))
    hist = gpx.add_powder_histogram(job['xye'], job['instprm'], fmthint='xye')
    phase = gpx.add_phase(job['cif'], phasename='phase', histograms=[hist], fmthint='CIF')
    hist.set_refinements({'Limits': [job['two_theta_min'], job['two_theta_max']]})
    for atom in phase.atoms():          # coordinates, occupancies, Uiso all fixed
        atom.refinement_flags = ''
        assert atom.adp_flag == 'I', f'anisotropic atom {atom.label}'
        atom.uiso = job['uiso']
    phase.set_refinements({'Cell': False})
    phase.set_HAP_refinements({'Scale': False, 'Pref.Ori.': False})
    out['spacegroup_g2'] = phase.data['General']['SGData']['SpGrp']
    if job.get('lattice_prealign'):
        out['failed_stage'] = 'prealign'
        out['prealign'] = prealign_lattice(gpx, hist, phase, job['lattice_prealign'])
    out['init_scale'], out['init_background'] = init_scale_background(gpx, hist, job['background_terms'])
    gpx.set_Controls('cycles', job['max_cycles_per_stage'])

    stage_refs = {
        'scale_background': lambda: hist.set_refinements({
            'Sample Parameters': ['Scale'],
            'Background': {'type': 'chebyschev-1', 'no. coeffs': job['background_terms'], 'refine': True}}),
        'zero': lambda: hist.set_refinements({'Instrument Parameters': ['Zero']}),
        'lattice': lambda: phase.set_refinements({'Cell': True}),
        'profile': lambda: hist.set_refinements({'Instrument Parameters': ['U', 'V', 'W', 'X', 'Y']}),
    }
    t_start = time.time()
    for stage in job['stages']:
        out['failed_stage'] = stage
        t0 = time.time()
        stage_refs[stage]()
        s = refine_stage(gpx, hist, t0)
        out['stages'][stage] = s
        reason = check(s, stage)
        if s is not None and s['aborted']:
            out.setdefault('stuck_stages', []).append(stage)
        if reason:
            out['fail_reason'] = reason
            return None
    out['failed_stage'] = ''
    out['main_wall_time'] = time.time() - t_start
    out['main_cpu_time'] = time.process_time()
    save_fit(hist, os.path.join(wd, 'fit_main.npz'))
    gpx.save()
    return gpx


def run_po(job, wd, out):
    G2sc.SetPrintLevel('none')
    gpx = G2sc.G2Project(gpxfile=os.path.join(wd, 'refine.gpx'))
    hist, phase = gpx.histograms()[0], gpx.phases()[0]
    ax = job['po_stage']['axis']
    for hname in phase.data['Histograms']:
        po = phase.data['Histograms'][hname]['Pref.Ori.']
        po[0], po[1], po[3] = 'MD', job['po_stage']['ratio_start'], list(ax)
    phase.set_HAP_refinements({'Pref.Ori.': True})
    t0 = time.time()
    s = refine_stage(gpx, hist, t0)
    out['stages']['pref_orient'] = s
    out['po_fail_reason'] = check(s, 'pref_orient')
    if s is not None and s['aborted']:
        out.setdefault('stuck_stages', []).append('pref_orient')
    if not out['po_fail_reason']:
        save_fit(hist, os.path.join(wd, 'fit_po.npz'))


def run_extra(job, wd, out, gpx):
    """Variant only (extra_stages set): further cumulative stages after stage d.
    A failure here does not fail the candidate: its score falls back to the last
    successful stage (recorded in final_stage)."""
    hist, phase = gpx.histograms()[0], gpx.phases()[0]
    refs = {
        'coords': lambda: [setattr(a, 'refinement_flags', 'X') for a in phase.atoms()],
    }
    for stage in job['extra_stages']:
        out['extra_stage'] = stage
        t0 = time.time()
        refs[stage]()
        s = refine_stage(gpx, hist, t0)
        out['stages'][stage] = s
        if s is not None and s['aborted']:
            out.setdefault('stuck_stages', []).append(stage)
        reason = check(s, stage)
        if reason:
            out['extra_fail_reason'] = reason
            return
        out['final_stage'] = stage
        save_fit(hist, os.path.join(wd, 'fit_extra.npz'))
    out['extra_fail_reason'] = ''


def main():
    job = json.load(open(sys.argv[1]))
    wd = os.path.dirname(os.path.abspath(sys.argv[1]))
    out = {'job_id': job['job_id'], 'stages': {}, 'fail_reason': '', 'po_fail_reason': 'not_run',
           'failed_stage': ''}
    # CPU-time limit (SIGPROF), not wall-clock, so the outcome does not depend on node load
    signal.signal(signal.SIGPROF, _alarm)
    arm = lambda t: signal.setitimer(signal.ITIMER_PROF, t)  # noqa: E731
    try:
        arm(job['timeout_s'])
        gpx = run_main(job, wd, out)
        arm(0)
    except Timeout:
        out['fail_reason'] = f"{out['failed_stage']}:timeout"
        gpx = None
    except Exception as e:
        arm(0)
        out['fail_reason'] = f"{out['failed_stage'] or 'setup'}:exception:{type(e).__name__}:{str(e)[:200]}"
        out['traceback'] = traceback.format_exc()[-2000:]
        gpx = None
    if gpx is not None:
        out['final_stage'] = job['stages'][-1]
    if gpx is not None and job.get('extra_stages'):
        try:
            arm(job['timeout_s'])
            run_extra(job, wd, out, gpx)
            arm(0)
        except Timeout:
            out['extra_fail_reason'] = f"{out.get('extra_stage', '')}:timeout"
        except Exception as e:
            arm(0)
            out['extra_fail_reason'] = f"{out.get('extra_stage', '')}:exception:{type(e).__name__}:{str(e)[:200]}"
    if gpx is not None and job.get('run_po', True):
        try:
            arm(job['timeout_s'])
            run_po(job, wd, out)
            arm(0)
        except Timeout:
            out['po_fail_reason'] = 'pref_orient:timeout'
        except Exception as e:
            arm(0)
            out['po_fail_reason'] = f'pref_orient:exception:{type(e).__name__}:{str(e)[:200]}'
    json.dump(out, open(os.path.join(wd, 'result.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
