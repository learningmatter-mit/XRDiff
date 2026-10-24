"""Step 2: refine every job in manifest.csv with GSAS-II, in parallel.

Runs in the repo env; each refinement is a separate subprocess in the GSAS-II
env (g2_worker.py) inside its own directory under rietveld_ranking/work/.
Invalid candidates are not refined (they get the worst score in analyze.py);
every other job gets a result.json, including failures and timeouts.

    python scripts/rietveld_ranking/refine.py --variant peaks --n_samples 5   # dry run
    python scripts/rietveld_ranking/refine.py --variant peaks                 # full run
"""
import argparse
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from common import dirs, load_config

ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / 'scripts/rietveld_ranking/g2_worker.py'
G2_PYTHON = os.environ.get('G2_PYTHON')   # python of the GSAS-II environment (see README)
if G2_PYTHON is None:
    sys.exit('set $G2_PYTHON to the python interpreter of your GSAS-II environment')


def run_job(job):
    wd = Path(job['workdir'])
    wd.mkdir(parents=True, exist_ok=True)
    res_path = wd / 'result.json'
    if res_path.exists() and not job['force']:
        return json.load(open(res_path))
    with open(wd / 'job.json', 'w') as f:
        json.dump(job, f, indent=1)
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    t0 = time.time()
    # hard backstop in case SIGALRM cannot interrupt compiled code
    hard_limit = 2 * job['timeout_s'] + 60
    try:
        p = subprocess.run([G2_PYTHON, str(WORKER), str(wd / 'job.json')], cwd=wd, env=env,
                           capture_output=True, text=True, timeout=hard_limit)
        (wd / 'worker.log').write_text(p.stdout[-20000:] + '\n--- stderr ---\n' + p.stderr[-20000:])
        if res_path.exists():
            out = json.load(open(res_path))
        else:
            out = {'job_id': job['job_id'], 'stages': {}, 'po_fail_reason': 'not_run',
                   'fail_reason': f'worker_crashed:rc={p.returncode}'}
    except subprocess.TimeoutExpired:
        out = {'job_id': job['job_id'], 'stages': {}, 'po_fail_reason': 'not_run',
               'fail_reason': 'hard_timeout'}
    out['total_wall_time'] = time.time() - t0
    json.dump(out, open(res_path, 'w'), indent=1)
    for f in wd.glob('*.bak*'):
        f.unlink()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(ROOT / 'scripts/rietveld_ranking/config.yaml'))
    ap.add_argument('--variant', default='peaks')
    ap.add_argument('--n_samples', type=int, default=None,
                    help='dry run: only the first N samples (by sample index)')
    ap.add_argument('--samples', type=int, nargs='+', default=None,
                    help='dry run: only these sample indices')
    ap.add_argument('--no_po', action='store_true', help='skip the preferred-orientation pass')
    ap.add_argument('--force', action='store_true', help='re-run jobs that already have a result')
    args = ap.parse_args()
    cfg = load_config(args.config)
    prep_dir, out_dir = dirs(cfg, args.variant)
    man = pd.read_csv(prep_dir / 'manifest.csv')
    if args.n_samples is not None:
        man = man[man.sample_idx < args.n_samples]
    if args.samples is not None:
        man = man[man.sample_idx.isin(args.samples)]
    # only the generation files listed in the config (a variant may restrict k); GT jobs always
    man = man[man.gen_file_k.isin([int(k) for k in cfg['gen_files']]) | (man.input_variant == 'gt')]
    man = man[man.valid]

    keys = ['two_theta_min', 'two_theta_max', 'uiso', 'background_terms', 'max_cycles_per_stage',
            'stages', 'po_stage', 'timeout_s', 'lattice_prealign', 'extra_stages']
    jobs = [dict(job_id=r.job_id, cif=r.cif_path, xye=r.xye_path, instprm=str(ROOT / cfg['instprm']),
                 workdir=str(out_dir / 'work' / r.job_id), run_po=cfg['run_po'] and not args.no_po, force=args.force,
                 **{k: cfg[k] for k in keys})
            for r in man.itertuples()]
    print(f'{len(jobs)} refinements -> {out_dir}, {cfg["n_workers"]} workers, '
          f'timeout {cfg["timeout_s"]} CPU s/pass, lattice_prealign={cfg["lattice_prealign"]}')
    t0 = time.time()
    n_fail = 0
    with Pool(cfg['n_workers']) as pool:
        for i, out in enumerate(pool.imap_unordered(run_job, jobs), 1):
            n_fail += bool(out.get('fail_reason'))
            if i % 50 == 0 or i == len(jobs):
                print(f'  {i}/{len(jobs)} done, {n_fail} failed, {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
