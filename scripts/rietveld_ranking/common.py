"""Config loading shared by prepare.py / refine.py / analyze.py / rruff_eval/report.py.

A variant config may name a `base:` config (relative to its own directory) and
override only the keys it changes, so the main protocol and a variant cannot
drift apart silently.
"""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path):
    path = Path(path)
    cfg = yaml.safe_load(open(path))
    base = cfg.pop('base', None)
    if base:
        merged = load_config(path.parent / base)
        merged.update(cfg)
        cfg = merged
    cfg.setdefault('output_subdir', 'rietveld_ranking')
    cfg.setdefault('lattice_prealign', None)
    cfg.setdefault('extra_stages', [])
    cfg.setdefault('run_po', True)
    return cfg


def dirs(cfg, variant):
    """(prep_dir, out_dir): prepare.py outputs (manifest, CIFs, observed) are shared;
    refinements and results of each protocol variant go to their own out_dir."""
    run_dir = ROOT / cfg['run_dirs'][variant]
    return run_dir / 'rietveld_ranking', run_dir / cfg['output_subdir']
