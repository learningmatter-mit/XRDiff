"""Recover per-atom type logits for an existing eval_diff_*.pt without re-running diffusion.

In CSPDiffusion (diffusion_w_type.py) atom types never feed back into the trajectory:
the decoder is conditioned on the element set, and types are read off once, as the masked
argmax of the last predictor call at t=1. That call sees geometry that differs from the saved
final geometry by one predictor step (sigma_1^2 * score on coords, one lattice update), so a
single decoder pass at t=1 on the saved final structures gives nearly the same logits.
The script reports how often its masked argmax reproduces the saved atom types.

Output: <run>/type_logits_<name>.pt with
    log_probs  [n_evals, N_total, 100] float16, log-softmax over the element set (-inf elsewhere)
    agree      [n_evals, N_total] bool, argmax == saved atom type
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_scatter import scatter

SCRIPTS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS_ROOT.parent))
sys.path.insert(0, str(SCRIPTS_ROOT))
from eval_utils import load_model, load_data

MAX_ATOMIC_NUM = 100

if not torch.cuda.is_available():
    # load_model / load_data call torch.load without map_location; checkpoints hold CUDA tensors.
    _torch_load = torch.load
    torch.load = lambda *a, **k: _torch_load(*a, **{**k, 'map_location': 'cpu'})


@torch.no_grad()
def recover_one(model, device, gt, num_atoms, frac_all, lat_all, saved_types, batch_size, limit):
    B = min(num_atoms.numel(), limit) if limit else num_atoms.numel()
    ptr = torch.cat([torch.zeros(1, dtype=torch.long), num_atoms.cumsum(0)])
    time_emb_1 = model.time_embedding(torch.ones(1, dtype=torch.long, device=device))

    out_lp, out_agree = [], []
    for s in range(0, B, batch_size):
        e = min(s + batch_size, B)
        a0, a1 = ptr[s].item(), ptr[e].item()
        na = num_atoms[s:e].to(device)
        node2graph = torch.repeat_interleave(torch.arange(e - s, device=device), na)
        gt_types = gt.atom_types[a0:a1].to(device)
        elem_set = scatter(F.one_hot(gt_types - 1, MAX_ATOMIC_NUM).float(), node2graph, dim=0, reduce='max')
        elem_set_per_atom = elem_set[node2graph]
        xrd = gt.xrd[s:e].float().to(device) if getattr(gt, 'xrd', None) is not None else None

        _, _, pred_t = model.decoder(
            time_emb_1.expand(e - s, -1), elem_set_per_atom,
            frac_all[a0:a1].to(device), lat_all[s:e].to(device), na, node2graph, xrd=xrd)
        lp = pred_t.masked_fill(elem_set_per_atom == 0, float('-inf')).log_softmax(-1)
        out_lp.append(lp.half().cpu())
        out_agree.append((lp.argmax(-1).cpu() + 1) == saved_types[a0:a1])
        print(f'{e}/{B}', flush=True)

    log_probs, agree = torch.cat(out_lp), torch.cat(out_agree)
    struct_agree = scatter(agree.float(), torch.repeat_interleave(torch.arange(B), num_atoms[:B]),
                           dim=0, reduce='min').mean().item()
    print(f'atom-level agreement {agree.float().mean():.5f}, structure-level {struct_agree:.5f}')
    if limit:
        top2 = log_probs.float().topk(2, dim=-1).values
        margin = (top2[:, 0] - top2[:, 1]).nan_to_num(posinf=99.)  # inf when set has 1 element
        for name, m in (('agree', margin[agree]), ('disagree', margin[~agree])):
            if m.numel():
                print(f'{name}: n={m.numel()} margin quantiles(10,50,90)=',
                      [round(v, 3) for v in m.quantile(torch.tensor([.1, .5, .9])).tolist()])
    return log_probs, agree


def main(args):
    run = Path(args.root_path).resolve()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _, _ = load_model(run, load_data=False)
    model.to(device).eval()

    name = args.label or args.eval_split
    data = load_data(str(run / f'eval_diff_{name}.pt'))
    gt = data['input_data_batch']
    n_evals = data['num_atoms'].shape[0]
    lps, agrees = [], []
    for j in range(n_evals):
        assert torch.equal(data['num_atoms'][j], gt.num_atoms)
        print(f'attempt {j + 1}/{n_evals}')
        lp, ag = recover_one(model, device, gt, data['num_atoms'][j], data['frac_coords'][j],
                             data['lattices'][j], data['atom_types'][j], args.batch_size, args.limit)
        lps.append(lp)
        agrees.append(ag)
    if not args.limit:
        torch.save({'log_probs': torch.stack(lps), 'agree': torch.stack(agrees)},
                   run / f'type_logits_{name}.pt')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root_path', required=True)
    p.add_argument('--eval_split', default='test')
    p.add_argument('--label', default='', help='e.g. rruff_pv, rruff_pv_n_evals_10 (overrides eval_split)')
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--limit', type=int, default=0, help='only first N structures, no output (smoke test)')
    main(p.parse_args())
