import pdb
import time
import argparse
import random
import torch
import numpy as np
import sys
from pathlib import Path

from torch_geometric.data import Batch

script_dir = Path(__file__).resolve().parent
SCRIPTS_ROOT = script_dir.parent
PROJECT_ROOT = SCRIPTS_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_ROOT))

from eval_utils import load_model, lattices_to_params_shape, recommand_step_lr


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def diffusion(loader, model, num_evals, step_lr=1e-5):

    frac_coords = []
    num_atoms = []
    atom_types = []
    lattices = []
    input_data_list = []
    for idx, batch in enumerate(loader):

        if torch.cuda.is_available():
            batch.cuda()
        batch_frac_coords, batch_num_atoms, batch_atom_types = [], [], []
        batch_lattices = []
        for eval_idx in range(num_evals):

            print(f'batch {idx} / {len(loader)}, sample {eval_idx} / {num_evals}')
            outputs, traj = model.sample(batch, step_lr=step_lr)
            batch_frac_coords.append(outputs['frac_coords'].detach().cpu())
            batch_num_atoms.append(outputs['num_atoms'].detach().cpu())
            batch_atom_types.append(outputs['atom_types'].detach().cpu())
            batch_lattices.append(outputs['lattices'].detach().cpu())
        frac_coords.append(torch.stack(batch_frac_coords, dim=0))
        num_atoms.append(torch.stack(batch_num_atoms, dim=0))
        atom_types.append(torch.stack(batch_atom_types, dim=0))
        lattices.append(torch.stack(batch_lattices, dim=0))

        input_data_list = input_data_list + batch.to_data_list()
    frac_coords = torch.cat(frac_coords, dim=1)
    num_atoms = torch.cat(num_atoms, dim=1)
    atom_types = torch.cat(atom_types, dim=1)
    lattices = torch.cat(lattices, dim=1)
    lengths, angles = lattices_to_params_shape(lattices)
    input_data_batch = Batch.from_data_list(input_data_list)

    return (
        frac_coords, atom_types, lattices, lengths, angles, num_atoms, input_data_batch
    )


def main(args):
    set_seed(args.seed)
    model_path = Path(args.model_path)

    # RRUFF only has test data; for other datasets respect eval_split.
    if args.eval_dataset:
        if args.eval_dataset.startswith('rruff'):
            is_rruff = True
        else:
            is_rruff = False
    else:
        is_rruff = False
    if args.eval_split == 'test' or is_rruff:
        model, test_loader, cfg = load_model(
            model_path, load_data=True, eval_dataset=args.eval_dataset)
    else:
        model, all_loaders, cfg = load_model(
            model_path, load_data=True, testing=False,
            eval_dataset=args.eval_dataset)
        train_loader, val_loader = all_loaders
        test_loader = val_loader if args.eval_split == 'val' else train_loader

    if torch.cuda.is_available():
        model.to('cuda')

    print('Evaluate the diffusion model.')
    # get dataset from cfg
    dataset = cfg.data['name']
    step_lr = (args.step_lr if args.step_lr >= 0
               else recommand_step_lr[
                   'csp' if args.num_evals == 1 else 'csp_multi'].get(dataset, 1e-5))

    start_time = time.time()
    (frac_coords, atom_types, lattices, lengths, angles,
     num_atoms, input_data_batch) = diffusion(
        test_loader, model, args.num_evals, step_lr)

    if args.eval_dataset and args.eval_dataset.startswith('rruff'):
        if args.num_evals > 1:
            diff_out_name = f'eval_diff_{args.eval_dataset}_n_evals_{args.num_evals}.pt'
        else:
            diff_out_name = f'eval_diff_{args.eval_dataset}.pt'
    elif args.eval_dataset:
        diff_out_name = f'eval_diff_{args.eval_dataset}_{args.eval_split}.pt'
    elif args.label:
        diff_out_name = f'eval_diff_{args.label}_{args.eval_split}.pt'
    elif args.num_evals > 1:
        diff_out_name = f'eval_diff_n_evals_{args.num_evals}_{args.eval_split}.pt'
    else:
        diff_out_name = f'eval_diff_{args.eval_split}.pt'

    save_dict = {
        'eval_setting': args,
        'input_data_batch': input_data_batch,
        'frac_coords': frac_coords,
        'num_atoms': num_atoms,
        'atom_types': atom_types,
        'lattices': lattices,
        'lengths': lengths,
        'angles': angles,
        'time': time.time() - start_time,
    }
    if is_rruff:
        save_dict['material_ids'] = [
            d['mp_id'] for d in test_loader.dataset.cached_data]

    torch.save(save_dict, model_path / diff_out_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--eval_split', default='val')
    parser.add_argument('--seed', type=int, default=42, help='RNG seed for deterministic evaluation.')
    parser.add_argument('--eval_dataset', default=None,
                        help='Dataset to evaluate on. For RRUFF subsets use '
                             '"rruff_20", "rruff_25", etc. (maps to rruff_XX_data.csv). '
                             'If None, uses the model\'s original dataset.')
    parser.add_argument('--step_lr', default=-1, type=float)
    parser.add_argument('--num_evals', default=1, type=int)
    parser.add_argument('--label', default='')
    args = parser.parse_args()
    main(args)
