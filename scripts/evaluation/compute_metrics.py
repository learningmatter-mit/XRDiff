from collections import Counter
import argparse
import os
import json
import random

import numpy as np
from pathlib import Path
from tqdm import tqdm
from p_tqdm import p_map
from scipy.stats import wasserstein_distance
import pandas as pd

from pymatgen.core.structure import Structure
from pymatgen.core.composition import Composition
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.structure_matcher import StructureMatcher
from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
from matminer.featurizers.composition.composite import ElementProperty

import warnings

# Suppress noisy but non-critical warnings
warnings.filterwarnings(
    "ignore",
    message="The version_base parameter is not specified.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    module="pymatgen.analysis.local_env",
    category=UserWarning,
)

import sys
sys.path.append('.')
# eval_utils lives one level up, in scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_utils import (
    smact_validity, structure_validity, CompScaler, get_fp_pdist,
    load_config, load_data, get_crystals_list, prop_model_eval, compute_cov)

CrystalNNFP = CrystalNNFingerprint.from_preset("ops")
CompFP = ElementProperty.from_preset('magpie')

Percentiles = {
    'mp20': np.array([-3.17562208, -2.82196882, -2.52814761]),
    'carbon': np.array([-154.527093, -154.45865733, -154.44206825]),
    'perovskite': np.array([0.43924842, 0.61202443, 0.7364607]),
}

COV_Cutoffs = {
    'mp20': {'struc': 0.4, 'comp': 10.},
    'carbon': {'struc': 0.2, 'comp': 4.},
    'perovskite': {'struc': 0.2, 'comp': 4},
}


class Crystal(object):

    def __init__(self, crys_array_dict):
        self.frac_coords = crys_array_dict['frac_coords']
        self.atom_types = crys_array_dict['atom_types']
        self.lengths = crys_array_dict['lengths']
        self.angles = crys_array_dict['angles']
        self.dict = crys_array_dict
        if len(self.atom_types.shape) > 1:
            self.dict['atom_types'] = (np.argmax(self.atom_types, axis=-1) + 1)
            self.atom_types = (np.argmax(self.atom_types, axis=-1) + 1)

        self.get_structure()
        self.get_composition()
        self.get_validity()
        self.get_fingerprints()


    def get_structure(self):
        if min(self.lengths.tolist()) < 0:
            self.constructed = False
            self.invalid_reason = 'non_positive_lattice'
        if np.isnan(self.lengths).any() or np.isnan(self.angles).any() or  np.isnan(self.frac_coords).any():
            self.constructed = False
            self.invalid_reason = 'nan_value'            
        else:
            try:
                self.structure = Structure(
                    lattice=Lattice.from_parameters(
                        *(self.lengths.tolist() + self.angles.tolist())),
                    species=self.atom_types, coords=self.frac_coords, coords_are_cartesian=False)
                self.constructed = True
            except Exception:
                self.constructed = False
                self.invalid_reason = 'construction_raises_exception'
            if self.structure.volume < 0.1:
                self.constructed = False
                self.invalid_reason = 'unrealistically_small_lattice'

    def get_composition(self):
        elem_counter = Counter(self.atom_types)
        composition = [(elem, elem_counter[elem])
                       for elem in sorted(elem_counter.keys())]
        elems, counts = list(zip(*composition))
        counts = np.array(counts)
        counts = counts / np.gcd.reduce(counts)
        self.elems = elems
        self.comps = tuple(counts.astype('int').tolist())

    def get_validity(self):
        self.comp_valid = smact_validity(self.elems, self.comps)
        if self.constructed:
            self.struct_valid = structure_validity(self.structure)
        else:
            self.struct_valid = False
        self.valid = self.comp_valid and self.struct_valid

    def get_fingerprints(self):
        elem_counter = Counter(self.atom_types)
        comp = Composition(elem_counter)
        self.comp_fp = CompFP.featurize(comp)
        try:
            site_fps = [CrystalNNFP.featurize(
                self.structure, i) for i in range(len(self.structure))]
        except Exception:
            # counts crystal as invalid if fingerprint cannot be constructed.
            self.valid = False
            self.comp_fp = None
            self.struct_fp = None
            return
        self.struct_fp = np.array(site_fps).mean(axis=0)


class RecEval(object):

    def __init__(self, pred_crys, gt_crys, structurematcher_kwargs={}, return_match_indices=False):
        assert len(pred_crys) == len(gt_crys)
        stol, angle_tol, ltol = structurematcher_kwargs.get('stol', 0.3), structurematcher_kwargs.get('angle_tol', 5), structurematcher_kwargs.get('ltol', 0.2)
        self.matcher = StructureMatcher(
            stol=stol, angle_tol=angle_tol, ltol=ltol)
        self.preds = pred_crys
        self.gts = gt_crys
        self.return_match_indices = return_match_indices

    def get_match_rate_and_rms(self):
        def process_one(pred, gt, is_valid):
            if not is_valid:
                return None
            try:
                rms_dist = self.matcher.get_rms_dist(
                    pred.structure, gt.structure)
                rms_dist = None if rms_dist is None else rms_dist[0]
                return rms_dist
            except Exception:
                return None
        validity = [c1.valid and c2.valid for c1,c2 in zip(self.preds, self.gts)]

        rms_dists = []
        for i in tqdm(range(len(self.preds))):
            rms_dists.append(process_one(
                self.preds[i], self.gts[i], validity[i]))
        rms_dists = np.array(rms_dists)
        match_rate = sum(rms_dists != None) / len(self.preds)
        mean_rms_dist = rms_dists[rms_dists != None].mean()
        result = {'match_rate': match_rate,
                'rms_dist': mean_rms_dist}     
        
        if self.return_match_indices:
            match_indices = np.where(rms_dists != None)[0].tolist()
            result['match_indices'] = match_indices
        
        return result     

    def get_metrics(self):
        metrics = {}
        metrics.update(self.get_match_rate_and_rms())
        return metrics

def _comp_key(crystal):
    return (tuple(crystal.elems), tuple(crystal.comps))


def _build_comp_to_gt_indices(gt_crys):
    comp_map = {}
    for i, gt in enumerate(gt_crys):
        comp_map.setdefault(_comp_key(gt), []).append(i)
    return comp_map


class RecEvalMetre(object):
    """Metre (Match-Everyone-To-Reference) evaluation for composition-conditional models.

    A prediction at index i is considered matched if it structurally matches
    ANY ground-truth structure that shares the same reduced composition as the
    prediction.  Use this for models like DiffCSP that receive only composition
    as input and therefore cannot be expected to target a specific polymorph.
    """

    def __init__(self, pred_crys, gt_crys, structurematcher_kwargs={}, return_match_indices=False):
        assert len(pred_crys) == len(gt_crys)
        stol = structurematcher_kwargs.get('stol', 0.3)
        angle_tol = structurematcher_kwargs.get('angle_tol', 5)
        ltol = structurematcher_kwargs.get('ltol', 0.2)
        self.matcher = StructureMatcher(stol=stol, angle_tol=angle_tol, ltol=ltol)
        self.preds = pred_crys
        self.gts = gt_crys
        self.return_match_indices = return_match_indices
        self._comp_to_gt = _build_comp_to_gt_indices(gt_crys)

    def _best_rms_against_comp(self, pred, gt_indices):
        if not pred.valid:
            return None
        best = None
        for idx in gt_indices:
            gt = self.gts[idx]
            if not gt.valid:
                continue
            try:
                rms = self.matcher.get_rms_dist(pred.structure, gt.structure)
                if rms is not None:
                    rms = rms[0]
                    if best is None or rms < best:
                        best = rms
            except Exception:
                pass
        return best

    def get_match_rate_and_rms(self):
        rms_dists = []
        for i in tqdm(range(len(self.preds))):
            pred = self.preds[i]
            gt_indices = self._comp_to_gt.get(_comp_key(pred), [])
            rms_dists.append(self._best_rms_against_comp(pred, gt_indices))

        rms_dists = np.array(rms_dists)
        match_rate = sum(rms_dists != None) / len(self.preds)
        matched = rms_dists[rms_dists != None]
        mean_rms_dist = matched.mean() if len(matched) > 0 else float('nan')
        result = {'match_rate': match_rate, 'rms_dist': mean_rms_dist}
        if self.return_match_indices:
            result['match_indices'] = np.where(rms_dists != None)[0].tolist()
        return result

    def get_metrics(self):
        return self.get_match_rate_and_rms()


class RecEvalBatchMetre(object):
    """Metre evaluation for best-of-N composition-conditional models (e.g. DiffCSP multi-eval).

    For each GT position i, checks whether any of the N candidate predictions
    matches ANY ground-truth structure with the same composition as gt[i].
    """

    def __init__(self, pred_crys, gt_crys, structurematcher_kwargs={}, return_match_indices=False):
        stol = structurematcher_kwargs.get('stol', 0.3)
        angle_tol = structurematcher_kwargs.get('angle_tol', 5)
        ltol = structurematcher_kwargs.get('ltol', 0.2)
        self.matcher = StructureMatcher(stol=stol, angle_tol=angle_tol, ltol=ltol)
        self.preds = pred_crys
        self.gts = gt_crys
        self.batch_size = len(self.preds)
        self.return_match_indices = return_match_indices
        self._comp_to_gt = _build_comp_to_gt_indices(gt_crys)

    def get_match_rate_and_rms(self):
        n = len(self.gts)
        rms_dists = []
        for i in tqdm(range(n)):
            comp_gt_indices = self._comp_to_gt.get(_comp_key(self.gts[i]), [])
            best = None
            for j in range(self.batch_size):
                pred = self.preds[j][i]
                if not pred.valid:
                    continue
                for gt_idx in comp_gt_indices:
                    ref_gt = self.gts[gt_idx]
                    if not ref_gt.valid:
                        continue
                    try:
                        rms = self.matcher.get_rms_dist(pred.structure, ref_gt.structure)
                        if rms is not None:
                            rms = rms[0]
                            if best is None or rms < best:
                                best = rms
                    except Exception:
                        pass
            rms_dists.append(best)

        rms_dists = np.array(rms_dists)
        match_rate = sum(rms_dists != None) / n
        matched = rms_dists[rms_dists != None]
        mean_rms_dist = matched.mean() if len(matched) > 0 else float('nan')
        result = {'match_rate': match_rate, 'rms_dist': mean_rms_dist}
        if self.return_match_indices:
            result['match_indices'] = np.where(rms_dists != None)[0].tolist()
        return result

    def get_metrics(self):
        return self.get_match_rate_and_rms()


class RecEvalBatch(object):

    def __init__(self, pred_crys, gt_crys, structurematcher_kwargs={}, return_match_indices=False):
        stol, angle_tol, ltol = structurematcher_kwargs.get('stol', 0.3), structurematcher_kwargs.get('angle_tol', 5), structurematcher_kwargs.get('ltol', 0.2)
        self.matcher = StructureMatcher(
            stol=stol, angle_tol=angle_tol, ltol=ltol)
        self.preds = pred_crys
        self.gts = gt_crys
        self.batch_size = len(self.preds)
        self.return_match_indices = return_match_indices

    def get_match_rate_and_rms(self):
        def process_one(pred, gt, is_valid):
            if not is_valid:
                return None
            try:
                rms_dist = self.matcher.get_rms_dist(
                    pred.structure, gt.structure)
                rms_dist = None if rms_dist is None else rms_dist[0]
                return rms_dist
            except Exception:
                return None

        rms_dists = []
        self.all_rms_dis = np.zeros((self.batch_size, len(self.gts)))
        for i in tqdm(range(len(self.preds[0]))):
            tmp_rms_dists = []
            for j in range(self.batch_size):
                rmsd = process_one(self.preds[j][i], self.gts[i], self.preds[j][i].valid)
                self.all_rms_dis[j][i] = rmsd
                if rmsd is not None:
                    tmp_rms_dists.append(rmsd)
            if len(tmp_rms_dists) == 0:
                rms_dists.append(None)
            else:
                rms_dists.append(np.min(tmp_rms_dists))
            
        rms_dists = np.array(rms_dists)
        match_rate = sum(rms_dists != None) / len(self.preds[0])
        mean_rms_dist = rms_dists[rms_dists != None].mean()
        
        result = {'match_rate': match_rate,
                'rms_dist': mean_rms_dist}    
        
        if self.return_match_indices:
            match_indices = np.where(rms_dists != None)[0].tolist()
            result['match_indices'] = match_indices
        
        return result    

    def get_metrics(self):
        metrics = {}
        metrics.update(self.get_match_rate_and_rms())
        return metrics



class GenEval(object):

    def __init__(self, pred_crys, gt_crys, n_samples=1000, eval_model_name=None):
        self.crys = pred_crys
        self.gt_crys = gt_crys
        self.n_samples = n_samples
        self.eval_model_name = eval_model_name

        valid_crys = [c for c in pred_crys if c.valid]
        if len(valid_crys) >= n_samples:
            sampled_indices = np.random.choice(
                len(valid_crys), n_samples, replace=False)
            self.valid_samples = [valid_crys[i] for i in sampled_indices]
        else:
            raise Exception(
                f'not enough valid crystals in the predicted set: {len(valid_crys)}/{n_samples}')

    def get_validity(self):
        comp_valid = np.array([c.comp_valid for c in self.crys]).mean()
        struct_valid = np.array([c.struct_valid for c in self.crys]).mean()
        valid = np.array([c.valid for c in self.crys]).mean()
        return {'comp_valid': comp_valid,
                'struct_valid': struct_valid,
                'valid': valid}


    def get_density_wdist(self):
        pred_densities = [c.structure.density for c in self.valid_samples]
        gt_densities = [c.structure.density for c in self.gt_crys]
        wdist_density = wasserstein_distance(pred_densities, gt_densities)
        return {'wdist_density': wdist_density}


    def get_num_elem_wdist(self):
        pred_nelems = [len(set(c.structure.species))
                       for c in self.valid_samples]
        gt_nelems = [len(set(c.structure.species)) for c in self.gt_crys]
        wdist_num_elems = wasserstein_distance(pred_nelems, gt_nelems)
        return {'wdist_num_elems': wdist_num_elems}

    def get_prop_wdist(self):
        if self.eval_model_name is not None:
            pred_props = prop_model_eval(self.eval_model_name, [
                                         c.dict for c in self.valid_samples])
            gt_props = prop_model_eval(self.eval_model_name, [
                                       c.dict for c in self.gt_crys])
            wdist_prop = wasserstein_distance(pred_props, gt_props)
            return {'wdist_prop': wdist_prop}
        else:
            return {'wdist_prop': None}

    def get_coverage(self):
        cutoff_dict = COV_Cutoffs[self.eval_model_name]
        (cov_metrics_dict, combined_dist_dict) = compute_cov(
            self.crys, self.gt_crys,
            struc_cutoff=cutoff_dict['struc'],
            comp_cutoff=cutoff_dict['comp'])
        return cov_metrics_dict

    def get_metrics(self):
        metrics = {}
        metrics.update(self.get_validity())
        metrics.update(self.get_density_wdist())
        metrics.update(self.get_prop_wdist())
        metrics.update(self.get_num_elem_wdist())
        metrics.update(self.get_coverage())
        return metrics

class OptEval(object):

    def __init__(self, crys, num_opt=100, eval_model_name=None):
        """
        crys is a list of length (<step_opt> * <num_opt>),
        where <num_opt> is the number of different initialization for optimizing crystals,
        and <step_opt> is the number of saved crystals for each intialzation.
        default to minimize the property.
        """
        step_opt = int(len(crys) / num_opt)
        self.crys = crys
        self.step_opt = step_opt
        self.num_opt = num_opt
        self.eval_model_name = eval_model_name

    def get_success_rate(self):
        valid_indices = np.array([c.valid for c in self.crys])
        valid_indices = valid_indices.reshape(self.step_opt, self.num_opt)
        valid_x, valid_y = valid_indices.nonzero()
        props = np.ones([self.step_opt, self.num_opt]) * np.inf
        valid_crys = [c for c in self.crys if c.valid]
        if len(valid_crys) == 0:
            sr_5, sr_10, sr_15 = 0, 0, 0
        else:
            pred_props = prop_model_eval(self.eval_model_name, [
                                         c.dict for c in valid_crys])
            percentiles = Percentiles[self.eval_model_name]
            props[valid_x, valid_y] = pred_props
            best_props = props.min(axis=0)
            sr_5 = (best_props <= percentiles[0]).mean()
            sr_10 = (best_props <= percentiles[1]).mean()
            sr_15 = (best_props <= percentiles[2]).mean()
        return {'SR5': sr_5, 'SR10': sr_10, 'SR15': sr_15}

    def get_metrics(self):
        return self.get_success_rate()


# Labels that use eval_{task}_{label}.{suffix} (no split in filename)
# Add new labels here instead of piling up elif branches
LABELS_OMIT_SPLIT = frozenset({'rruff_20', 'rruff_25', 'LiFePO4', 'SrTiO3', 'SrTiO3_strain', 'GaN', 'NdFe2', 'multi_NdFe2'})


def _structurematcher_kwargs_key(kwargs):
    """Return a JSON-serializable, deterministic key for StructureMatcher kwargs."""
    if isinstance(kwargs, str):
        kwargs = json.loads(kwargs) if kwargs.strip() else {}
    d = {k: v for k, v in (kwargs or {}).items()}
    return json.dumps(d, sort_keys=True)


def get_file_paths(root_path, task, split='test', label='', suffix='pt'):
    if label == '':
        name_part = split
    elif label in LABELS_OMIT_SPLIT or label.startswith('rruff'):
        name_part = label
    else:
        name_part = f'{label}_{split}'
    return os.path.join(root_path, f'eval_{task}_{name_part}.{suffix}')


def get_crystal_array_list(file_path, batch_idx=0):
    data = load_data(file_path)
    if batch_idx == -1:
        batch_size = data['frac_coords'].shape[0]
        crys_array_list = []
        for i in range(batch_size):
            tmp_crys_array_list = get_crystals_list(
                data['frac_coords'][i],
                data['atom_types'][i],
                data['lengths'][i],
                data['angles'][i],
                data['num_atoms'][i])
            crys_array_list.append(tmp_crys_array_list)
    elif batch_idx == -2:
        crys_array_list = get_crystals_list(
            data['frac_coords'],
            data['atom_types'],
            data['lengths'],
            data['angles'],
            data['num_atoms'])        
    else:
        crys_array_list = get_crystals_list(
            data['frac_coords'][batch_idx],
            data['atom_types'][batch_idx],
            data['lengths'][batch_idx],
            data['angles'][batch_idx],
            data['num_atoms'][batch_idx])

    if 'input_data_batch' in data:
        batch = data['input_data_batch']
        if isinstance(batch, dict):
            true_crystal_array_list = get_crystals_list(
                batch['frac_coords'], batch['atom_types'], batch['lengths'],
                batch['angles'], batch['num_atoms'])
        else:
            true_crystal_array_list = get_crystals_list(
                batch.frac_coords, batch.atom_types, batch.lengths,
                batch.angles, batch.num_atoms)
    else:
        true_crystal_array_list = None

    return crys_array_list, true_crystal_array_list


def get_gt_crys_ori(cif):
    structure = Structure.from_str(cif,fmt='cif')
    lattice = structure.lattice
    crys_array_dict = {
        'frac_coords':structure.frac_coords,
        'atom_types':np.array([_.Z for _ in structure.species]),
        'lengths': np.array(lattice.abc),
        'angles': np.array(lattice.angles)
    }
    return Crystal(crys_array_dict) 

def _compute_no_dup_match_rate(recon_file_path, match_indices, n_samples):
    """Deduplicated match rate for RRUFF: one match per unique mineral suffices.

    Returns dict with no_dup_match_rate (and counts) or empty dict if
    material_ids are unavailable.
    """
    data = load_data(recon_file_path)
    material_ids = data.get('material_ids')
    if material_ids is None:
        print("WARNING: material_ids not found in eval .pt — "
              "re-run evaluate.py to include them. Skipping no_dup_match_rate.")
        return {}

    mineral_names = [mid.split('__')[0] for mid in material_ids]
    matched_set = set(match_indices)
    mineral_matched = {}
    for i, name in enumerate(mineral_names):
        if name not in mineral_matched:
            mineral_matched[name] = False
        if i in matched_set:
            mineral_matched[name] = True

    n_unique = len(mineral_matched)
    n_matched = sum(mineral_matched.values())
    rate = n_matched / n_unique if n_unique > 0 else 0.0
    return {
        'no_dup_match_rate': rate,
        'no_dup_matched': n_matched,
        'no_dup_total': n_unique,
    }


def main(args):
    all_metrics = {}

    # Ensure deterministic sampling where we rely on numpy/random (e.g., GenEval).
    random.seed(args.seed)
    np.random.seed(args.seed)

    cfg = load_config(args.root_path)
    eval_model_name = cfg.data.eval_model_name

    if 'opt' in args.tasks:
        opt_file_path = get_file_paths(args.root_path, 'opt', args.label)
        crys_array_list, _ = get_crystal_array_list(opt_file_path)
        opt_crys = p_map(lambda x: Crystal(x), crys_array_list)

        opt_evaluator = OptEval(opt_crys, eval_model_name=eval_model_name)
        opt_metrics = opt_evaluator.get_metrics()
        all_metrics.update(opt_metrics)

    elif 'gen' in args.tasks:

        gen_file_path = get_file_paths(args.root_path, 'gen', args.label)
        recon_file_path = get_file_paths(args.root_path, 'recon', args.label)
        crys_array_list, _ = get_crystal_array_list(gen_file_path, batch_idx = -2)
        gen_crys = p_map(lambda x: Crystal(x), crys_array_list)
        if args.gt_file != '':
            csv = pd.read_csv(args.gt_file)
            gt_crys = p_map(get_gt_crys_ori, csv['cif'])
        else:
            _, true_crystal_array_list = get_crystal_array_list(
                recon_file_path)
            gt_crys = p_map(lambda x: Crystal(x), true_crystal_array_list)
        gen_evaluator = GenEval(
            gen_crys, gt_crys, eval_model_name=eval_model_name)
        gen_metrics = gen_evaluator.get_metrics()
        all_metrics.update(gen_metrics)


    else:

        recon_file_path = get_file_paths(args.root_path, 'diff', args.eval_split, args.label)
        batch_idx = -1 if args.multi_eval else 0
        crys_array_list, true_crystal_array_list = get_crystal_array_list(
            recon_file_path, batch_idx = batch_idx)
        
        # Auto-detect rruff gt_file if label is 'rruff' and gt_file not specified
        gt_file = args.gt_file
        
        if gt_file != '':
            csv = pd.read_csv(gt_file)
            gt_crys = p_map(get_gt_crys_ori, csv['cif'])
        else:
            gt_crys = p_map(lambda x: Crystal(x), true_crystal_array_list)

        if not args.multi_eval:
            pred_crys = p_map(lambda x: Crystal(x), crys_array_list)
        else:
            pred_crys = []
            for i in range(len(crys_array_list)):
                print(f"Processing batch {i}")
                pred_crys.append(p_map(lambda x: Crystal(x), crys_array_list[i]))   

        is_rruff = args.label.startswith('rruff')
        return_match_indices = args.return_match_indices or is_rruff

        if args.multi_eval:
            rec_evaluator = RecEvalBatch(pred_crys, gt_crys, structurematcher_kwargs=args.structurematcher_kwargs, return_match_indices=return_match_indices)
        else:
            rec_evaluator = RecEval(pred_crys, gt_crys, structurematcher_kwargs=args.structurematcher_kwargs, return_match_indices=return_match_indices)

        recon_metrics = rec_evaluator.get_metrics()
        kwargs_key = _structurematcher_kwargs_key(args.structurematcher_kwargs)
        if kwargs_key == '{}':
            kwargs_key = 'structure matcher default'
        entry = {'match_rate': recon_metrics['match_rate'], 'rms_dist': recon_metrics['rms_dist']}
        if 'match_indices' in recon_metrics:
            entry['match_indices'] = recon_metrics['match_indices']

        if is_rruff and 'match_indices' in recon_metrics:
            entry.update(_compute_no_dup_match_rate(
                recon_file_path, recon_metrics['match_indices'],
                len(gt_crys)))

        all_metrics.setdefault('matcher', {})[kwargs_key] = entry

   

    print(all_metrics)

    if args.label == '':
        metrics_out_file = f'eval_metrics_{args.eval_split}.json'
    elif args.label in LABELS_OMIT_SPLIT or args.label.startswith('rruff'):
        metrics_out_file = f'eval_metrics_{args.label}.json'
    elif args.multi_eval:
        metrics_out_file = f'eval_metrics_{args.label}_{args.eval_split}.json'
    else:
        metrics_out_file = f'eval_metrics_{args.label}_{args.eval_split}.json'
    metrics_out_file = os.path.join(args.root_path, metrics_out_file)

    # Merge new metrics into file. Recon metrics are keyed by structurematcher_kwargs.
    def _merge_metrics(existing, new):
        if not isinstance(existing, dict):
            return new
        # Merge matcher dict: each run adds/overwrites its kwargs-keyed entry
        if 'matcher' in new:
            existing.setdefault('matcher', {})
            existing['matcher'].update(new['matcher'])
        # Merge any other top-level keys (e.g. from opt/gen/gen-eval tasks)
        for k, v in new.items():
            if k == 'matcher':
                continue
            existing[k] = v
        return existing

    if Path(metrics_out_file).exists():
        with open(metrics_out_file, 'r') as f:
            written_metrics = json.load(f)
        written_metrics = _merge_metrics(written_metrics, all_metrics)
        with open(metrics_out_file, 'w') as f:
            json.dump(written_metrics, f)
    else:
        with open(metrics_out_file, 'w') as f:
            json.dump(all_metrics, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', required=True)
    parser.add_argument('--eval_split', default='val')
    parser.add_argument('--label', type=str, default='', help='Label for the evaluation (e.g., "rruff"). If None, uses the model\'s original dataset.')
    parser.add_argument('--seed', type=int, default=42, help='RNG seed for deterministic metric sampling.')
    parser.add_argument('--tasks', nargs='+', default=['csp'])
    parser.add_argument('--gt_file',default='')
    parser.add_argument('--multi_eval',action='store_true')
    parser.add_argument('--return_match_indices', action='store_true', 
                       help='Return indices of matched structures in metrics')
    parser.add_argument('--structurematcher_kwargs', default=None, type=lambda x: json.loads(x) if isinstance(x, str) else (x or {}),
                        help='Keyword arguments for StructureMatcher (dict or JSON string, e.g. \'{"stol":0.5,"angle_tol":10,"ltol":0.3}\')')
    args = parser.parse_args()
    if args.structurematcher_kwargs is None:
        args.structurematcher_kwargs = {}
    # make sure if label == multi then also multi_eval is True
    assert args.label != 'multi' or args.multi_eval, "If label is 'multi', multi_eval must be True"
    main(args)
