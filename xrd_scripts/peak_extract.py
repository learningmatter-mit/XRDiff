"""XRD peak detection and prominence, with visualization."""
import sys
import os
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks
import hydra
from omegaconf import OmegaConf

script_dir = Path(__file__).parent
PROJECT_ROOT = script_dir.parent
sys.path.insert(0, str(PROJECT_ROOT))

from diffcsp.common import utils

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


def load_data(dataset_name, xrd_bins=0.1, xrd_type='full'):
    """Load XRD arrays and material IDs, organized by split.

    For RRUFF: reads directly from CSV so all samples are included
    (the .pt cache drops samples that fail structure graph processing).
    For other datasets: uses the Hydra datamodule pipeline.

    Returns:
        dict mapping split name -> (xrd_arrays, material_ids)
            xrd_arrays: list of 1D numpy arrays
            material_ids: list of strings
    """
    if dataset_name.startswith('rruff'):
        return _load_rruff_from_csv(dataset_name, xrd_bins)
    return _load_via_hydra(dataset_name, xrd_bins, xrd_type)


def _load_rruff_from_csv(dataset_name='rruff_20', xrd_bins=0.1):
    csv_path = PROJECT_ROOT / "data" / "rruff" / f"{dataset_name}_data.csv"
    df = pd.read_csv(csv_path)
    xrd_col = f'xrd_{xrd_bins}'
    xrd_arrays = [np.array(json.loads(row)) for row in df[xrd_col]]
    material_ids = df['material_id'].tolist()
    sample_labels = df['mineral_name'].tolist()
    print(f"Loaded {len(xrd_arrays)} RRUFF samples from CSV, XRD dim: {len(xrd_arrays[0])}")
    return {'test': (xrd_arrays, material_ids, sample_labels)}


def _load_via_hydra(dataset_name, xrd_bins=0.1, xrd_type='full'):
    if 'PROJECT_ROOT' not in os.environ:
        os.environ['PROJECT_ROOT'] = str(PROJECT_ROOT)

    cfg_path = PROJECT_ROOT / "conf" / "data" / f"{dataset_name}.yaml"
    data_cfg = OmegaConf.create({'data': OmegaConf.load(cfg_path)})
    ds_cfg = data_cfg.data.datamodule.datasets
    has_train = "train" in ds_cfg
    has_val = "val" in ds_cfg
    has_test = "test" in ds_cfg

    if xrd_bins is not None:
        data_cfg.data.xrd_bins = xrd_bins
        if has_train:
            ds_cfg.train.xrd_bins = xrd_bins
        for split in ('val', 'test'):
            if split in ds_cfg:
                for ds in ds_cfg[split]:
                    ds.xrd_bins = xrd_bins

    if xrd_type is not None:
        data_cfg.data.xrd_type = xrd_type
        if has_train:
            ds_cfg.train.xrd_type = xrd_type
        for split in ('val', 'test'):
            if split in ds_cfg:
                for ds in ds_cfg[split]:
                    ds.xrd_type = xrd_type

    OmegaConf.resolve(data_cfg)
    datamodule = hydra.utils.instantiate(data_cfg.data.datamodule, _recursive_=False)

    def _extract(dataset):
        for d in dataset.cached_data:
            d.setdefault(dataset.prop, 0.0)
        xrd = [dataset[i]['xrd'].squeeze().cpu().numpy()
               for i in range(len(dataset))]
        ids = [item.get('mp_id', item.get('material_id', 'unknown'))
               for item in dataset.cached_data]
        return xrd, ids, ids

    splits = {}

    if has_train:
        datamodule.setup('fit')
        splits['train'] = _extract(datamodule.train_dataset)
        if has_val and datamodule.val_datasets:
            splits['val'] = _extract(datamodule.val_datasets[0])

    if has_test and dataset_name != 'alex_mp_20_uni':
        datamodule.setup('test')
        if datamodule.test_datasets:
            splits['test'] = _extract(datamodule.test_datasets[0])

    if not splits:
        raise ValueError(f"No data splits found for dataset '{dataset_name}'.")

    first_xrd = next(iter(splits.values()))[0]
    print(f"XRD dimension: {len(first_xrd[0])}")
    return splits


def apply_artifacts(xrd, artifacts, xrd_bins=0.1):
    """Apply experimental-like artifacts to a simulated XRD pattern.

    Args:
        xrd: 1D array, the XRD intensities (assumed max-normalized ~1).
        artifacts: list of str, artifact types to apply ('background', 'broadening').
        xrd_bins: float, bin width in degrees.

    Returns:
        Modified XRD array (re-normalized to [0, 1]).
    """
    xrd = xrd.copy().astype(float)
    two_theta = np.arange(5, 5 + len(xrd) * xrd_bins, xrd_bins)[:len(xrd)]

    if 'broadening' in artifacts:
        from scipy.ndimage import gaussian_filter1d
        sigma_deg = 0.25
        sigma_bins = sigma_deg / xrd_bins
        broadened = gaussian_filter1d(xrd, sigma=sigma_bins)
        xrd = np.maximum(xrd, broadened)

    if 'background' in artifacts:
        decay = 0.1
        amplitude = 0.05 * xrd.max()
        bg = amplitude * np.exp(-decay * (two_theta - two_theta[0]))
        xrd = xrd + bg

    peak = xrd.max()
    if peak > 0:
        xrd = xrd / peak
    return xrd


def get_peak_info(xrd, height=0, width=0, prominence=0):
    """Get peak information: intensity, position, prominence, width."""
    peaks, props = find_peaks(xrd, height=height, width=width, prominence=prominence)
    return xrd[peaks], peaks, props['prominences'], props['widths']


def plot_xrd_with_peaks(xrd, sample_label, height=0, width=0, prominence=0,
                        xrd_bins=0.1, max_peaks=20, artifacts=None):
    """Plot XRD pattern with detected peaks on a 2theta axis.
    """
    _, positions, prominences, _ = get_peak_info(xrd, height, width, prominence)
    two_theta = np.arange(5, 5 + len(xrd) * xrd_bins, xrd_bins)[:len(xrd)]

    if len(positions) > max_peaks:
        top_idx = np.argsort(prominences)[-max_peaks:]
        positions = positions[top_idx]
        prominences = prominences[top_idx]

    contour_heights = xrd[positions] - prominences

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(two_theta, xrd, 'b-', linewidth=0.8,
            label=f'XRD pattern {sample_label}')
    ax.plot(two_theta[positions], xrd[positions], "rx", markersize=6,
            markeredgewidth=1.5, label=f'Top {len(positions)} peaks')
    ax.vlines(two_theta[positions], contour_heights, xrd[positions],
              colors='red', alpha=0.5, linewidth=1, linestyles='--')
    ax.set_xlabel(r'2$\theta$ (°)', fontsize=12)
    ax.set_ylabel('Intensity', fontsize=12)
    ax.set_xlim(5, 90)
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()


def extract_peaks_info(xrd_arrays, material_ids, height=0, width=0, prominence=0):
    """Extract peak information for all XRD patterns.

    Args:
        xrd_arrays: list of 1D numpy arrays (XRD patterns).
        material_ids: list of material ID strings.
    """
    peak_info = []
    for xrd, mid in zip(xrd_arrays, material_ids):
        intensity, position, prominence_vals, width_vals = get_peak_info(
            xrd, height, width, prominence)
        position_norm = position / (len(xrd) - 1)
        width_norm = (width_vals / width_vals.max()
                      if len(width_vals) > 0 and width_vals.max() > 0
                      else np.array([]))
        peak_info.append({
            'material_id': mid,
            'peak_intensity': json.dumps(intensity.tolist()),
            'peak_position': json.dumps(position_norm.tolist()),
            'peak_prominence': json.dumps(prominence_vals.tolist()),
            'peak_width': json.dumps(width_norm.tolist()),
            'num_peaks': len(position),
        })
    return peak_info


def main(dataset_name, num_samples_for_visualization=5, idx_for_vis=0,
         height=0, width=0, prominence=0,
         xrd_bins=0.1, xrd_type='full', extract_peak_info=False,
         artifacts=None):
    """Visualize XRD peaks and extract peak information."""
    print(f"\n{'='*60}\nXRD Peak Detection Visualization\n"
          f"Dataset: {dataset_name}\n{'='*60}\n")

    splits = load_data(dataset_name, xrd_bins=xrd_bins, xrd_type=xrd_type)
    # get the rruff stem, and if it's not rruff, use the dataset name
    dataset_name_stem = dataset_name.split('_')[0] if dataset_name.startswith('rruff') else dataset_name

    if artifacts:
        print(f"Applying artifacts: {artifacts}")
        for key in splits:
            xrd_arrays, mids, labels = splits[key]
            xrd_arrays = [apply_artifacts(x, artifacts, xrd_bins) for x in xrd_arrays]
            splits[key] = (xrd_arrays, mids, labels)

    # Use the first available split for visualization.
    primary_name = next(iter(splits))
    xrd_arrays, material_ids, sample_labels = splits[primary_name]

    if num_samples_for_visualization > 0 or idx_for_vis is not None:
        if idx_for_vis is not None:
            num_samples = 1
        else:
            num_samples = min(num_samples_for_visualization, len(xrd_arrays))
        plot_dir = PROJECT_ROOT / "xrd_scripts" / "peak_plots" / dataset_name_stem
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(num_samples):
            if idx_for_vis is not None:
                i = idx_for_vis
            xrd, mid, label = xrd_arrays[i], material_ids[i], sample_labels[i]
            print(f"Processing sample {i+1}/{num_samples}: {mid}")

            plot_xrd_with_peaks(xrd, label, height, width, prominence,
                                xrd_bins=xrd_bins, artifacts=artifacts)
            plt.savefig(plot_dir / f"{mid}_peaks.png", dpi=150, bbox_inches='tight')
            plt.close()

            _, pos, prom, _ = get_peak_info(xrd, height, width, prominence)
            print(f"  Found {len(pos)} peaks")
            if len(pos) > 0:
                print(f"  Peak positions: {pos[:10].tolist()}"
                      f"{'...' if len(pos) > 10 else ''}")
                print(f"  Prominence range: [{prom.min():.4f}, {prom.max():.4f}]")
                print(f"  Mean prominence: {prom.mean():.4f}")
            print()

        print(f"Plots saved to {plot_dir}")

    # Extract and save peak information for every split.
    if extract_peak_info:
        print("\nExtracting peak information for all samples...")
        out_dir = PROJECT_ROOT / "data" / dataset_name_stem
        out_dir.mkdir(parents=True, exist_ok=True)

        for split_name, (split_xrd, split_ids, _labels) in splits.items():
            peak_info = extract_peaks_info(split_xrd, split_ids, height, width, prominence)
            if dataset_name.startswith('rruff'):
                out_path = out_dir / f"{dataset_name}_data_peak_info.csv"
            else:
                out_path = out_dir / f"{split_name}_peak_info.csv"
            pd.DataFrame(peak_info).to_csv(out_path, index=False)
            print(f"Saved {split_name} peak info ({len(peak_info)} samples) to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Visualize XRD patterns with peak detection')
    parser.add_argument('--dataset', type=str, default='alex_mp_20_uni',
                        help='Dataset name (e.g., "rruff_20", "rruff_25", "alex_mp_20_poly")')
    parser.add_argument('--num_samples_for_visualization', type=int, default=0,
                        help='Number of samples to plot')
    parser.add_argument('--height', type=float, default=0,
                        help='Minimum peak height')
    parser.add_argument('--width', type=float, default=0,
                        help='Minimum peak width')
    parser.add_argument('--prominence', type=float, default=0,
                        help='Minimum peak prominence')
    parser.add_argument('--xrd_bins', type=float, default=0.1,
                        help='XRD binning (0.01 or 0.1)')
    parser.add_argument('--xrd_type', type=str, default='full',
                        choices=['full', 'peaks'],
                        help='XRD type: "full" for full pattern, "peaks" for peak features')
    parser.add_argument('--idx_for_vis', type=int, default=None,
                        help='Index for visualization')
    parser.add_argument('--extract_peak_info', action='store_true',
                        help='Extract peak information')
    parser.add_argument('--add_artifact', nargs='+', default=None, type=str,
                        choices=['background', 'broadening'],
                        help='Artifacts to apply (e.g. --add_artifact background broadening)')

    args = parser.parse_args()
    main(args.dataset, args.num_samples_for_visualization, args.idx_for_vis,
         args.height, args.width, args.prominence,
         xrd_bins=args.xrd_bins, xrd_type=args.xrd_type,
         extract_peak_info=args.extract_peak_info,
         artifacts=args.add_artifact)
