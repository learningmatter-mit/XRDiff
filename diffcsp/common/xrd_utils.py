import numpy as np
import pandas as pd
import os
import pickle as pkl
from tqdm import tqdm
import json

from pymatgen.core import Structure
from pymatgen.io.cif import CifParser
from pymatgen.analysis.diffraction.xrd import XRDCalculator
xrd_calculator = XRDCalculator(wavelength='CuKa', symprec=0.1)
import warnings
warnings.filterwarnings(
    "ignore",
    message="Issues encountered while parsing CIF",
    module="pymatgen.io.cif",
    category=UserWarning,
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def caglioti_fwhm(theta, U, V, W):
    """
    Calculate the FWHM using the Caglioti formula.
    theta: float, the angle in degrees
    U, V, W: Caglioti parameters
    """
    rad_theta = np.radians(theta / 2)  # Convert theta to radians
    return (U * np.tan(rad_theta)**2 + V * np.tan(rad_theta) + W)**0.5

def pseudo_voigt(x, center, amplitude, U, V, W, eta):
    """
    Pseudo-Voigt function using Caglioti FWHM.
    x: array-like, the independent variable
    center: float, the center of the peak
    amplitude: float, the height of the peak
    U, V, W: Caglioti parameters
    eta: float, the fraction of the Lorentzian component (0 <= eta <= 1)
    """
    fwhm = caglioti_fwhm(center, U, V, W)
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to sigma for Gaussian
    
    lorentzian = amplitude * (fwhm**2 / ((x - center)**2 + fwhm**2))
    gaussian = amplitude * np.exp(-(x - center)**2 / (2 * sigma**2))
    return eta * lorentzian + (1 - eta) * gaussian


def superimposed_pseudo_voigt(x, xy_merge, U, V, W, eta):
    """
    Superimpose multiple pseudo-Voigt functions using Caglioti FWHM.
    x: array-like, the independent variable
    xy_merge: nx2 array, first column is peak locations, second column is intensities
    U, V, W: Caglioti parameters
    eta: float, the fraction of the Lorentzian component (0 <= eta <= 1)
    """
    total = np.zeros_like(x)
    for row in xy_merge:
        center, amplitude = row
        total += pseudo_voigt(x, center, amplitude, U, V, W, eta)
    total = total / max(total)
    return total


def simulate_pv_xrd_for_row(xy_merge, U, V, W, eta, bin=0.01):
    """
    Simulate a pseudo-Voigt XRD pattern for a given set of peaks.

    Args:
    xy_merge: array-like, the 2D array of peak locations and intensities
    U, V, W: float, the Caglioti parameters
    noise: float, the standard deviation of the noise

    Returns:
    sim_xrd: array-like, the simulated XRD pattern
    """

    x = np.arange(5, 90, bin)
    sim_xrd = superimposed_pseudo_voigt(x, xy_merge, U, V, W, eta)

    return sim_xrd


def get_sim_xrd_from_pattern(pattern, eta, caglioti_params, bin=0.01):
    """
    Get the simulated XRD pattern for pymatgen generated pattern object
    """
    x = np.array(pattern.x)
    y = np.array(pattern.y)
    

    # # Get indices of top 20 intensities
    # top_idx = np.argsort(y)[-20:]
    # # Sort those indices by x (2-theta) for consistency
    # top_idx_sorted = top_idx[np.argsort(x[top_idx])]
    # x = x[top_idx_sorted]
    # y = y[top_idx_sorted]

    xy_merge = np.column_stack((x, y))

    U, V, W = caglioti_params[0], caglioti_params[1], caglioti_params[2] 

    sim_xrd = simulate_pv_xrd_for_row(xy_merge, U, V, W, eta, bin=bin)
    return sim_xrd


def save_results_pickle(results, filename):
    with open(filename, 'wb') as f:
        pkl.dump(results, f)


def get_sim_xrd_from_preprocessed_results(data, eta, caglioti_params, index=None, bin=0.01):
    """
    Get the simulated XRD pattern using the structure directly from preprocessed results.

    Parameters:
    preprocessed_results: list of dictionaries from preprocess() function
    index: int, the index of the crystal in the list

    Returns:
    numpy array: simulated XRD pattern
    """
    if index is not None:
        cif = data.loc[index, "cif"]
        structure = Structure.from_str(cif, fmt="cif")
        pattern = xrd_calculator.get_pattern(structure)
        sim_xrd = get_sim_xrd_from_pattern(pattern, eta, caglioti_params, bin=bin) #simulates
        data.loc[index, f'xrd_{bin}'] = json.dumps(sim_xrd.tolist())
    else: 
        structure = data
        pattern = xrd_calculator.get_pattern(structure)
        sim_xrd = get_sim_xrd_from_pattern(pattern, eta, caglioti_params, bin=bin) #simulates
        return sim_xrd.tolist()

def main(datasets, bins):
    for dataset in datasets:
        DATA_DIR = os.path.join(BASE_DIR, 'data', dataset)
        train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
        val = pd.read_csv(os.path.join(DATA_DIR, 'val.csv'))
        test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

        eta = 0.1 # eta=0 means all Gaussian contribution, eta=1 means all Lorenzian
        U, V, W = 0.1, 0.01, 0.1
        caglioti_params = [U,V,W]

        def process_and_save(data, split_name):
            for i in tqdm(range(len(data))):
                get_sim_xrd_from_preprocessed_results(data, eta, caglioti_params, index=i, bin=bins)
            
            # Save the modified DataFrame to CSV
            data.to_csv(os.path.join(DATA_DIR, f"{split_name}.csv"), index=False)
            print(f'{dataset} {split_name} with xrd saved')
        
        # Process train, val, and test datasets
        process_and_save(train, 'train')
        process_and_save(val, 'val')
        process_and_save(test, 'test')


if __name__ == '__main__':
    datasets = ['oqmd_full', 'alex_mp_20']
    bins = 0.1 # 0.01 or 0.1
    main(datasets, bins)