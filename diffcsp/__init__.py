"""
DiffCSP package. Suppress noisy Hydra/third-party warnings on import.
Import this first so filters apply before any Hydra/dataset code loads.
"""
import warnings

# Hydra / OmegaConf
warnings.filterwarnings("ignore", message=".*version_base.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Defaults list is missing.*_self_.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*compose.*is no longer experimental.*")
warnings.filterwarnings("ignore", module="hydra._internal.defaults_list")

# PyTorch / PyG
warnings.filterwarnings("ignore", message=".*DataLoader.*deprecated.*")
warnings.filterwarnings("ignore", message=".*Creating a tensor from a list.*")
warnings.filterwarnings("ignore", message=".*To copy construct from a tensor.*")

# Pymatgen
warnings.filterwarnings("ignore", module="pymatgen.analysis.local_env", category=UserWarning)
