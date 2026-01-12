import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import math
from pathlib import Path
import torch.nn.functional as F
from scipy.signal import welch

# Sample funtions
def sample_rotation(max_angle=0.4, device="cpu"):
    return torch.empty(1, device=device).uniform_(-max_angle, max_angle)

def sample_gain(min_gain=0.5, max_gain=1.5, device="cpu"):
    return torch.empty(1, device=device).uniform_(min_gain, max_gain)

def sample_target(radius=1.5, device="cpu"):
    x = (2 * torch.rand(1, device=device) - 1) * radius
    y = (2 * torch.rand(1, device=device) - 1) * radius
    return torch.cat([x, y], dim=0)


# Mean and SEM
def mean_sem(x):
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean()
    sem = x.std(ddof=1) / math.sqrt(len(x))
    return mean, sem

# Seeds
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def mean_sem_across_seeds(values):
    arr = np.asarray(values, dtype=np.float64)
    mean = arr.mean(axis=0)
    sem  = arr.std(axis=0, ddof=1) / math.sqrt(arr.shape[0])
    return mean, sem

# For EEG
# Paths
PROJECT_ROOT = Path("..")
DATA_ROOT = PROJECT_ROOT / "files"

# EEG configuration
CHANNELS = ["C3", "Cz", "C4"]
BANDS = [(8, 12), (13, 30)]
EPOCH_TMIN, EPOCH_TMAX = 0.0, 2.0

# Which runs to use
SUPPORT_RUNS = [4, 8]
QUERY_RUNS = [12]

# Default few-shot sizes
DEFAULT_K_SUPPORT = 8
DEFAULT_K_QUERY   = 20

def get_device_from_model(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device