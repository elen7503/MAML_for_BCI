import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from scipy.signal import welch

# Simple policy network
class Policy(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2),
        )

    def forward(self, s):
        return self.net(s)

#EEG Classifier
class EEGClassifier(nn.Module):
    def __init__(self, d_in: int, hidden: int = 64, n_out: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_out),
        )

    def forward(self, x):
        return self.net(x)