import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import copy
import numpy as np
import torch.nn.functional as F
from src.control import rollout, trajectory_loss

# Steps adaptation
def adapt_k_steps(policy_init, target_xy, k, theta=0.0, gain=1.0, inner_lr=0.05, T=30, dt=0.1):
    device = next(policy_init.parameters()).device
    pol = copy.deepcopy(policy_init).to(device)
    pol.train()
    opt = torch.optim.SGD(pol.parameters(), lr=inner_lr)

    for _ in range(k):
        traj = rollout(pol, target_xy, T=T, dt=dt, theta=theta, gain=gain)
        loss = trajectory_loss(traj, target_xy)
        opt.zero_grad()
        loss.backward()
        opt.step()

    pol.eval()
    return pol