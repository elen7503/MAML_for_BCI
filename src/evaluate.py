import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import copy
import numpy as np
import math
import torch.nn.functional as F
from src.utils import sample_target, sample_gain, sample_rotation, get_device_from_model
from src.eeg_data import sample_within_subject_task, DEFAULT_K_SUPPORT, DEFAULT_K_QUERY
from src.control import rollout, trajectory_loss


# Stationary experiment evaluation
def evaluate_adaptation(policy_init, n_tasks=200, steps_list=(0,1,3,5,10), radius=1.5, inner_lr=0.05, T=30, dt=0.1):
    results = {k: [] for k in steps_list}
    device = next(policy_init.parameters()).device
    for _ in range(n_tasks):

        target_xy = sample_target(radius=radius, device=device)

        for k in steps_list:
            pol = copy.deepcopy(policy_init).to(device)
            pol.train()
            opt = torch.optim.SGD(pol.parameters(), lr=inner_lr)

            for _ in range(k):
                traj = rollout(pol, target_xy, T=T, dt=dt)
                loss = trajectory_loss(traj, target_xy)
                opt.zero_grad()
                loss.backward()
                opt.step()

            pol.eval()
            traj = rollout(pol, target_xy, T=T, dt=dt)
            final_loss = trajectory_loss(traj, target_xy).item()
            results[k].append(final_loss)
    means = {k: sum(v)/len(v) for k,v in results.items()}
    return means

def final_losses_at_k(policy_init, n_tasks=200, k=10, radius=1.5, inner_lr=0.05, T=30, dt=0.1):
    vals = []
    device = next(policy_init.parameters()).device
    for _ in range(n_tasks):
        target_xy = sample_target(radius=radius, device=device)
        pol = adapt_k_steps(policy_init, target_xy, k=k, theta=0.0, gain=1.0, inner_lr=inner_lr, T=T, dt=dt)
        traj = rollout(pol, target_xy, T=T, dt=dt, theta=0.0, gain=1.0)
        vals.append(trajectory_loss(traj, target_xy).item())
    return vals

# Rotation draft experiment
def evaluate_adaptation_drift(policy_init, n_tasks=200, steps_list=(0,1,3,5,10), max_angle=0.4, inner_lr=0.05, T=30, dt=0.1, gain=1):
    results = {k: [] for k in steps_list}
    device = next(policy_init.parameters()).device
    for _ in range(n_tasks):
        target_xy = sample_target(radius=1.5, device=device)
        theta = sample_rotation(max_angle=max_angle)

        for k in steps_list:
            device = next(policy_init.parameters()).device
            pol = copy.deepcopy(policy_init).to(device)
            pol.train()
            opt = torch.optim.SGD(pol.parameters(), lr=inner_lr)

            for _ in range(k):
                traj = rollout(pol, target_xy, T=T, dt=dt, theta=theta,gain=gain)
                loss = trajectory_loss(traj, target_xy)
                opt.zero_grad()
                loss.backward()
                opt.step()

            pol.eval()
            traj = rollout(pol, target_xy, T=T, dt=dt, theta=theta, gain=gain)
            final_loss = trajectory_loss(traj, target_xy).item()
            results[k].append(final_loss)

    means = {k: sum(v)/len(v) for k,v in results.items()}
    return means

def final_losses_at_k_rotation(policy_init, n_tasks=200, k=10, max_angle=0.4, inner_lr=0.05, T=30, dt=0.1):
    vals = []
    device = next(policy_init.parameters()).device
    for _ in range(n_tasks):
        target_xy = sample_target(radius=1.5, device=device)
        theta = sample_rotation(max_angle=max_angle)
        pol = adapt_k_steps(policy_init, target_xy, k=k, theta=theta, gain=1.0, inner_lr=inner_lr, T=T, dt=dt)
        traj = rollout(pol, target_xy, T=T, dt=dt, theta=theta, gain=1.0)
        vals.append(trajectory_loss(traj, target_xy).item())
    return vals

# Gain draft evaluation
def evaluate_adaptation_gain(policy_init, n_tasks=200, steps_list=(0,1,3,5,10), min_gain=0.5, max_gain=1.5, inner_lr=0.05, T=30, dt=0.1):
    results = {k: [] for k in steps_list}
    device = next(policy_init.parameters()).device

    for _ in range(n_tasks):
        target_xy = sample_target(radius=1.5, device=device)
        gain = sample_gain(min_gain=min_gain, max_gain=max_gain, device=device)

        for k in steps_list:
            pol = adapt_k_steps(policy_init, target_xy, k=k, theta=0.0, gain=gain,
                                inner_lr=inner_lr, T=T, dt=dt)
            traj = rollout(pol, target_xy, T=T, dt=dt, theta=0.0, gain=gain)
            results[k].append(trajectory_loss(traj, target_xy).item())

    return {k: sum(v)/len(v) for k, v in results.items()}


def final_losses_at_k_gain(policy_init, n_tasks=200, k=10, min_gain=0.5, max_gain=1.5, inner_lr=0.05, T=30, dt=0.1):
    vals = []
    device = next(policy_init.parameters()).device
    for _ in range(n_tasks):
        target_xy = sample_target(radius=1.5, device=device)
        gain = sample_gain(min_gain=min_gain, max_gain=max_gain)
        pol = adapt_k_steps(policy_init, target_xy, k=k, theta=0.0, gain=gain, inner_lr=inner_lr, T=T, dt=dt)
        traj = rollout(pol, target_xy, T=T, dt=dt, theta=0.0, gain=gain)
        vals.append(trajectory_loss(traj, target_xy).item())
    return vals

def param_deltas(policy_init, target_xy, k, theta=0.0, gain=1.0, inner_lr=0.05, T=30, dt=0.1):
    device = next(policy_init.parameters()).device
    pol0 = copy.deepcopy(policy_init).to(device)
    polk = adapt_k_steps(policy_init, target_xy, k=k, theta=theta, gain=gain, inner_lr=inner_lr, T=T, dt=dt)

    deltas = {}
    for (n0, p0), (nk, pk) in zip(pol0.named_parameters(), polk.named_parameters()):
        deltas[n0] = (pk.detach() - p0.detach()).cpu()
    return deltas

# EEG implementation
@torch.no_grad()
def eval_model_on_query(model, xq, yq):
    logits = model(xq)
    loss = F.cross_entropy(logits, yq).item()
    acc = (logits.argmax(dim=1) == yq).float().mean().item()
    return float(loss), float(acc)

def evaluate_within_subject_adaptation(model_init, test_subjects, n_tasks: int = 50, steps_list=(0, 1, 3, 5, 10),
    k_support: int = DEFAULT_K_SUPPORT, k_query: int = DEFAULT_K_QUERY, inner_lr: float = 0.05, seed: int = 0):
    device = get_device_from_model(model_init)
    rng = np.random.default_rng(seed)

    results = {k: {"loss": [], "acc": []} for k in steps_list}

    tasks_done = 0
    attempts = 0
    while tasks_done < n_tasks and attempts < 2000:
        sid = int(rng.choice(test_subjects))
        try:
            task = sample_within_subject_task(
                sid,
                k_support=k_support,
                k_query=k_query,
                seed=int(rng.integers(1e9)),
                device=device,
            )
        except Exception:
            task = None

        if task is None:
            attempts += 1
            continue

        xs, ys, xq, yq = task

        for k in steps_list:
            model = copy.deepcopy(model_init).to(device)
            model.train()
            opt = torch.optim.SGD(model.parameters(), lr=inner_lr)

            for _ in range(k):
                logits_s = model(xs)
                loss_s = F.cross_entropy(logits_s, ys)
                opt.zero_grad()
                loss_s.backward()
                opt.step()

            model.eval()
            loss_q, acc_q = eval_model_on_query(model, xq, yq)
            results[k]["loss"].append(loss_q)
            results[k]["acc"].append(acc_q)

        tasks_done += 1
        attempts += 1

    if tasks_done < n_tasks:
        raise RuntimeError(
            "Could not collect enough eval tasks. Reduce k_query/k_support "
            "or verify test subjects have the required runs.")

    summary = {}
    for k in steps_list:
        loss_arr = np.asarray(results[k]["loss"], dtype=np.float64)
        acc_arr  = np.asarray(results[k]["acc"], dtype=np.float64)

        summary[k] = {
            "loss_mean": float(loss_arr.mean()),
            "loss_sem": float(loss_arr.std(ddof=1) / math.sqrt(len(loss_arr))),
            "acc_mean": float(acc_arr.mean()),
            "acc_sem": float(acc_arr.std(ddof=1) / math.sqrt(len(acc_arr))),
        }
    return summary 