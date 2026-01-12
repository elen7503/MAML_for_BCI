import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import higher
from pathlib import Path
import torch.nn.functional as F
from scipy.signal import welch
from src.utils import set_seed, sample_target, sample_gain, sample_rotation
from src.models import Policy, EEGClassifier
from src.control import rollout, trajectory_loss
from src.eeg_data import sample_within_subject_task, DEFAULT_K_SUPPORT, DEFAULT_K_QUERY

#Stationary experiment
def train_maml_stationary(seed, meta_iters=500, meta_batch=8, inner_steps=4,
                          inner_lr=0.05, meta_lr=1e-3, T=30, dt=0.1):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meta_policy = Policy(hidden=64).to(device)
    meta_opt = torch.optim.Adam(meta_policy.parameters(), lr=meta_lr)
    meta_losses = []

    for it in range(meta_iters):
        meta_opt.zero_grad()
        batch_loss = 0.0
        tasks = [sample_target(radius=1.5, device=device) for _ in range(meta_batch)]

        for target_xy in tasks:
            inner_opt = torch.optim.SGD(meta_policy.parameters(), lr=inner_lr)

            with higher.innerloop_ctx(meta_policy, inner_opt, copy_initial_weights=False) as (fpolicy, diffopt):
                for _ in range(inner_steps):
                    traj = rollout(fpolicy, target_xy, T=T, dt=dt, theta=0.0, gain=1.0)
                    diffopt.step(trajectory_loss(traj, target_xy))

                traj_q = rollout(fpolicy, target_xy, T=T, dt=dt, theta=0.0, gain=1.0)
                batch_loss += trajectory_loss(traj_q, target_xy)

        batch_loss = batch_loss / meta_batch
        batch_loss.backward()
        meta_opt.step()
        meta_losses.append(batch_loss.item())

    baseline_policy = Policy(hidden=64).to(device)
    return meta_policy, baseline_policy, meta_losses

#Rotation drift
def train_maml_rotation(seed, meta_iters=500, meta_batch=8, inner_steps=4, inner_lr=0.05, meta_lr=1e-3, T=30, dt=0.1, max_angle=0.4):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meta_policy = Policy(hidden=64).to(device)
    meta_opt = torch.optim.Adam(meta_policy.parameters(), lr=meta_lr)
    meta_losses = []

    for it in range(meta_iters):
        meta_opt.zero_grad()
        batch_loss = 0.0

        tasks = [(sample_target(radius=1.5, device=device), sample_rotation(max_angle=max_angle)) for _ in range(meta_batch)]

        for target_xy, theta in tasks:
            inner_opt = torch.optim.SGD(meta_policy.parameters(), lr=inner_lr)

            with higher.innerloop_ctx(meta_policy, inner_opt, copy_initial_weights=False) as (fpolicy, diffopt):
                for _ in range(inner_steps):
                    traj = rollout(fpolicy, target_xy, T=T, dt=dt, theta=theta, gain=1.0)
                    loss = trajectory_loss(traj, target_xy)
                    diffopt.step(loss)

                traj_q = rollout(fpolicy, target_xy, T=T, dt=dt, theta=theta, gain=1.0)
                post_loss = trajectory_loss(traj_q, target_xy)
                batch_loss += post_loss

        batch_loss = batch_loss / meta_batch
        batch_loss.backward()
        meta_opt.step()
        meta_losses.append(batch_loss.item())

    baseline_policy = Policy(hidden=64).to(device)

    return meta_policy, baseline_policy, meta_losses

# Gain drift
def train_maml_gain(seed, meta_iters=500, meta_batch=8, inner_steps=4, inner_lr=0.05, meta_lr=1e-3, T=30, dt=0.1, min_gain=0.5, max_gain=1.5):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meta_policy_gain = Policy(hidden=64).to(device)
    meta_opt_gain = torch.optim.Adam(meta_policy_gain.parameters(), lr=meta_lr)
    meta_losses_gain = []

    for it in range(meta_iters):
        meta_opt_gain.zero_grad()
        batch_loss = 0.0

        tasks = [(sample_target(radius=1.5, device=device), sample_gain(min_gain=min_gain, max_gain=max_gain)) for _ in range(meta_batch)]

        for target_xy, gain in tasks:
            inner_opt = torch.optim.SGD(meta_policy_gain.parameters(), lr=inner_lr)

            with higher.innerloop_ctx(meta_policy_gain, inner_opt, copy_initial_weights=False) as (fpolicy, diffopt):
                for _ in range(inner_steps):
                    traj = rollout(fpolicy, target_xy, T=T, dt=dt, theta=0.0, gain=gain)
                    loss = trajectory_loss(traj, target_xy)
                    diffopt.step(loss)

                traj_q = rollout(fpolicy, target_xy, T=T, dt=dt, theta=0.0, gain=gain)
                post_loss = trajectory_loss(traj_q, target_xy)
                batch_loss += post_loss

        batch_loss = batch_loss / meta_batch
        batch_loss.backward()
        meta_opt_gain.step()
        meta_losses_gain.append(batch_loss.item())

    baseline_policy_gain = Policy(hidden=64).to(device)

    return meta_policy_gain, baseline_policy_gain, meta_losses_gain

# EEG experiment
def train_maml_within_subject_drift(seed: int, train_subjects, meta_iters: int = 300, meta_batch: int = 4,
    inner_steps: int = 5, inner_lr: float = 0.05, meta_lr: float = 1e-3, k_support: int = DEFAULT_K_SUPPORT,
    k_query: int = DEFAULT_K_QUERY, hidden: int = 64, device: torch.device | None = None,):
    set_seed(seed)
    rng = np.random.default_rng(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Infer input dim (d_in)
    d_in = None
    for sid in train_subjects:
        try:
            task = sample_within_subject_task(
                sid,
                k_support=k_support,
                k_query=k_query,
                seed=int(rng.integers(1e9)),
                device=device,
            )
            if task is not None:
                xs, ys, xq, yq = task
                d_in = xs.shape[1]
                break
        except Exception:
            continue

    if d_in is None:
        raise RuntimeError(
            "Could not infer feature dimension. Likely: files missing, "
            "event labels not T1/T2, or runs not present.")

    meta_model = EEGClassifier(d_in=d_in, hidden=hidden).to(device)
    meta_opt = torch.optim.Adam(meta_model.parameters(), lr=meta_lr)

    meta_losses = []
    meta_accs = []

    for it in range(meta_iters):
        meta_opt.zero_grad()
        total_loss = 0.0
        total_acc = 0.0
        tasks_used = 0
        attempts = 0

        while tasks_used < meta_batch and attempts < 500:
            sid = int(rng.choice(train_subjects))
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
            inner_opt = torch.optim.SGD(meta_model.parameters(), lr=inner_lr)

            with higher.innerloop_ctx(meta_model, inner_opt, copy_initial_weights=False) as (fmodel, diffopt):
                for _ in range(inner_steps):
                    logits_s = fmodel(xs)
                    loss_s = F.cross_entropy(logits_s, ys)
                    diffopt.step(loss_s)

                logits_q = fmodel(xq)
                loss_q = F.cross_entropy(logits_q, yq)
                acc_q = (logits_q.argmax(dim=1) == yq).float().mean()

                total_loss = total_loss + loss_q
                total_acc = total_acc + acc_q

            tasks_used += 1
            attempts += 1

        if tasks_used < meta_batch:
            raise RuntimeError(
                "Could not form a full meta-batch. Reduce k_query/k_support "
                "or check data availability for those runs.")

        batch_loss = total_loss / meta_batch
        batch_acc = total_acc / meta_batch

        batch_loss.backward()
        meta_opt.step()

        meta_losses.append(float(batch_loss.item()))
        meta_accs.append(float(batch_acc.item()))

        if (it + 1) % 25 == 0:
            print(f"[{it+1:04d}] meta_loss={batch_loss.item():.4f}  query_acc={batch_acc.item():.3f}")

    baseline_model = EEGClassifier(d_in=d_in, hidden=hidden).to(device)
    return meta_model, baseline_model, meta_losses, meta_accs