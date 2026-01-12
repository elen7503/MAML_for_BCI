import torch

def rollout(policy, target_xy, T=25, dt=0.1, theta=0.0, gain=1.0):
    device = next(policy.parameters()).device
    x = torch.zeros(2, device=device)
    traj = [x]

    theta_t = theta.to(device).squeeze() if torch.is_tensor(theta) else torch.tensor(theta, device=device)
    gain_t  = gain.to(device).squeeze()  if torch.is_tensor(gain)  else torch.tensor(gain, device=device)

    c = torch.cos(theta_t)
    s = torch.sin(theta_t)
    R = torch.stack([torch.stack([c, -s]), torch.stack([s,  c])], dim=0)

    # ensure target on device
    if hasattr(target_xy, "to"):
        target_xy = target_xy.to(device)

    for _ in range(T):
        state = torch.cat([x, target_xy], dim=0).unsqueeze(0)
        a = torch.tanh(policy(state).squeeze(0))
        a = gain_t * (R @ a)
        x = x + dt * a
        traj.append(x)

    return torch.stack(traj, dim=0)

def trajectory_loss(traj, target_xy, action_penalty=1e-3):
    device = traj.device
    if hasattr(target_xy, "to"):
        target_xy = target_xy.to(device)
    final_dist2 = torch.sum((traj[-1] - target_xy) ** 2)
    smoothness = torch.mean(torch.sum((traj[1:] - traj[:-1])**2, dim=1))
    return final_dist2 + action_penalty * smoothness
