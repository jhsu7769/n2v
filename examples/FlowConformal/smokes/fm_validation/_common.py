"""Shared helpers for Phase 0B qualitative FM benchmarks.

Produces the 4-panel figure used by all three benchmarks:
(a) training data, (b) generated samples, (c) trajectories, (d) density contours.
"""

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from n2v.probabilistic.flow.model import VelocityField
from n2v.probabilistic.flow.ode import FlowODE
from n2v.probabilistic.flow.train import train_flow


def train_and_render(target_sampler, title, out_path, n_epochs=600,
                     n_data=4000, seed=0, n_samples=2000, n_trajectories=30):
    torch.manual_seed(seed)
    data = target_sampler(n_data, seed)
    d = data.shape[1]

    vf = VelocityField(dim=d, hidden=128, n_layers=4, activation='silu')
    vf, losses = train_flow(
        vf, data, n_epochs=n_epochs, batch_size=256, lr=1e-3,
        coupling='hungarian', use_ema=True,
    )
    vf.eval()
    flow = FlowODE(vf)

    # Generate samples
    z0 = torch.randn(n_samples, d)
    with torch.no_grad():
        y_gen = flow.inverse(z0, t=1.0, n_steps=100)

    # Trajectories: integrate from noise to data, recording intermediates
    z_traj = torch.randn(n_trajectories, d)
    t_grid = torch.linspace(0, 1, 20)
    trajs = [z_traj.clone().numpy()]
    for i in range(1, len(t_grid)):
        with torch.no_grad():
            trajs.append(flow.inverse(z_traj, t=t_grid[i].item(), n_steps=50).numpy())
    trajs = np.stack(trajs)  # (T, n_traj, d)

    # Density: learned score is ||phi(y)||_2; contour of -log chi_d pdf is
    # nontrivial, just render the score itself as "typicality" contour.
    x_min, x_max = data[:, 0].min().item() - 0.5, data[:, 0].max().item() + 0.5
    y_min, y_max = data[:, 1].min().item() - 0.5, data[:, 1].max().item() + 0.5
    xs = np.linspace(x_min, x_max, 60)
    ys = np.linspace(y_min, y_max, 60)
    xx, yy = np.meshgrid(xs, ys)
    grid = torch.tensor(np.stack([xx.ravel(), yy.ravel()], axis=1), dtype=torch.float32)
    with torch.no_grad():
        z_grid = flow.forward(grid, t=1.0, n_steps=100)
        score = z_grid.norm(dim=1).numpy()
    score = score.reshape(xx.shape)

    # 4-panel figure
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].scatter(data[:, 0], data[:, 1], s=3, alpha=0.3)
    axes[0].set_title('(a) training data'); axes[0].set_aspect('equal')
    axes[1].scatter(y_gen[:, 0], y_gen[:, 1], s=3, alpha=0.3, color='C1')
    axes[1].set_title('(b) generated samples'); axes[1].set_aspect('equal')
    for ti in range(trajs.shape[1]):
        axes[2].plot(trajs[:, ti, 0], trajs[:, ti, 1], alpha=0.5, linewidth=1)
    axes[2].set_title('(c) trajectories t=0 to t=1'); axes[2].set_aspect('equal')
    cs = axes[3].contourf(xx, yy, score, levels=20, cmap='viridis')
    axes[3].scatter(data[:, 0], data[:, 1], s=1, alpha=0.2, color='white')
    plt.colorbar(cs, ax=axes[3])
    axes[3].set_title('(d) ||phi(y)||_2 contour + data')
    axes[3].set_aspect('equal')
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    # Numerical summary
    gen_mean = y_gen.mean(dim=0)
    gen_std = y_gen.std(dim=0)
    data_mean = data.mean(dim=0)
    data_std = data.std(dim=0)
    print(f"Saved {out_path}")
    print(f"  data mean = {data_mean.tolist()}, gen mean = {gen_mean.tolist()}")
    print(f"  data std  = {data_std.tolist()}, gen std  = {gen_std.tolist()}")
    print(f"  final loss = {losses[-1]:.6f}")
