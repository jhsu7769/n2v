"""3D flow-reachset visualization against Star-union ground truth.

Trains a flow on `ThreeBlobClassifier3D`, builds the conformal probabilistic
reachset, and renders it as a marching-cubes isosurface alongside the
Star-union ground-truth mesh. The two meshes are drawn on the same grid
so visual alignment is meaningful (red = flow reachset, blue = Star union
ground truth).

Writes `figures/three_blob_3d_flow_vs_star_union.html` (< 500 KB with
CDN plotly; opens in any browser, and in VSCode Live Preview).
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

from examples.FlowConformal.networks import ThreeBlobClassifier3D
from examples.FlowConformal.benchmarks._common import (
    _forward, exact_star_union_volume,
)
from n2v.probabilistic.flow.calibrate import calibrate
from n2v.probabilistic.flow.model import VelocityField
from n2v.probabilistic.flow.ode import FlowODE
from n2v.probabilistic.flow.sampling import sample_l_inf_ball
from n2v.probabilistic.flow.scores import FlowScore
from n2v.probabilistic.flow.sets import ProbabilisticSet
from n2v.probabilistic.flow.star_viz import render_probabilistic_set_isosurface_3d
from n2v.probabilistic.flow.train import train_flow


def main():
    alpha = 0.01
    seed = 0
    n_train = 10_000
    n_calib = 2_000
    flow_epochs = 5000  # Use the best sweep config: hidden=256, L=6, sin-t, 5000ep

    torch.manual_seed(seed)
    net = ThreeBlobClassifier3D().eval()
    x_center = np.zeros(3)
    radius = 1.0

    print('Computing Star-union ground truth...')
    star_vol, stars = exact_star_union_volume(
        net, x_center=x_center, radius=radius, output_dim=3, n_mc=200_000,
    )
    print(f'  n_stars = {len(stars)}  Star-union volume = {star_vol:.2f}')

    print('Sampling inputs and pushing through net...')
    x_center_t = torch.as_tensor(x_center, dtype=torch.float32)
    x_tr = sample_l_inf_ball(x_center=x_center_t, radius=radius,
                                n_samples=n_train, seed=seed, dim=3)
    x_ca = sample_l_inf_ball(x_center=x_center_t, radius=radius,
                                n_samples=n_calib, seed=seed + 1_000_000, dim=3)
    y_tr = _forward(net, x_tr)
    y_ca = _forward(net, x_ca)

    print(f'Training flow (bigger net, sinusoidal-t, {flow_epochs} epochs)...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    vf = VelocityField(dim=3, hidden=256, n_layers=6, activation='silu',
                       time_embed='sinusoidal').to(device)
    t0 = time.time()
    vf, _ = train_flow(
        vf, y_tr.to(device), n_epochs=flow_epochs, batch_size=2048, lr=1e-3,
        coupling='sinkhorn', sinkhorn_reg='auto', sinkhorn_iters=20,
        use_ema=True, standardize_outputs=True,
    )
    flow = FlowODE(vf.eval())
    print(f'  trained in {time.time() - t0:.1f}s')

    # Calibrate the flow score
    ell = int(math.ceil((n_calib + 1) * (1 - alpha)))
    score_fn = FlowScore(flow, t=1.0, n_steps=30, method='rk4', batch_size=65536)
    thresh = calibrate(score_fn(y_ca), ell).item()
    prob_set = ProbabilisticSet(
        score_fn=score_fn, threshold=thresh,
        m=n_calib, ell=ell, epsilon=alpha, dim=3,
    )
    print(f'  flow threshold q = {thresh:.3f}')

    # Bounding box from the data envelope
    y_all = torch.cat([y_tr, y_ca], dim=0)
    lo = y_all.min(dim=0).values
    hi = y_all.max(dim=0).values
    pad = 0.1 * (hi - lo).clamp(min=1e-6)
    bbox = (lo - pad, hi + pad)

    # Forward samples for the scatter overlay (distinct from y_tr)
    x_samp = sample_l_inf_ball(x_center=x_center_t, radius=radius,
                                  n_samples=2000, seed=seed + 3_000_000, dim=3)
    y_samp = _forward(net, x_samp).cpu().numpy()

    out_html = Path(__file__).parent / 'figures' / 'three_blob_3d_flow_vs_star_union.html'
    print(f'Rendering 64^3 isosurface ({out_html.name}) ...')
    t0 = time.time()
    render_probabilistic_set_isosurface_3d(
        prob_set=prob_set,
        bounding_box=bbox,
        forward_samples=y_samp,
        star_meshes=stars,
        title='ThreeBlobClassifier3D: flow conformal reachset (red) vs Star-union (blue)',
        out_html=out_html,
        resolution=64,
    )
    print(f'  render in {time.time() - t0:.1f}s')
    print(f'Saved {out_html}')


if __name__ == '__main__':
    main()
