"""Figure 7 — 2D banana score-geometry visualization (standalone).

Trains a small flow on the output distribution of RotatedBananaNet
(uniform input on [0, 1]^2 ↦ banana strip in R^2) and overlays the
calibrated reach sets of four score families on a single axes:

  - Hyper-rectangle (Hashemi-naive)         : red filled polygon
  - Ellipsoid (Mahalanobis)                  : orange filled polygon
  - GMM (k=10)                               : blue filled polygon
  - Flow (ours)                              : green filled polygon
  - Exact reach set (n2v Star propagation)   : solid black outline

Each sublevel set is rasterized over a uniform 2D grid, then turned
into a contour at the calibrated threshold. All sets are drawn with
the same alpha so overlaps are visible.

This is a standalone script — no CSV input required. The flow is
trained from scratch every run (~30 s on CPU, ~5 s on GPU). Pass
``--epochs N`` to override the default 100 (smoke runs use 20).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Project-root sys.path setup (so this script can be invoked directly).
PAPER_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PAPER_DIR.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "examples") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "examples"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import add_common_args, apply_paper_style, save_figure  # noqa: E402

from FlowConformal.networks import RotatedBananaNet  # noqa: E402
from n2v.probabilistic.flow import FlowODE, VelocityField  # noqa: E402
from n2v.probabilistic.flow.calibrate import calibrate  # noqa: E402
from n2v.probabilistic.flow.scores import (  # noqa: E402
    EllipsoidScore, FlowScore, GMMScore, HyperrectScore,
)


# Method visualization metadata.
#
#   - ``zorder``: drawing order. Smallest set drawn on top so the inside
#     is visible through the larger sets' translucent fills. The bigger
#     a set typically is, the lower its zorder.
#   - ``alpha``: outer (looser) sets get lower alpha so they don't
#     swamp the figure; tighter sets get higher alpha so they read as
#     the visual focus.
METHOD_PLOT = {
    "hyperrect": {"label": "Hyper-rectangle (Hashemi-naive)",
                  "color": "#e89090",   # light red — looser set
                  "alpha": 0.18,
                  "zorder": 1},
    "ellipsoid": {"label": "Ellipsoid (Mahalanobis)",
                  "color": "#f4b683",   # light orange
                  "alpha": 0.22,
                  "zorder": 2},
    "gmm":       {"label": "GMM ($k{=}3$)",
                  "color": "#7fb1d3",   # mid blue
                  "alpha": 0.45,
                  "zorder": 3},
    "flow":      {"label": "Flow (ours)",
                  "color": "#1b9e3a",   # green
                  "alpha": 0.55,
                  "zorder": 4},
}

# Color used for the exact reach-set outline. Slightly softened from
# pure black so it doesn't visually dominate the calibrated regions.
EXACT_REACH_COLOR = "#555555"


def _train_flow_minimal(
    y_train_centered: torch.Tensor,
    epochs: int,
    *,
    hidden: int = 64,
    n_layers: int = 4,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 0,
) -> FlowODE:
    """Minimal flow trainer (uniform-time CFM, Gaussian noise prior).

    Avoids the heavyweight Sinkhorn coupling in the production trainer
    so the figure-generating script stays under ~30 s on CPU.
    """
    torch.manual_seed(seed)
    dim = y_train_centered.shape[1]
    vf = VelocityField(dim=dim, hidden=hidden, n_layers=n_layers)
    optimizer = torch.optim.Adam(vf.parameters(), lr=lr)
    dataset = torch.utils.data.TensorDataset(y_train_centered)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        for (x1_batch,) in loader:
            x0_batch = torch.randn_like(x1_batch)
            t = torch.rand(x1_batch.shape[0])
            x_t = (1 - t.unsqueeze(1)) * x0_batch + t.unsqueeze(1) * x1_batch
            target_v = x1_batch - x0_batch
            pred_v = vf(t, x_t)
            loss = torch.nn.functional.mse_loss(pred_v, target_v)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    vf.eval()
    return FlowODE(vf)


def _compute_exact_polygons(net) -> list[np.ndarray]:
    """Return list of (V, 2) ndarrays — convex hulls of each output Star.

    Uses the same n2v Star-propagation path as
    `flow_matching_training/fig_training_progression.py`. Falls back to
    an empty list if the dependencies aren't importable (the smoke
    run will still produce a figure, just without the black outline).
    """
    try:
        from scipy.spatial import ConvexHull  # type: ignore

        from n2v.nn import NeuralNetwork
        from n2v.sets.box import Box
    except Exception as e:  # pragma: no cover
        print(f"  [warn] exact reach unavailable ({e!r}); skipping outline")
        return []

    box = Box(
        np.zeros((2, 1), dtype=float),
        np.ones((2, 1), dtype=float),
    )
    wrapper = NeuralNetwork(net.net)
    stars = wrapper.reach(box.to_star(), method="exact")

    polys = []
    rng = np.random.default_rng(0)
    samples_per_star = 500
    for star in stars:
        V = star.V
        offset = V[:, 0]
        basis = V[:, 1:]
        nVar = star.nVar
        plb = star.predicate_lb.flatten()
        pub = star.predicate_ub.flatten()

        accepted = []
        n_try = 0
        while sum(len(a) for a in accepted) < samples_per_star and n_try < 50:
            n_try += 1
            alpha = rng.uniform(plb, pub, size=(samples_per_star * 4, nVar))
            if star.C is not None and star.C.size > 0:
                mask = (star.C @ alpha.T <= star.d + 1e-9).all(axis=0)
                alpha = alpha[mask]
            if len(alpha) == 0:
                continue
            accepted.append(alpha)
        if not accepted:
            continue
        alpha_all = np.vstack(accepted)
        y_samples = offset[None, :] + alpha_all @ basis.T
        if len(y_samples) < 3:
            continue
        if np.ptp(y_samples, axis=0).min() < 1e-6:
            continue
        try:
            hull = ConvexHull(y_samples)
        except Exception:
            continue
        polys.append(y_samples[hull.vertices])
    return polys


def _calibrated_threshold(score_fn, y_calib: torch.Tensor, alpha: float) -> float:
    """Conformal threshold for a score on y_calib at level alpha."""
    n = y_calib.shape[0]
    import math
    ell = max(1, min(n, int(math.ceil((n + 1) * (1 - alpha)))))
    with torch.no_grad():
        scores = score_fn(y_calib)
    return calibrate(scores, ell).item()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, default_output="fig7_banana_score_geometries.png")
    parser.add_argument("--epochs", type=int, default=2000,
                        help="Flow training epochs (default 2000; "
                             "use --epochs 20 for fastest smoke). At "
                             "2000 epochs the flow density is visibly "
                             "tight against the exact reach polygons; "
                             "fewer epochs leave the flow set looser "
                             "than the GMM set.")
    parser.add_argument("--n-train", type=int, default=4000,
                        help="Flow training-set size (default 4000).")
    parser.add_argument("--n-calib", type=int, default=4000,
                        help="Calibration-set size (default 4000); the "
                             "same conformal calibration set is used "
                             "for all four score families so the only "
                             "difference between methods is the score "
                             "geometry itself.")
    parser.add_argument("--alpha", type=float, default=0.10,
                        help="Miscoverage level for conformal calibration "
                             "(default 0.10 → 90% coverage). Higher than "
                             "the production setting (α=0.001) so the "
                             "geometric differences between score "
                             "families are visible without all four sets "
                             "ballooning out to extreme outer quantiles.")
    parser.add_argument("--grid-res", type=int, default=300,
                        help="Grid resolution per axis for sublevel-set "
                             "rendering (default 300).")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    output = args.output or (Path(__file__).resolve().parent / "fig7_banana_score_geometries.png")

    apply_paper_style()
    plt.rcParams["text.usetex"] = False

    print("[fig7] training RotatedBananaNet...")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # Quick training: the network's __init__ already runs ~2k Adam steps.
    net = RotatedBananaNet(n_train_steps=2000).eval()

    # Sample y_train and y_calib from the network's output distribution.
    print("[fig7] sampling y_train / y_calib...")
    with torch.no_grad():
        x_train = torch.rand(args.n_train, 2)
        x_calib = torch.rand(args.n_calib, 2)
        y_train = net(x_train)
        y_calib = net(x_calib)
    center = y_train.mean(dim=0)

    # Train flow on centered y_train (so the prior at 0 matches data center).
    print(f"[fig7] training flow ({args.epochs} epochs)...")
    t0 = time.time()
    flow = _train_flow_minimal(
        y_train - center, epochs=args.epochs, seed=args.seed,
    )
    print(f"[fig7]   trained flow in {time.time() - t0:.1f} s")

    # Compute exact reach polygons (black outline).
    print("[fig7] computing exact reach via n2v Star propagation...")
    polys = _compute_exact_polygons(net)
    print(f"[fig7]   got {len(polys)} polygons")

    # Bounding box for grid rendering: union of {y_train, y_calib} extents
    # plus a generous pad so each method's sublevel set has room.
    all_y = torch.cat([y_train, y_calib], dim=0).cpu().numpy()
    lo = all_y.min(axis=0) - 0.5
    hi = all_y.max(axis=0) + 0.5
    xs = np.linspace(lo[0], hi[0], args.grid_res)
    ys = np.linspace(lo[1], hi[1], args.grid_res)
    X, Y = np.meshgrid(xs, ys)
    grid_pts = torch.tensor(
        np.column_stack([X.ravel(), Y.ravel()]), dtype=torch.float32,
    )

    # Build the four scores.
    print("[fig7] calibrating four score families...")
    scales = (y_calib.std(dim=0).clamp(min=1e-8))
    hyperrect = HyperrectScore(center=y_calib.mean(dim=0), scales=scales)

    cov = torch.cov(y_calib.T)
    cov_inv = torch.linalg.inv(cov + 1e-6 * torch.eye(cov.shape[0]))
    ellipsoid = EllipsoidScore(center=y_calib.mean(dim=0), cov_inv=cov_inv)

    # GMM k=3 is appropriate for a 2D banana; k=10 over-parameterizes the
    # density and produces an artificially tight (but unfaithful) set in
    # this regime — see the figure's caption for discussion.
    try:
        gmm = GMMScore.fit(y_calib, n_components=3)
    except Exception as e:
        print(f"[fig7]   GMM fit failed ({e!r}); falling back to k=2")
        gmm = GMMScore.fit(y_calib, n_components=2)

    # FlowScore wraps the trained FlowODE; we apply the centering manually
    # here to match the y-train centering used during training.
    class _CenteredFlowScore:
        def __init__(self, fs, c):
            self.fs = fs
            self.c = c

        def __call__(self, y):
            return self.fs(y - self.c)

    base_flow_score = FlowScore(flow, t=1.0, n_steps=30, method="rk4")
    flow_score = _CenteredFlowScore(base_flow_score, center)

    scores = {
        "hyperrect": hyperrect,
        "ellipsoid": ellipsoid,
        "gmm":       gmm,
        "flow":      flow_score,
    }

    thresholds = {
        name: _calibrated_threshold(s, y_calib, args.alpha)
        for name, s in scores.items()
    }
    print(f"[fig7]   thresholds: {thresholds}")

    # Evaluate each score on the grid.
    print("[fig7] rendering sublevel sets on grid...")
    score_fields = {}
    for name, s in scores.items():
        with torch.no_grad():
            vals = s(grid_pts)
        if isinstance(vals, torch.Tensor):
            vals = vals.cpu().numpy()
        score_fields[name] = vals.reshape(args.grid_res, args.grid_res)

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(7, 6))

    # Draw filled regions in inside-out order (largest = lowest zorder)
    # so smaller sets remain visible on top. Per-method alpha lets the
    # outer (loose) sets fade to the background while the tight sets
    # (flow / GMM) read as the figure's focus.
    draw_order = sorted(
        score_fields.keys(),
        key=lambda nm: METHOD_PLOT[nm]["zorder"],
    )
    for name in draw_order:
        field = score_fields[name]
        meta = METHOD_PLOT[name]
        threshold = thresholds[name]
        ax.contourf(
            X, Y, field,
            levels=[-1e30, threshold],
            colors=[meta["color"]],
            alpha=meta["alpha"],
            zorder=meta["zorder"],
        )
        ax.contour(
            X, Y, field,
            levels=[threshold],
            colors=[meta["color"]],
            linewidths=1.4,
            alpha=min(1.0, meta["alpha"] + 0.35),
            zorder=meta["zorder"] + 0.5,
        )

    # Exact reach set as a softened-grey outline (each polygon's hull).
    # Drawn on top so it remains the figure's reference contour, but the
    # colour is dimmer than pure black so the calibrated regions read
    # first.
    exact_z = max(m["zorder"] for m in METHOD_PLOT.values()) + 2
    for verts in polys:
        v_closed = np.vstack([verts, verts[:1]])
        ax.plot(v_closed[:, 0], v_closed[:, 1],
                color=EXACT_REACH_COLOR, linewidth=1.4, zorder=exact_z)

    # Lighter calibration scatter — kept faint so it doesn't compete
    # with the filled regions.
    ax.scatter(
        all_y[::25, 0], all_y[::25, 1],
        s=2, color="black", alpha=0.10, zorder=exact_z - 1,
    )

    # Single legend covering all four calibrated regions and the exact
    # reach outline. Solid filled patches read more cleanly than the
    # translucent on-axes fills, so we build dedicated legend handles
    # instead of relying on the contourf artists.
    from matplotlib.lines import Line2D  # noqa: E402
    from matplotlib.patches import Patch  # noqa: E402
    legend_handles = []
    for name in ("hyperrect", "ellipsoid", "gmm", "flow"):
        meta = METHOD_PLOT[name]
        legend_handles.append(
            Patch(facecolor=meta["color"],
                  edgecolor=meta["color"],
                  alpha=min(1.0, meta["alpha"] + 0.25),
                  label=meta["label"]),
        )
    if polys:
        legend_handles.append(
            Line2D([0], [0], color=EXACT_REACH_COLOR, linewidth=1.4,
                   label="Exact reach set"),
        )
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8,
              framealpha=0.9)

    ax.set_xlabel("$y_1$", fontsize=9)
    ax.set_ylabel("$y_2$", fontsize=9)
    ax.set_title(
        f"Score-geometry comparison on RotatedBananaNet "
        f"($\\alpha$={args.alpha}, flow epochs={args.epochs})",
        fontsize=10,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])

    fig.tight_layout()
    save_figure(fig, output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
