"""Demo: render the cached ThreeBlobClassifier3D Star union + forward samples.

Produces two HTML figures:
* ``three_blob_3d_star_union.html`` — one translucent parallelepiped per
  Star. Faithful but visually noisy.
* ``three_blob_3d_convex_hull.html`` — a single convex-hull mesh over all
  Star vertices. Clean single surface; over-approximation of the true
  reach set, tight when the reach set is convex.

Prints the hull volume alongside the cached MC volume (~213.7) so the
caller can see how tight the convex over-approximation is.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))

from examples.FlowConformal.networks import ThreeBlobClassifier3D  # noqa: E402
from examples.FlowConformal.utils import compute_exact_reach  # noqa: E402
from n2v.probabilistic.flow.sampling import sample_l_inf_ball  # noqa: E402
from n2v.probabilistic.flow.star_viz import (  # noqa: E402
    render_star_convex_hull_3d,
    render_star_union_3d,
    render_star_union_isosurface_3d,
)


def main():
    torch.manual_seed(0)
    net = ThreeBlobClassifier3D()
    net.eval()
    x_center = torch.zeros(3)
    radius = 1.0

    # Star propagation via n2v sound reachability.
    reach = compute_exact_reach(net, x_center.numpy(), radius, output_dim=3)
    stars = reach['stars']

    # Forward-sample the input ball and push through the network.
    x = sample_l_inf_ball(
        x_center=x_center, radius=radius, n_samples=2000, seed=0, dim=3,
    )
    with torch.no_grad():
        y = net(x).numpy()

    out_dir = Path(__file__).parent / 'figures'
    out_dir.mkdir(exist_ok=True)

    # Convex-hull view
    out_hull = out_dir / 'three_blob_3d_convex_hull.html'
    _, hull_vol = render_star_convex_hull_3d(
        stars, forward_samples=y,
        title='ThreeBlobClassifier3D: convex hull of Star union + forward samples',
        out_html=out_hull,
    )
    print(f"Saved {out_hull}")

    # Quantify how tight the hull over-approximation is.
    cached_vol = 213.72  # from _exact_volume_cache (MC on Star union).
    print(f"\nVolume comparison:")
    print(f"  convex hull volume      = {hull_vol:.2f}")
    print(f"  exact Star-union MC vol = {cached_vol:.2f} (cached)")
    ratio = hull_vol / cached_vol if cached_vol > 0 else float('nan')
    print(f"  hull / exact ratio      = {ratio:.3f}")
    if ratio < 1.1:
        print("  -> hull is tight; reach set is near-convex")
    elif ratio < 1.5:
        print("  -> hull is moderately tight; reach set has some non-convexity")
    else:
        print("  -> hull is loose; reach set is substantially non-convex")

    # Isosurface view: a single mesh that preserves non-convex structure.
    # Resolution 64 gives ~2-3 MB HTML (fast to load) with minimal loss of
    # detail on a smooth reach set. Bump to 96 if finer geometry matters.
    out_iso = out_dir / 'three_blob_3d_isosurface.html'
    print(f"\nRendering isosurface at resolution 64 (may take ~20s) ...")
    render_star_union_isosurface_3d(
        stars, forward_samples=y,
        title='ThreeBlobClassifier3D: Star union isosurface + forward samples',
        out_html=out_iso, resolution=64,
    )
    print(f"Saved {out_iso}")


if __name__ == '__main__':
    main()
