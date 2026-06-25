"""Soundness of the bilinear (set @ set) matmul reach op.

The reach op must produce a true over-approximation: every concrete
``scale * (left @ right)`` reachable from the operand boxes must be contained in
the output set. We sample the two operands *independently* (the worst case for
the concretise mode, which drops operand cross-correlation) across all sign
regimes, batched-head shapes mirroring attention's QK^T and A.V, and a negative
scale. Star containment is checked with the authoritative LP membership.
"""

import numpy as np
import pytest

from n2v.sets import Star, Zono, Box
from n2v.nn.layer_ops.bilinear_matmul_reach import (
    bilinear_matmul_box,
    bilinear_matmul_zono,
    bilinear_matmul_star,
)

SEED = 20260625
N_SAMPLES = 300
TOL = 1e-7

# (left_shape, right_shape): standard batched matmul, out = left @ right.
# QK^T style: [heads, L, d] @ [heads, d, L];  A.V style: [heads, L, L] @ [heads, L, d].
SHAPES = [
    ((3, 5, 16), (3, 16, 5)),
    ((3, 5, 5), (3, 5, 16)),
    ((2, 2), (2, 4)),          # no leading head axis
]
REGIMES = ["pos", "neg", "mixed"]
SCALES = [0.25, 1.0, -0.25]


def _interval(rng, shape, regime):
    """Random [lo, hi] box of given shape in the requested sign regime."""
    if regime == "pos":
        lo = rng.uniform(0.1, 1.0, size=shape)
        hi = lo + rng.uniform(0.0, 1.0, size=shape)
    elif regime == "neg":
        hi = rng.uniform(-1.0, -0.1, size=shape)
        lo = hi - rng.uniform(0.0, 1.0, size=shape)
    else:  # mixed -- straddles 0
        lo = rng.uniform(-1.0, -0.05, size=shape)
        hi = rng.uniform(0.05, 1.0, size=shape)
    return lo.astype(np.float64), hi.astype(np.float64)


def _out_bounds(out_set):
    if isinstance(out_set, Box):
        return out_set.lb.reshape(-1), out_set.ub.reshape(-1)
    if isinstance(out_set, Zono):
        lb, ub = out_set.get_bounds()
        return np.asarray(lb).reshape(-1), np.asarray(ub).reshape(-1)
    lb, ub = out_set.get_ranges()
    return np.asarray(lb).reshape(-1), np.asarray(ub).reshape(-1)


def _cases():
    for lshape, rshape in SHAPES:
        for regime in REGIMES:
            for scale in SCALES:
                yield lshape, rshape, regime, scale


@pytest.mark.parametrize("lshape,rshape,regime,scale", list(_cases()))
def test_bilinear_matmul_box_sound(lshape, rshape, regime, scale):
    rng = np.random.default_rng(SEED)
    al, au = _interval(rng, lshape, regime)
    bl, bu = _interval(rng, rshape, regime)

    out = bilinear_matmul_box(
        [Box(al.reshape(-1), au.reshape(-1))],
        [Box(bl.reshape(-1), bu.reshape(-1))],
        lshape, rshape, scale=scale,
    )[0]
    lo, hi = _out_bounds(out)

    for _ in range(N_SAMPLES):
        xl = rng.uniform(al, au)
        xr = rng.uniform(bl, bu)
        true = (scale * (xl @ xr)).reshape(-1)
        assert np.all(true >= lo - TOL) and np.all(true <= hi + TOL)


@pytest.mark.parametrize("lshape,rshape,regime,scale", list(_cases()))
def test_bilinear_matmul_zono_sound(lshape, rshape, regime, scale):
    rng = np.random.default_rng(SEED + 1)
    al, au = _interval(rng, lshape, regime)
    bl, bu = _interval(rng, rshape, regime)

    out = bilinear_matmul_zono(
        [Zono.from_bounds(al.reshape(-1), au.reshape(-1))],
        [Zono.from_bounds(bl.reshape(-1), bu.reshape(-1))],
        lshape, rshape, scale=scale,
    )[0]
    lo, hi = _out_bounds(out)

    for _ in range(N_SAMPLES):
        xl = rng.uniform(al, au)
        xr = rng.uniform(bl, bu)
        true = (scale * (xl @ xr)).reshape(-1)
        assert np.all(true >= lo - TOL) and np.all(true <= hi + TOL)


@pytest.mark.parametrize("lshape,rshape,regime,scale", list(_cases()))
def test_bilinear_matmul_star_sound(lshape, rshape, regime, scale):
    rng = np.random.default_rng(SEED + 2)
    al, au = _interval(rng, lshape, regime)
    bl, bu = _interval(rng, rshape, regime)

    out = bilinear_matmul_star(
        [Star.from_bounds(al.reshape(-1), au.reshape(-1))],
        [Star.from_bounds(bl.reshape(-1), bu.reshape(-1))],
        lshape, rshape, scale=scale,
    )[0]

    for _ in range(N_SAMPLES):
        xl = rng.uniform(al, au)
        xr = rng.uniform(bl, bu)
        true = (scale * (xl @ xr)).reshape(-1)
        # Authoritative LP membership in the output star.
        assert out.contains(true, method="lp")


def test_mccormick_mode_fails_loud():
    s = Star.from_bounds(np.zeros(4), np.ones(4))
    with pytest.raises(NotImplementedError):
        bilinear_matmul_star([s], [s], (2, 2), (2, 2), mode="mccormick")


def test_operand_length_mismatch_raises():
    s = Star.from_bounds(np.zeros(4), np.ones(4))
    with pytest.raises(ValueError):
        bilinear_matmul_star([s, s], [s], (2, 2), (2, 2))
