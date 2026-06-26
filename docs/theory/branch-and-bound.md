# Branch-and-bound verification in n2v (manual, toolbox-general)

This documents the manually-implemented bounding (IBP) and search (branch-and-bound)
layers in n2v — no external verifier (α,β-CROWN etc.) is imported — and what they
can and cannot do on the VNN-COMP 2023 ViT benchmark.

## 1. The two layers

**Bounding (incomplete, sound).** n2v's reach engine *is* the bounder, at three
precision tiers, all sound:
- **Box reach = IBP** (interval bound propagation): `set_type=Box`. Pure interval
  arithmetic through every layer; fastest, loosest.
- **Star / Zono reach**: keep affine (predicate/generator) correlation; tighter.
  `relu_star_approx` is a triangle relaxation; the ViT attention uses the sound
  `bilinear_matmul` + `softmax_attention` ops.
A spec is decided by `verify_specification(reach_sets, spec)` → `UNSAT` (safe) /
`UNKNOWN`. This is incomplete: a true-but-untight property returns UNKNOWN.

**Search (complete-in-the-limit, sound).** Three manual mechanisms, all in
`n2v/nn/bab.py` / `relu_reach.py` (no external verifier):
- **Exact ReLU splitting** — `NeuralNetwork.reach(..., method='exact')` splits
  *every* unstable ReLU via `relu_star_exact`, returning the exact union of
  linear pieces. Complete; verifies relaxation gaps `approx` cannot (approx
  UNKNOWN → exact UNSAT, 27 stars). Explodes (2^#unstable) — no pruning.
- **Controlled neuron-split BaB** — `verify_bab_relu` (nn.Sequential of
  Linear/Flatten/ReLU): branches *one unstable ReLU at a time* (forced
  active/inactive via halfspace intersection + LP `precomputed_bounds`, so the
  triangle ReLU treats a forced neuron exactly), **prunes** safe/infeasible
  subdomains, and falsifies the fixed input box once. The right split space for
  ReLU classifiers; prunes far below 2^#unstable.
- **Input-domain BaB** — `verify_bab` / `verify_bab_model`: recursively bisect
  the **input box**, bound (reach + `verify_specification`), **falsify**
  (`n2v.utils.falsify`, manual random/PGD/APGD), prune safe, branch rest
  (sensitivity / widest-edge). Fully layer-agnostic (only touches the input box).

All are sound by construction: prune only on UNSAT (or infeasible split);
FALSIFIED only with a concrete counterexample; VERIFIED only when every leaf of a
*covering* split is safe; UNKNOWN on budget — never an over-claim. Validated on a
relaxation-gap toy (single-shot UNKNOWN → BaB VERIFIED), with falsification and
budget-soundness (`tests/unit/test_bab.py`, 8 tests).

## 2. Applying BaB to the ViT benchmark — result

Bounder = `ViTReacher` symbolic-av (CROWN-class value-path); falsifier = `falsify`
on the torch model; branching = input-domain bisection, sensitivity from the reach
output's input-predicate generators.

It runs and is sound, but **input-domain splitting does not scale to the ViT
benchmark**:

| ε (×/255) | single-shot symbolic-av | BaB (input-split) |
|---|---|---|
| 0.50 | VERIFIED (margin +0.084) | VERIFIED, 1 node (no split needed) |
| 0.55 | VERIFIED (margin +0.036) | VERIFIED, 1 node |
| 0.60 | UNKNOWN | UNKNOWN — 150 nodes, **depth 80**, a leaf reached +0.027 but the full covering tree did not close |
| 1.00 (target) | UNKNOWN (margin ≈ −0.2…−0.55) | far out of reach |

The single-shot certified radius for ibp_3_3_8 inst0 is ≈ 0.57/255. Just *above*
it, BaB needs an ~80-deep split path to make one corner safe — and an input box of
**3072 dimensions** has exponentially many such corners. So BaB extends the
verified radius only negligibly: the curse of dimensionality. This is exactly why
image-classifier verifiers split **neurons**, not inputs.

But ReLU-neuron splitting does not help here either: the ViT looseness at full ε
is dominated by the **softmax** attention-weight relaxation (box-lifted A), not
the FF ReLUs. And the fully-symbolic star alternative (McCormick QKᵀ + symbolic
softmax) blows up the predicate count (QKᵀ alone is ~14k product predicates for
ibp's 17 tokens × 3 heads × 16 dims), making the LP intractable — the reason the
field uses CROWN-style *linear bound propagation* (no per-term predicates, no LP)
rather than star+LP for transformers.

## 3. Honest conclusion

- **Can we verify the ViT benchmark with sound star+LP + BaB? Not at full ε.**
  Input-split BaB can't beat the input dimensionality; ReLU-split can't touch the
  softmax bottleneck; full symbolic star+LP doesn't scale (predicate/LP blow-up).
  This matches the literature: α,β-CROWN's 79/200 comes from **general-nonlinearity
  branch-and-bound on top of CROWN linear bounds** (Shi et al. 2023, the very
  paper behind this benchmark), not from interval/star bounding or input splitting.
- **What is delivered, sound and manual:** IBP (Box reach), Star/Zono reach, the
  symbolic-av CROWN-class precision mode, complete ReLU-split (exact reach), and a
  general input-domain BaB engine — all toolbox-wide, no external verifier.
- **The path to the 79/200 sound reference** (scoped, not built): replace star+LP
  with CROWN-style backward linear bounds (O(input-dim) symbolic, no LP), add a
  branching rule over the **softmax pre-activations / attention nonlinearity**
  (the α,β-CROWN general-nonlinearity BaB), and bound-guide + prune. The BaB
  search scaffold here (queue, prune-on-UNSAT, falsify, covering-split soundness)
  is reusable; the missing piece is the per-neuron/per-nonlinearity split applied
  to a CROWN-class bounder.
