# n2v — VNN-COMP 2026 Benchmark Support

_Status snapshot: 2026-06-10._

This document records which VNN-COMP 2026 benchmarks n2v can handle, and
**documents the layer and specification types that are not supported**.

## How this was measured

- **Static audit:** every benchmark's ONNX was loaded (onnx2torch + fx
  trace) and its op set inspected; the representative VNNLIB was parsed.
- **Runtime smoke test:** one instance per benchmark was run end-to-end
  through the competition runner (`vnncomp_runner.py`). The `2.0` version
  was used where present (the new VNNLIB 2.0 format), otherwise `1.0`.
  Reach-heavy benchmarks were re-run with full CPU resources (a single
  smoke pass under heavy concurrency starves them and inflates timeouts).
- A smoke result is a **one-instance snapshot**, not a full-benchmark
  score. Timeouts marked below are at the smoke budget (100–150 s) and
  mostly indicate a benchmark that needs per-benchmark configuration
  tuning, not a hard limitation.

**Legend.** `(P)` = result produced by the **probabilistic** method
(conformal / flow) — a coverage guarantee, **not** a sound proof. `✗` in
the spec column = the spec type is rejected by the parser (reported
`unknown`). SAT results all come from falsification (random + PGD), which
needs only forward passes and therefore works even when a layer is
unsupported for reachability.

## Sound reachability support (the support question)

"Support" here means: load the ONNX model, parse the VNNLIB spec, and run
**sound** set-based reachability (`approx` Star) through every layer to
completion. This is stricter than "solved" — it excludes SAT found by
falsification and UNSAT from the probabilistic method (a coverage
guarantee, not a sound proof).

**Supported: 14 / 34.** Every unsupported case fails as an **error →
`unknown`**, never as a silent wrong verdict, so there is no soundness
*violation* (we never emit a false `unsat` from sound reach).

- **Supported (14):** acasxu_2023, challenging_certified_training_2026
  _(slow)_, collins_rul_cnn_2022, cora_2024, dist_shift_2023,
  linearizenn_2024, malbeware, metaroom_2023, relusplitter,
  relusplitter_2026, safenlp_2024, sat_relu, test, tllverifybench_2023.

**Not supported (20), by cause:**

- **Spec type rejected by the parser (4):**
  `adaptive_cruise_control_non_linear_2026` (nonlinear constraints),
  `monotonic_acasxu_2026` + `isomorphic_acasxu_2026` (two-network
  relational), `smart_turn_multimodal_2026` (multi-input + quantized).
- **ONNX op not implemented in reach (9):** `ConvTranspose2d` —
  `cgan_2023`, `cgan2026`; `Gather` — `nn4sys`, `lsnc_relu`,
  `ml4acopf_2024`; `Softmax`/transformer ops — `vit_2023`,
  `traffic_signs_recognition_2023`; YOLO control-flow
  (`Where`/`ScatterND`/`ArgMax`/…) — `cctsdb_yolo_2023`; `Pow` —
  `collins_aerospace_benchmark`.
- **Bug on otherwise-supported layers (fixable, not a missing feature)
  (6):** residual `Add` of Stars with mismatched predicate counts —
  `cersyve` (FC!), `cifar100_2024`, `tinyimagenet_2024`, `yolo_2023`;
  flat→image `Reshape` not producing an `ImageStar` for an internal
  `Conv2D` — `soundnessbench`, `soundnessbench_2026`.
- **Data missing (1):** `vggnet16_2022` — ONNX fetched by `setup.sh`.

## Solve snapshot (1 instance / benchmark)

Separately, the full runner (falsification + reach + probabilistic
fallback) produced, on one instance each:
`sat` 8 · `unsat` 10 · `unknown` 9 · `timeout` 6 · `no model file` 1 →
**18 / 34 "solved"** — but note SAT comes from falsification and several
UNSAT are probabilistic `(P)`, so this overstates *sound* support (14).

## Per-benchmark table

| Benchmark | 2.0 spec | Unsupported ops | Smoke result | Time | Status |
|---|---|---|---|---|---|
| acasxu_2023 | 2.0 ✓ | — | timeout | 173s | timeout at smoke budget — needs per-benchmark tuning |
| adaptive_cruise_control_non_linear_2026 | nonlinear ✗ | — | unknown | 5s | spec has nonlinear constraints — unsupported |
| cctsdb_yolo_2023 | 2.0 ✓ | (YOLO control-flow) | unknown | 7s | reach blocked by Gather/Where/ScatterND/ArgMax/Range/… |
| cersyve | 2.0 ✓ | — | sat | 8s | SAT — counterexample (falsification) |
| cgan2026 | 2.0 ✓ | ConvTranspose2d | sat | 5s | SAT — counterexample (falsification) |
| cgan_2023 | 2.0 ✓ | ConvTranspose2d | sat | 5s | SAT — counterexample (falsification) |
| challenging_certified_training_2026 | 2.0 ✓ | — | timeout | 159s | timeout at smoke budget — needs per-benchmark tuning |
| cifar100_2024 | 2.0 ✓ | — | unsat (P) | 10s | UNSAT — probabilistic (coverage, not sound) |
| collins_aerospace_benchmark | 2.0 ✓ | OnnxPow | sat | 57s | SAT — counterexample (falsification) |
| collins_rul_cnn_2022 | 2.0 ✓ | — | sat | 5s | SAT — counterexample (falsification) |
| cora_2024 | 2.0 ✓ | — | timeout | 155s | timeout at smoke budget — needs per-benchmark tuning |
| dist_shift_2023 | 2.0 ✓ | — | unsat | 41s | UNSAT — sound |
| isomorphic_acasxu_2026 | 2-network ✗ | — | unknown | 6s | two-network relational spec — unsupported |
| linearizenn_2024 | 2.0 ✓ | — | sat | 28s | SAT — counterexample (falsification) |
| lsnc_relu | 2.0 ✓ | OnnxGather | unknown | 107s | reach blocked by OnnxGather; inconclusive |
| malbeware | 2.0 ✓ | — | unsat | 55s | UNSAT — sound |
| metaroom_2023 | 2.0 ✓ | — | unsat | 21s | UNSAT — sound |
| ml4acopf_2024 | 2.0 ✓ | OnnxGather, OnnxRound, OnnxUnsqueezeStaticAxes | unsat (P) | 43s | UNSAT — probabilistic (coverage, not sound) |
| monotonic_acasxu_2026 | 2-network ✗ | — | unknown | 11s | two-network relational spec — unsupported |
| nn4sys | 1.0 only ✓ | OnnxGather | unknown | 55s | reach blocked by OnnxGather; inconclusive |
| relusplitter | 2.0 ✓ | — | unknown | 38s | inconclusive — needs tuning |
| relusplitter_2026 | 2.0 ✓ | — | unsat | 39s | UNSAT — sound |
| safenlp_2024 | 2.0 ✓ | — | sat | 9s | SAT — counterexample (falsification) |
| sat_relu | 2.0 ✓ | — | sat | 9s | SAT — counterexample (falsification) |
| smart_turn_multimodal_2026 | multi-input ✗ | DequantizeLinear | unknown | 12s | multi-input + quantized model — unsupported |
| soundnessbench | 2.0 ✓ | — | timeout | 155s | timeout at smoke budget — needs per-benchmark tuning |
| soundnessbench_2026 | 2.0 ✓ | — | timeout | 154s | timeout at smoke budget — needs per-benchmark tuning |
| test | 2.0 ✓ | — | unsat | 9s | UNSAT — sound |
| tinyimagenet_2024 | 2.0 ✓ | — | timeout | 158s | timeout at smoke budget — needs per-benchmark tuning |
| tllverifybench_2023 | 2.0 ✓ | — | unknown | 10s | inconclusive — needs tuning |
| traffic_signs_recognition_2023 | 2.0 ✓ | Softmax | unsat (P) | 46s | UNSAT — probabilistic (coverage, not sound) |
| vggnet16_2022 | 2.0 ✓ | — | no model file | 8s | ONNX absent — run setup.sh (large-model download) |
| vit_2023 | 2.0 ✓ | OnnxBatchNormGeneric, OnnxConstantOfShape, OnnxGather, OnnxShape, OnnxSoftmaxV1V11, OnnxUnsqueezeStaticAxes | unsat (P) | 106s | UNSAT — probabilistic (coverage, not sound) |
| yolo_2023 | 2.0 ✓ | OnnxPadStatic | unsat (P) | 58s | UNSAT — probabilistic (coverage, not sound) |

## Documented unsupported types

### Specification types (parser rejects → `unknown`)

n2v parses VNNLIB **1.0** and the **2.0** single-network linear fragment.
The following spec types are outside that fragment and are reported
`unknown` (the parser raises a clear error rather than silently
mis-verifying):

- **Nonlinear arithmetic constraints** — `adaptive_cruise_control_non_linear_2026`.
  Specs contain products of variables (e.g. `(* X[0,1] X[0,1])`), `!=`,
  `==`, and deep `or`/`and` nesting. Reachability over linear sets cannot
  express these.
- **Two-network / relational specs** — `monotonic_acasxu_2026`,
  `isomorphic_acasxu_2026`. The VNNLIB declares two networks
  (`equal-to` / `isomorphic-to`) and the `instances.csv` ONNX field is a
  list `[('f', …), ('g', …)]`. Requires joint reachability over both
  networks with coupled inputs — not yet implemented. (The runner detects
  the two-network ONNX field and returns `unknown`.)
- **Multi-input specs** — `smart_turn_multimodal_2026` (two declared
  inputs; also a quantized model — see below).

### Layer / ONNX op types (reach unsupported)

The reach engine (Star/Zono/Box/ImageStar) does not implement these ops.
A benchmark using one of them cannot be **proved** UNSAT by sound reach,
but SAT instances are still found by falsification, and some are reported
via the probabilistic method.

- **ConvTranspose2d** — `cgan_2023`, `cgan2026`. (Both solved: SAT via
  falsification.)
- **Gather** — `nn4sys`, `lsnc_relu`, `ml4acopf_2024`.
- **Pow** — `collins_aerospace_benchmark`. (Solved: SAT via falsification.)
- **DequantizeLinear / quantized models** — `smart_turn_multimodal_2026`
  (onnx2torch cannot convert the quantized graph).
- **YOLO post-processing control-flow** — `cctsdb_yolo_2023`
  (`ArgMax`, `Where`, `ScatterND`, `Range`, `Expand`, `Clip`,
  `ConstantOfShape`, `Shape`, …).
- **Transformer attention ops** — `vit_2023` (`Softmax`, `Shape`,
  `ConstantOfShape`, `Unsqueeze`, `BatchNormGeneric`, …). _Out of scope by
  decision — handled separately by other contributors._
- **Round / PadStatic** — appear in `ml4acopf_2024`, `yolo_2023` (those
  benchmarks are routed through the probabilistic method).

### Performance-limited (supported, but slow on the snapshot)

Solvable in principle (model and spec both supported) but exceeded the
smoke budget — these need per-benchmark configuration tuning:
`acasxu_2023` (wide input box → ReLU split blow-up), `cora_2024`,
`soundnessbench`, `soundnessbench_2026`, `tinyimagenet_2024`,
`challenging_certified_training_2026`. (`relusplitter`, `tllverifybench_2023`
returned `unknown` and likewise need relaxation/strategy tuning.)

### Needs data

- `vggnet16_2022` — the ONNX is not committed to the benchmark repo; it is
  fetched by `setup.sh` (large-model WebDAV download). Not a tool
  limitation.

## Soundness note

Results marked `(P)` are produced by the probabilistic (conformal / flow)
method and carry a coverage guarantee, **not** a sound proof — relevant
for how they should be reported/scored. The choice of sound vs.
probabilistic per benchmark is a tuning decision deferred to a later pass.
