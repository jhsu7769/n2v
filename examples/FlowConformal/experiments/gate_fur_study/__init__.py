"""Gate-only false-UNSAT-rate (FUR) stress-test study.

Runs ours and Hashemi-clipping on every SAT-ground-truth instance from
Exp 1 + Exp 2 *with the APGD pre-step disabled*, so that the verdict
gate is exposed directly to every SAT instance instead of being
preempted by the falsifier on most of them.

This isolates the **intrinsic gate FUR** from the deployed-system FUR.
The deployed numbers (with APGD on) stay in Phase 1+2 outputs; the
gate-only numbers from this study are reported as a stress test.

The other probabilistic baselines (ProbStar / SaVer / RS) already run
without an external APGD step, so their existing Phase 1+2 SAT-only
slices are already gate-only and are aggregated alongside the new
ours/Hashemi runs.
"""
