# Legacy per-benchmark runners

These were the original "one script per benchmark" runners from before
the convention switched to **one script per (experiment, tool), with
`--benchmark` CLI**. They are kept here (rather than deleted) because:

1. Some still produce data the new runners don't yet (e.g. volume
   comparison in Exp 3) — to be ported, then these can be removed.
2. They preserve the historical record of what was run during the
   lock-probe phase.

**Do not extend.** Add new functionality to the per-(experiment, tool)
runners under `exp<N>_<dir>/exp<N>_run_<tool>.py`.

## Inventory

* `exp1_run_<benchmark>.py` (7 files) — replaced by
  `experiments/exp1_vnncomp_subset/exp1_run_ours.py --benchmark <name>`.
* `exp2_run_vit.py`, `exp2_run_yolo.py` — replaced by
  `experiments/exp2_prob_scale/exp2_run_ours.py --benchmark <name>`.

## Still-active in their original location

The following Exp 2 / Exp 3 modules look "legacy" in name but are
imported as **libraries** by other scripts and were intentionally left
in place:

* `experiments/exp2_prob_scale/exp2_run_cifar10_resnet110.py` —
  exposes `_load_pretrained`, `_make_loader` used by
  `baselines/_common.py:_load_cifar10_resnet110` and
  `exp2_soundness_audit.py`.
* `experiments/exp2_prob_scale/exp2_run_vit_small_cifar10.py` —
  same pattern.
* `experiments/exp3_synthetic/exp3_run_synthetic.py` etc. —
  imported by `experiments/exp_ablation/ablation_run_score.py`; the
  Exp 3 volume-MC logic still lives here pending port to the new runner.
