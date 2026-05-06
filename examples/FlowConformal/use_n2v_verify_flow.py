"""Demo: n2v.probabilistic.verify_flow library API (Plan B entry point).

Shows the modern flow-conformal+AMLS pipeline imported as a top-level
n2v library function, and the opt-in falsifier knob.

By default ``verify_flow`` runs flow-conformal+AMLS only (Stage 1
falsifier disabled). Pass ``use_falsifier=True`` to enable a Stage 1
falsifier that can return SAT verdicts by finding a real counterexample.

Run:

    python -m examples.FlowConformal.use_n2v_verify_flow
"""
from __future__ import annotations

import numpy as np
import torch

from examples.FlowConformal.networks import RotatedBananaNet
from n2v.probabilistic import verify_flow
from n2v.sets.halfspace import HalfSpace


def main() -> None:
    torch.manual_seed(0)
    net = RotatedBananaNet().eval()
    # Unreachable unsafe region: y_0 <= -100 is far below the banana's
    # support, so the run should certify UNSAT regardless of the
    # falsifier setting.
    spec = HalfSpace(np.array([[1.0, 0.0]]), np.array([[-100.0]]))

    common = dict(
        network=net,
        input_lb=np.array([0.0, 0.0]),
        input_ub=np.array([1.0, 1.0]),
        spec=spec,
        alpha=0.001, m=2000, ell=1999,
        scenario_n_samples=2000, scenario_beta=0.001,
        n_train=5000, flow_epochs=2000, flow_config='base',
        verification_method='amls',
        seed=0,
    )

    print('=== verify_flow (default: use_falsifier=False) ===')
    r1 = verify_flow(**common)
    print(f"verdict={r1['verdict']}, q={r1.get('q')}, "
          f"epsilon_total={r1.get('epsilon_total')}")

    print('=== verify_flow (use_falsifier=True; opt-in Stage 1) ===')
    r2 = verify_flow(use_falsifier=True, **common)
    print(f"verdict={r2['verdict']}")


if __name__ == '__main__':
    main()
