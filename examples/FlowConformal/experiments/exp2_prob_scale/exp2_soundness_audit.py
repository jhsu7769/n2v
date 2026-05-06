"""Exp 2 soundness audit: post-hoc cex search for any UNSAT verdict.

Reads a CSV of verdicts from ``exp2_run_*`` (any method). For each row
matching ``--filter-verdict`` (default UNSAT), runs an aggressive
adversarial attack pipeline directly on the network and flags any
instance where a real counterexample is found — that's a false UNSAT,
i.e. a soundness violation.

The audit is independent of the verification pipeline. It only uses the
benchmark loader to recover ``(network, x_clean, y_clean, eps)`` and
then runs:
  1. AutoAttack standard ('apgd-ce', 'apgd-t', 'fab-t', 'square') if the
     ``autoattack`` package is installed; otherwise a manual APGD-CE
     fallback (~50 lines).
  2. 5K-restart PGD (default) — iterate random-restart PGD up to
     ``--pgd-restarts`` times, each with ``--pgd-steps``, breaking on
     first cex.

Usage:
    cd /home/sasakis/v/tools/n2v
    /home/sasakis/miniconda3/envs/n2v/bin/python -u -m \\
        examples.FlowConformal.experiments.exp2_prob_scale.exp2_soundness_audit \\
        --input-csv outputs/exp2_cifar10_resnet110_ours_sigma0.25.csv \\
        --benchmark cifar10_resnet110 \\
        --output-csv outputs/exp2_cifar10_resnet110_audit.csv

Per-instance time on A30:
  ~5-10 s for AutoAttack alone, ~50-100 s with 5K-restart PGD.
For 100 UNSAT instances per benchmark, ~1-3 hours per benchmark.

Output CSV columns:
    benchmark, instance_name, seed, original_verdict,
    audit_attack ('autoattack' / 'manual_apgd' / 'pgd' / 'none'),
    found_cex (0/1), cex_pred, audit_wall_s, error
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch


_OUT_DIR = Path(__file__).parent / 'outputs'


# ---------------------------------------------------------------------------
# Benchmark loaders
# ---------------------------------------------------------------------------
# Each loader maps an ``instance_name`` from the input CSV to a tuple
# ``(network, x_clean: Tensor[d], y_clean: int, eps: float)`` with x_clean
# the flat (d,) input and y_clean the model's predicted class on x_clean.
# Loader returns ``None`` if the instance can't be reconstructed (skip).


def _classification_loader_cifar10_resnet110(args) -> Callable:
    """CIFAR-10 ResNet-110 RS (Cohen et al.) loader."""
    from examples.FlowConformal.experiments.exp2_prob_scale.exp2_run_cifar10_resnet110 import (
        _load_pretrained, _load_cifar10_test, _EPS,
    )
    sigma = getattr(args, 'sigma', '0.25')
    network = _load_pretrained(sigma)
    if network is None:
        raise SystemExit(
            f'cifar10_resnet110: pretrained weights for sigma={sigma} not '
            f'found; cannot audit.')
    network.eval()

    def _load(instance_name: str):
        # instance_name = 'cifar10_test_<idx>_label_<label>'
        try:
            parts = instance_name.split('_')
            idx = int(parts[2])
        except Exception:
            return None
        # Load test set up to idx+1 (cheap; cached on disk).
        imgs, labels = _load_cifar10_test(idx + 1)
        x = torch.as_tensor(imgs[idx], dtype=torch.float32)
        # Use model's predicted class (matches verification spec).
        with torch.no_grad():
            y = int(network(x.unsqueeze(0)).argmax(dim=1).item())
        return network, x, y, float(_EPS)

    return _load


def _classification_loader_vit_small_cifar10(args) -> Callable:
    """ViT-Small / CIFAR-10 loader."""
    from examples.FlowConformal.experiments.exp2_prob_scale.exp2_run_vit_small_cifar10 import (
        _load_model, _load_cifar10_test, _EPS,
    )
    network, _model_name = _load_model(getattr(args, 'prefer', 'timm'))
    if network is None:
        raise SystemExit('vit_small_cifar10: no backbone available; cannot audit.')
    network.eval()

    def _load(instance_name: str):
        try:
            parts = instance_name.split('_')
            idx = int(parts[2])
        except Exception:
            return None
        imgs, _ = _load_cifar10_test(idx + 1)
        x = torch.as_tensor(imgs[idx], dtype=torch.float32)
        with torch.no_grad():
            y = int(network(x.unsqueeze(0)).argmax(dim=1).item())
        return network, x, y, float(_EPS)

    return _load


def _vnncomp_loader_factory(benchmark_root: Path) -> Callable:
    """VNN-COMP-style loader: instance_name = '<onnx>+<vnnlib>'.

    Returns ``(network, x_clean, y_clean, eps)`` where x_clean is the box
    centre, y_clean is the model's predicted class on x_clean, and eps is
    half the box width (max axis-wise; non-uniform boxes get the L_inf
    radius so PGD/AutoAttack search the inscribed L_inf ball).

    For specs that aren't classification-robustness (e.g. yolo bounding
    box props), this still works as a "find any input in the box that
    flips the predicted class" check; the audit script's claim is "if a
    cex on classification-robustness exists, the original verdict is
    unsound". For non-classification specs we use the spec directly: the
    network is treated as a vector-output map, and a cex is any x in box
    such that ``spec(network(x))`` evaluates to "unsafe" — i.e. the
    `verifier_spec` HalfSpace (or list) is satisfied. We delegate to a
    spec-aware checker.
    """
    from examples.FlowConformal.experiments.exp1_vnncomp_subset._common import (
        load_instance, parse_instances_csv,
    )
    instances_csv = benchmark_root / 'instances.csv'
    rows = parse_instances_csv(instances_csv)
    rel_index = {f'{Path(o).name}+{Path(v).name}': (o, v) for o, v, _ in rows}

    def _load(instance_name: str):
        if instance_name not in rel_index:
            return None
        onnx_rel, vnn_rel = rel_index[instance_name]
        try:
            network, boxes, spec = load_instance(benchmark_root, onnx_rel, vnn_rel)
        except Exception:
            return None
        network.eval()
        # Use the first box (multiple boxes => OR-of-input-regions; we
        # audit each separately by appending '#box_idx' if needed; for now
        # take box 0 — the verification pipeline already returns SAT on
        # the first SAT box).
        lb, ub = boxes[0]
        lb_t = torch.as_tensor(lb, dtype=torch.float32)
        ub_t = torch.as_tensor(ub, dtype=torch.float32)
        x_centre = 0.5 * (lb_t + ub_t)
        eps_vec = 0.5 * (ub_t - lb_t)
        # L_inf radius = min axis-wise half-width (inscribed ball). Using
        # min keeps perturbations valid; we'll also clamp to [lb, ub]
        # explicitly inside the attacks.
        eps = float(eps_vec.max().item())  # max keeps box-feasible attack
        with torch.no_grad():
            y = int(network(x_centre.unsqueeze(0)).argmax(dim=1).item())
        # Stash spec + box bounds on the network object so audit_instance
        # can use spec-aware cex checking (for non-classification specs).
        # The classification audit (default) just checks argmax flip.
        network._audit_spec = spec
        network._audit_lb = lb_t
        network._audit_ub = ub_t
        return network, x_centre, y, eps

    return _load


def get_loader(benchmark: str, args) -> Callable:
    """Dispatch to the right loader for ``benchmark``."""
    import os
    if benchmark == 'cifar10_resnet110':
        return _classification_loader_cifar10_resnet110(args)
    if benchmark == 'vit_small_cifar10':
        return _classification_loader_vit_small_cifar10(args)
    if benchmark == 'yolo_2023':
        root = Path(os.path.expanduser(
            '~/v/other/VNNCOMP/vnncomp2025_benchmarks/benchmarks/yolo_2023'))
        return _vnncomp_loader_factory(root)
    if benchmark == 'vit_2023':
        root = Path(os.path.expanduser(
            '~/v/other/VNNCOMP/vnncomp2025_benchmarks/benchmarks/vit_2023'))
        return _vnncomp_loader_factory(root)
    raise ValueError(f'unknown benchmark: {benchmark}')


# ---------------------------------------------------------------------------
# Counterexample check
# ---------------------------------------------------------------------------

def _is_cex_classification(network, x_adv: torch.Tensor, y_clean: int) -> bool:
    """Standard classification cex: argmax(f(x_adv)) != y_clean."""
    with torch.no_grad():
        pred = int(network(x_adv.unsqueeze(0) if x_adv.dim() == 1 else x_adv)
                   .argmax(dim=1).flatten()[0].item())
    return pred != y_clean


def _is_cex_spec(network, x_adv: torch.Tensor) -> bool:
    """Spec-aware cex check using the spec stashed on the network.

    Returns True iff x_adv lies inside the box and the spec evaluates to
    "unsafe" (i.e. ``G y <= g`` for some HalfSpace, by SAT/UNSAT
    convention; HalfSpace.contains tests this). Falls back to False if
    no spec was attached.
    """
    spec = getattr(network, '_audit_spec', None)
    if spec is None:
        return False
    lb = getattr(network, '_audit_lb', None)
    ub = getattr(network, '_audit_ub', None)
    if lb is not None and ub is not None:
        lb_d = lb.to(x_adv.device)
        ub_d = ub.to(x_adv.device)
        if (x_adv < lb_d).any() or (x_adv > ub_d).any():
            return False
    with torch.no_grad():
        y = network(x_adv.unsqueeze(0) if x_adv.dim() == 1 else x_adv)
        y_np = y.detach().cpu().numpy().reshape(-1)
    from n2v.sets.halfspace import HalfSpace
    if isinstance(spec, HalfSpace):
        return bool(spec.contains(y_np))
    if isinstance(spec, list) and len(spec) > 0:
        if isinstance(spec[0], HalfSpace):
            # OR-of-ANDs: cex if ANY HalfSpace contains y.
            return any(hs.contains(y_np) for hs in spec)
        if isinstance(spec[0], dict):
            # AND-of-OR groups: each group has 'Hg' which is a single
            # HalfSpace whose rows encode an OR (any row's constraint
            # satisfied => group satisfied). cex iff ALL groups satisfied.
            for g in spec:
                Hg = g.get('Hg', None)
                if Hg is None:
                    return False
                # HalfSpace.contains tests G y <= g rowwise; for OR-of-rows
                # we want ANY row satisfied: (G @ y) <= g.flatten() for any
                # row. Compute manually for clarity.
                y2 = y_np.reshape(-1)
                Gy = Hg.G @ y2
                gflat = np.asarray(Hg.g).reshape(-1)
                if not bool(np.any(Gy <= gflat)):
                    return False
            return True
    return False


def _check_cex(network, x_adv: torch.Tensor, y_clean: int) -> bool:
    """Use spec-aware check when a spec is attached (VNN-COMP), else
    classification argmax flip.
    """
    if hasattr(network, '_audit_spec'):
        return _is_cex_spec(network, x_adv)
    return _is_cex_classification(network, x_adv, y_clean)


# ---------------------------------------------------------------------------
# AutoAttack (with manual APGD-CE fallback)
# ---------------------------------------------------------------------------

def autoattack_audit(network, x_clean: torch.Tensor, y_clean: int,
                     eps: float, device: str = 'cpu') -> tuple[bool, np.ndarray | None]:
    """Run AutoAttack standard. Returns (found_cex, x_adv_or_None)."""
    try:
        from autoattack import AutoAttack
    except ImportError:
        return _manual_apgd_audit(network, x_clean, y_clean, eps, device=device)
    network.to(device)
    network.eval()
    x_b = x_clean.to(device).unsqueeze(0)
    y_b = torch.tensor([y_clean], dtype=torch.long, device=device)
    adversary = AutoAttack(network, norm='Linf', eps=eps,
                           version='standard', verbose=False, device=device)
    try:
        x_adv = adversary.run_standard_evaluation(x_b, y_b, bs=1)
    except Exception as e:
        print(f'  autoattack raised {type(e).__name__}: {e}; '
              f'falling back to manual APGD', file=sys.stderr)
        return _manual_apgd_audit(network, x_clean, y_clean, eps, device=device)
    x_adv1 = x_adv.squeeze(0).detach()
    if _check_cex(network, x_adv1, y_clean):
        return True, x_adv1.cpu().numpy()
    return False, None


def _manual_apgd_audit(network, x_clean: torch.Tensor, y_clean: int,
                        eps: float, *, n_steps: int = 100, n_restarts: int = 5,
                        device: str = 'cpu') -> tuple[bool, np.ndarray | None]:
    """Manual APGD-CE fallback used when ``autoattack`` is not installed.

    Implements an Auto-PGD-CE inner loop with momentum + step-size halving
    on plateau (Croce & Hein 2020, simplified). Sufficient for a smoke /
    fallback audit; the full AutoAttack ensemble is preferred.
    """
    network.to(device)
    network.eval()
    x_clean = x_clean.to(device)
    y_t = torch.tensor([y_clean], dtype=torch.long, device=device)
    lb = getattr(network, '_audit_lb', x_clean - eps).to(device)
    ub = getattr(network, '_audit_ub', x_clean + eps).to(device)
    lb = torch.maximum(lb, x_clean - eps)
    ub = torch.minimum(ub, x_clean + eps)

    for restart in range(n_restarts):
        # Random init in the L_inf box.
        delta = (torch.rand_like(x_clean) * 2 - 1) * eps
        x = torch.clamp(x_clean + delta, lb, ub).detach().requires_grad_(True)
        step = 2.0 * eps / max(1, n_steps // 4)
        prev_loss = -1e9
        for t in range(n_steps):
            logits = network(x.unsqueeze(0))
            loss = torch.nn.functional.cross_entropy(logits, y_t)
            grad = torch.autograd.grad(loss, x)[0]
            with torch.no_grad():
                x = x + step * grad.sign()
                x = torch.clamp(x, lb, ub)
            x = x.detach().requires_grad_(True)
            cur_loss = loss.item()
            if cur_loss < prev_loss + 1e-6:
                step *= 0.75
            prev_loss = cur_loss
        x_adv = x.detach()
        if _check_cex(network, x_adv, y_clean):
            return True, x_adv.cpu().numpy()
    return False, None


# ---------------------------------------------------------------------------
# 5K-restart PGD
# ---------------------------------------------------------------------------

def pgd_5k_audit(network, x_clean: torch.Tensor, y_clean: int, eps: float, *,
                 n_restarts: int = 5000, n_steps: int = 100,
                 step_size: float | None = None,
                 device: str = 'cpu',
                 progress_every: int = 100) -> tuple[bool, np.ndarray | None]:
    """5K-restart PGD. Each restart: random init in the L_inf eps-box,
    ``n_steps`` steps of sign-grad ascent on cross-entropy, project to
    box, check argmax flip / spec-cex. Break on first cex.

    Args:
        step_size: If None, defaults to ``2.5 * eps / n_steps`` (a common
            PGD heuristic — slightly larger than ``eps/n_steps`` so the
            attack can traverse the full radius).

    Returns ``(found_cex, x_adv_ndarray_or_None)``.
    """
    network.to(device)
    network.eval()
    x_clean = x_clean.to(device)
    y_t = torch.tensor([y_clean], dtype=torch.long, device=device)
    if step_size is None:
        step_size = 2.5 * eps / max(1, n_steps)
    # Box bounds (use stashed spec box if available; else plain eps-ball
    # around x_clean clipped to [0, 1]^d for image domains).
    lb = getattr(network, '_audit_lb', None)
    ub = getattr(network, '_audit_ub', None)
    if lb is None or ub is None:
        lb = torch.clamp(x_clean - eps, 0.0, 1.0).to(device)
        ub = torch.clamp(x_clean + eps, 0.0, 1.0).to(device)
    else:
        lb = lb.to(device)
        ub = ub.to(device)
        # Intersect with eps-ball around x_clean (defensive; for VNN-COMP
        # the box IS the spec box, so the intersection is just the box).
        lb = torch.maximum(lb, x_clean - eps)
        ub = torch.minimum(ub, x_clean + eps)

    for restart in range(n_restarts):
        delta = (torch.rand_like(x_clean) * 2 - 1) * eps
        x = torch.clamp(x_clean + delta, lb, ub).detach().requires_grad_(True)
        for _ in range(n_steps):
            logits = network(x.unsqueeze(0))
            loss = torch.nn.functional.cross_entropy(logits, y_t)
            grad = torch.autograd.grad(loss, x)[0]
            with torch.no_grad():
                x = x + step_size * grad.sign()
                x = torch.clamp(x, lb, ub)
            x = x.detach().requires_grad_(True)
        x_adv = x.detach()
        if _check_cex(network, x_adv, y_clean):
            return True, x_adv.cpu().numpy()
        if progress_every and (restart + 1) % progress_every == 0:
            print(f'    pgd: {restart + 1}/{n_restarts} restarts, no cex yet',
                  flush=True)
    return False, None


# ---------------------------------------------------------------------------
# Per-instance audit
# ---------------------------------------------------------------------------

def audit_instance(loader_fn, instance_name: str, *, skip_aa: bool, skip_pgd: bool,
                   pgd_restarts: int, pgd_steps: int, device: str) -> dict:
    """Run AutoAttack + 5K-restart PGD on one instance. Returns a dict
    suitable for CSV output.
    """
    t0 = time.time()
    try:
        out = loader_fn(instance_name)
    except Exception as e:
        return {'audit_attack': 'none', 'found_cex': 0, 'cex_pred': '',
                'audit_wall_s': f'{time.time() - t0:.1f}',
                'error': f'loadfailed {type(e).__name__}: {e}'}
    if out is None:
        return {'audit_attack': 'none', 'found_cex': 0, 'cex_pred': '',
                'audit_wall_s': f'{time.time() - t0:.1f}',
                'error': 'instance not in loader index'}
    network, x_clean, y_clean, eps = out

    # Sanity: skip if loader yielded a NaN/inf input or mismatched dim.
    if not torch.isfinite(x_clean).all():
        return {'audit_attack': 'none', 'found_cex': 0, 'cex_pred': '',
                'audit_wall_s': f'{time.time() - t0:.1f}',
                'error': 'non-finite x_clean'}

    # Pass 1: AutoAttack.
    if not skip_aa:
        found, x_adv = autoattack_audit(network, x_clean, y_clean, eps,
                                         device=device)
        if found:
            with torch.no_grad():
                pred = int(network(torch.as_tensor(x_adv,
                                                     dtype=torch.float32,
                                                     device=device).unsqueeze(0))
                           .argmax(dim=1).item())
            return {'audit_attack': 'autoattack', 'found_cex': 1,
                    'cex_pred': pred, 'audit_wall_s': f'{time.time() - t0:.1f}',
                    'error': ''}

    # Pass 2: 5K-restart PGD.
    if not skip_pgd:
        found, x_adv = pgd_5k_audit(network, x_clean, y_clean, eps,
                                     n_restarts=pgd_restarts,
                                     n_steps=pgd_steps, device=device)
        if found:
            with torch.no_grad():
                pred = int(network(torch.as_tensor(x_adv,
                                                     dtype=torch.float32,
                                                     device=device).unsqueeze(0))
                           .argmax(dim=1).item())
            return {'audit_attack': 'pgd', 'found_cex': 1,
                    'cex_pred': pred, 'audit_wall_s': f'{time.time() - t0:.1f}',
                    'error': ''}

    return {'audit_attack': 'none', 'found_cex': 0, 'cex_pred': '',
            'audit_wall_s': f'{time.time() - t0:.1f}', 'error': ''}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description='Exp 2 soundness audit')
    p.add_argument('--input-csv', type=Path, required=True,
                   help='CSV with verdicts (must have instance_name + verdict columns)')
    p.add_argument('--benchmark', required=True,
                   choices=['cifar10_resnet110', 'vit_small_cifar10',
                            'yolo_2023', 'vit_2023'])
    p.add_argument('--output-csv', type=Path, default=None,
                   help='Output CSV path (default: outputs/exp2_<benchmark>_audit.csv)')
    p.add_argument('--filter-verdict', default='UNSAT',
                   help='Audit only rows with this verdict (default UNSAT)')
    p.add_argument('--max-instances', type=int, default=None,
                   help='Audit at most N instances (default all)')
    p.add_argument('--skip-aa', action='store_true',
                   help='Skip AutoAttack pass')
    p.add_argument('--skip-pgd', action='store_true',
                   help='Skip 5K-restart PGD pass')
    p.add_argument('--pgd-restarts', type=int, default=5000,
                   help='Number of PGD random restarts (default 5000)')
    p.add_argument('--pgd-steps', type=int, default=100,
                   help='Steps per PGD restart (default 100)')
    p.add_argument('--device',
                   default='cuda' if torch.cuda.is_available() else 'cpu',
                   help='torch device (default: auto)')
    # CIFAR10 ResNet-110 sigma flag (dispatched to loader)
    p.add_argument('--sigma', default='0.25',
                   choices=['0.12', '0.25', '0.50', '1.00'],
                   help='RS sigma for cifar10_resnet110 loader')
    p.add_argument('--prefer', default='timm',
                   choices=['timm', 'torchvision', 'tiny'],
                   help='ViT backbone preference for vit_small_cifar10 loader')
    args = p.parse_args()

    if not args.input_csv.exists():
        print(f'ERROR: input CSV not found: {args.input_csv}', file=sys.stderr)
        sys.exit(2)

    # Default output path.
    out_csv = args.output_csv or (
        _OUT_DIR / f'exp2_{args.benchmark}_audit.csv')
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # AutoAttack availability.
    try:
        import autoattack  # noqa: F401
        aa_available = True
    except ImportError:
        aa_available = False
    print(f'[audit] autoattack available: {aa_available}'
          f'{"" if aa_available else " (will use manual APGD-CE fallback)"}',
          flush=True)

    # Read input rows; filter by verdict.
    with open(args.input_csv, newline='') as f:
        rows = list(csv.DictReader(f))
    target_rows = [r for r in rows if r.get('verdict') == args.filter_verdict]
    if args.max_instances is not None:
        target_rows = target_rows[:args.max_instances]
    print(f'[audit] {len(target_rows)} {args.filter_verdict} rows to audit '
          f'(of {len(rows)} total) from {args.input_csv}', flush=True)

    if not target_rows:
        print(f'[audit] no rows matched verdict={args.filter_verdict}; '
              f'writing empty output and exiting', flush=True)
        with open(out_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=[
                'benchmark', 'instance_name', 'seed', 'original_verdict',
                'audit_attack', 'found_cex', 'cex_pred', 'audit_wall_s',
                'error'])
            w.writeheader()
        return

    # Build benchmark loader (may fail with SystemExit if data missing).
    try:
        loader_fn = get_loader(args.benchmark, args)
    except SystemExit as e:
        print(f'[audit] {e}', file=sys.stderr)
        # Write empty output so callers can see the audit ran but had no
        # benchmark data to audit against.
        with open(out_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=[
                'benchmark', 'instance_name', 'seed', 'original_verdict',
                'audit_attack', 'found_cex', 'cex_pred', 'audit_wall_s',
                'error'])
            w.writeheader()
        sys.exit(0)

    # Audit loop.
    fields = ['benchmark', 'instance_name', 'seed', 'original_verdict',
              'audit_attack', 'found_cex', 'cex_pred', 'audit_wall_s',
              'error']
    n_cex = 0
    t_start = time.time()
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        f.flush()
        for k, r in enumerate(target_rows, start=1):
            inst = r.get('instance_name', '')
            elapsed = time.time() - t_start
            print(f'[audit {k}/{len(target_rows)} t={elapsed:.0f}s] {inst}',
                  flush=True)
            audit = audit_instance(loader_fn, inst,
                                    skip_aa=args.skip_aa,
                                    skip_pgd=args.skip_pgd,
                                    pgd_restarts=args.pgd_restarts,
                                    pgd_steps=args.pgd_steps,
                                    device=args.device)
            out_row = {fld: '' for fld in fields}
            out_row['benchmark'] = r.get('benchmark', args.benchmark)
            out_row['instance_name'] = inst
            out_row['seed'] = r.get('seed', '')
            out_row['original_verdict'] = r.get('verdict', '')
            for k2, v in audit.items():
                out_row[k2] = v
            w.writerow(out_row)
            f.flush()
            if audit['found_cex']:
                n_cex += 1
                print(f'  *** FALSE UNSAT: cex via {audit["audit_attack"]} '
                      f'(predicted class={audit["cex_pred"]}) ***', flush=True)
            else:
                print(f'  no cex (attack={audit["audit_attack"]}, '
                      f'wall={audit["audit_wall_s"]}s)', flush=True)

    print(f'\n[audit] === complete ===')
    print(f'[audit] wrote {out_csv}')
    print(f'[audit] total wall: {(time.time() - t_start) / 60:.1f} min')
    print(f'[audit] FALSE-UNSAT count: {n_cex} / {len(target_rows)}')
    if n_cex > 0:
        sys.exit(1)  # signal soundness violation to callers


if __name__ == '__main__':
    main()
