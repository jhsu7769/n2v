"""GPU batch-size + n_train ablation for the flow trainer.

Goal: identify the bottleneck behind ~22% GPU utilization in the
production pipeline. We sweep:

  * n_train in {1000, 2000, 5000, 10000, 20000} — how much training data
  * batch_size in {512, 1024, 2048, 4096, 8192} — per-step GPU work
  * flow_epochs fixed at 1000 — same step count across cells

For each cell we measure:

  * wall time (s)
  * peak GPU memory (MB)
  * average GPU utilization (% sampled every 200ms during training)
  * final loss

Pure flow-training only — no network forward pass, no AMLS, no
falsifier. Uses synthetic 2-D banana-style data (a Gaussian mixture)
so the data prep is trivial and we measure pure trainer overhead.
"""
from __future__ import annotations

import argparse
import csv
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

_PROJ_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))


# Sweep configuration -------------------------------------------------------

N_TRAIN_VALUES = [1000, 2000, 5000, 10000, 20000]
BATCH_SIZE_VALUES = [512, 1024, 2048, 4096, 8192]
FLOW_EPOCHS = 1000


# GPU utilization sampler ---------------------------------------------------

class _GpuUtilSampler(threading.Thread):
    """Background thread that samples ``nvidia-smi`` every ``period``
    seconds and records utilization + memory.
    """

    def __init__(self, period: float = 0.2):
        super().__init__(daemon=True)
        self.period = period
        self._stop_event = threading.Event()
        self.utils: list[float] = []
        self.mems: list[float] = []

    def run(self):
        import subprocess
        while not self._stop_event.is_set():
            try:
                out = subprocess.check_output(
                    ['nvidia-smi',
                     '--query-gpu=utilization.gpu,memory.used',
                     '--format=csv,noheader,nounits'],
                    timeout=1.0,
                ).decode().strip().split('\n')[0]
                util_s, mem_s = out.split(',')
                self.utils.append(float(util_s.strip()))
                self.mems.append(float(mem_s.strip()))
            except Exception:
                pass
            self._stop_event.wait(self.period)

    def stop(self):
        self._stop_event.set()
        self.join(timeout=2)


def _gen_data(n_train: int, dim: int = 2, seed: int = 0) -> torch.Tensor:
    """Synthetic 2-mode mixture in ``dim`` dimensions."""
    rng = np.random.default_rng(seed)
    half = n_train // 2
    a = rng.normal(loc=[+1.5] * dim, scale=0.4, size=(half, dim))
    b = rng.normal(loc=[-1.5] * dim, scale=0.4, size=(n_train - half, dim))
    data = np.concatenate([a, b], axis=0)
    rng.shuffle(data)
    return torch.tensor(data, dtype=torch.float32)


def run_one_cell(n_train: int, batch_size: int, n_epochs: int,
                 *, seed: int = 0) -> dict:
    """Train a flow once and return timing + utilization metrics."""
    from n2v.probabilistic.verify_flow import _train_flow

    y_train = _gen_data(n_train, dim=2, seed=seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    sampler = _GpuUtilSampler(period=0.2)
    sampler.start()
    t0 = time.time()
    _, losses = _train_flow(
        y_train, dim=2, n_epochs=n_epochs, seed=seed,
        batch_size=batch_size, return_losses=True,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall = time.time() - t0
    sampler.stop()

    util_avg = float(np.mean(sampler.utils)) if sampler.utils else float('nan')
    util_max = float(np.max(sampler.utils)) if sampler.utils else float('nan')
    mem_max = float(np.max(sampler.mems)) if sampler.mems else float('nan')
    peak_mem_mb = (float(torch.cuda.max_memory_allocated() / (1024 * 1024))
                   if torch.cuda.is_available() else 0.0)
    final_loss = float(losses[-1]) if losses else float('nan')
    initial_loss = float(losses[0]) if losses else float('nan')
    return {
        'n_train': n_train,
        'batch_size': batch_size,
        'n_epochs': n_epochs,
        'wall_s': wall,
        'gpu_util_avg_pct': util_avg,
        'gpu_util_max_pct': util_max,
        'gpu_mem_used_max_mb': mem_max,
        'torch_peak_alloc_mb': peak_mem_mb,
        'loss_initial': initial_loss,
        'loss_final': final_loss,
        'loss_ratio': initial_loss / final_loss if final_loss > 0 else float('nan'),
        'samples': len(sampler.utils),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-csv', type=Path, required=True)
    parser.add_argument('--n-epochs', type=int, default=FLOW_EPOCHS)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    fields = [
        'n_train', 'batch_size', 'n_epochs', 'wall_s',
        'gpu_util_avg_pct', 'gpu_util_max_pct',
        'gpu_mem_used_max_mb', 'torch_peak_alloc_mb',
        'loss_initial', 'loss_final', 'loss_ratio', 'samples',
    ]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        f.flush()

        print(f'[gpu-probe] device: '
              f'{"cuda" if torch.cuda.is_available() else "cpu"}',
              flush=True)
        print(f'[gpu-probe] sweep: n_train={N_TRAIN_VALUES} '
              f'batch_size={BATCH_SIZE_VALUES} n_epochs={args.n_epochs}',
              flush=True)
        print(f'[gpu-probe] {len(N_TRAIN_VALUES) * len(BATCH_SIZE_VALUES)} '
              f'cells total', flush=True)

        t_start = time.time()
        for nt in N_TRAIN_VALUES:
            for bs in BATCH_SIZE_VALUES:
                if bs > nt:
                    print(f'[gpu-probe] skip n_train={nt} batch_size={bs} '
                          f'(batch > data)', flush=True)
                    continue
                print(f'[gpu-probe] n_train={nt} batch_size={bs} ...',
                      flush=True)
                r = run_one_cell(nt, bs, args.n_epochs, seed=args.seed)
                print(f'[gpu-probe]   wall={r["wall_s"]:.1f}s '
                      f'gpu_util_avg={r["gpu_util_avg_pct"]:.1f}% '
                      f'mem={r["gpu_mem_used_max_mb"]:.0f}MB '
                      f'loss {r["loss_initial"]:.4f}->{r["loss_final"]:.4f}',
                      flush=True)
                w.writerow(r)
                f.flush()

        print(f'\n[gpu-probe] done in {(time.time()-t_start)/60:.1f} min',
              flush=True)


if __name__ == '__main__':
    main()
