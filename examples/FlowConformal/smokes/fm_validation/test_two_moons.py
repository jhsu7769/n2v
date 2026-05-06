"""Two-moons: canonical 2D FM benchmark.

Generates sklearn two-moons training data, trains a small flow, and
produces the 4-panel figure in fm_validation/figures/two_moons.png.
"""

from pathlib import Path
import torch
from sklearn.datasets import make_moons

from examples.FlowConformal.smokes.fm_validation._common import (
    train_and_render,
)


def sample_moons(n, seed):
    X, _ = make_moons(n_samples=n, noise=0.05, random_state=seed)
    # Normalize to roughly [-1, 1]
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return torch.tensor(X, dtype=torch.float32)


def main():
    out_dir = Path(__file__).parent / 'figures'
    out_dir.mkdir(exist_ok=True)
    train_and_render(
        sample_moons,
        title='Two moons',
        out_path=out_dir / 'two_moons.png',
        n_epochs=800,
    )


if __name__ == '__main__':
    main()
