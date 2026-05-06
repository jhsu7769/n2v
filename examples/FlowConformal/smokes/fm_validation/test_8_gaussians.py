"""8-Gaussians ring: multimodal 2D FM benchmark."""

import math
from pathlib import Path
import torch

from examples.FlowConformal.smokes.fm_validation._common import (
    train_and_render,
)


def sample_8_gaussians(n, seed, radius=3.0, std=0.3):
    gen = torch.Generator().manual_seed(seed)
    centers = torch.tensor([
        [radius * math.cos(2 * math.pi * k / 8),
         radius * math.sin(2 * math.pi * k / 8)]
        for k in range(8)
    ])
    labels = torch.randint(0, 8, (n,), generator=gen)
    noise = torch.randn(n, 2, generator=gen) * std
    return centers[labels] + noise


def main():
    out_dir = Path(__file__).parent / 'figures'
    out_dir.mkdir(exist_ok=True)
    train_and_render(
        sample_8_gaussians,
        title='8 Gaussians',
        out_path=out_dir / '8_gaussians.png',
        n_epochs=800,
    )


if __name__ == '__main__':
    main()
