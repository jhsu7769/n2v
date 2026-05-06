"""Checkerboard: sharp-boundary 2D FM benchmark."""

from pathlib import Path
import torch

from examples.FlowConformal.smokes.fm_validation._common import (
    train_and_render,
)


def sample_checkerboard(n, seed, cells=4):
    gen = torch.Generator().manual_seed(seed)
    samples = []
    while len(samples) < n:
        batch = (torch.rand(n, 2, generator=gen) * 2 - 1) * cells / 2
        ix = batch[:, 0].floor().long()
        iy = batch[:, 1].floor().long()
        keep = ((ix + iy) % 2 == 0)
        samples.append(batch[keep])
        if sum(s.shape[0] for s in samples) >= n:
            break
    return torch.cat(samples, dim=0)[:n]


def main():
    out_dir = Path(__file__).parent / 'figures'
    out_dir.mkdir(exist_ok=True)
    train_and_render(
        sample_checkerboard,
        title='Checkerboard',
        out_path=out_dir / 'checkerboard.png',
        n_epochs=1000,
    )


if __name__ == '__main__':
    main()
