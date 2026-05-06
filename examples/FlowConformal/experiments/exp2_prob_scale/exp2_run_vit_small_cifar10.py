"""Exp 2: ViT-Small / CIFAR-10 sweep at L∞ ε=8/255.

Loads ``vit_tiny_patch16_224`` from timm pretrained on ImageNet, then
fine-tunes the head on CIFAR-10 (or loads a cached fine-tuned checkpoint
if present). Verifies classification robustness on test images.

Fallback chain when timm is unavailable:
  1. timm.create_model('vit_tiny_patch16_224', pretrained=True), then
     fine-tune to CIFAR-10 (cached at
     ``~/.cache/n2v_exp2_vit_small_cifar10/finetuned.pt``).
  2. torchvision.models.vit_b_16 (heavier; same fine-tune pattern).
  3. A small in-file CIFAR-sized ViT trained from scratch (only used in
     smoke mode; cheap-but-low-accuracy fallback).

The verification spec is the same classification-robustness setup as
the ResNet-110 script (UNSAFE = "any other class beats the predicted
class"). Image is upsampled to the model's native input resolution
(224 for both timm/torchvision; 32 for the CIFAR-sized fallback).

Usage:
    cd /home/sasakis/v/tools/n2v
    /home/sasakis/miniconda3/envs/n2v/bin/python -u -m \\
        examples.FlowConformal.experiments.exp2_prob_scale.exp2_run_vit_small_cifar10 \\
        --smoke
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from examples.FlowConformal.experiments.exp2_prob_scale._common import (
    add_common_args, get_pipeline_kwargs,
    linf_box_for_image, make_classification_robustness_spec, run_sweep,
)


_BENCHMARK_NAME = 'vit_small_cifar10'
_OUT_DIR = Path(__file__).parent / 'outputs'

_EPS = 8.0 / 255.0
_NUM_CLASSES = 10

_CACHE = Path(os.path.expanduser('~/.cache/n2v_exp2_vit_small_cifar10'))
_FINETUNED_CKPT = _CACHE / 'finetuned.pt'


# ---------------------------------------------------------------------------
# Model variants
# ---------------------------------------------------------------------------

class _UpsampleToVit(nn.Module):
    """Wrap a model that takes (B, 3, H, W) — for ViT pretrained backbones,
    H = W = 224 — and accept flat (B, 3072) CIFAR inputs by upsampling and
    normalizing with ImageNet stats.
    """
    _MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    _STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __init__(self, base: nn.Module, target_size: int = 224):
        super().__init__()
        self.base = base
        self.target_size = target_size
        self.register_buffer('_mean', self._MEAN.clone())
        self.register_buffer('_std', self._STD.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.view(-1, 3, 32, 32)
        if x.shape[-1] != self.target_size:
            x = F.interpolate(x, size=self.target_size, mode='bilinear',
                              align_corners=False)
        x = (x - self._mean) / self._std
        return self.base(x)


def _try_load_timm() -> 'nn.Module | None':
    try:
        import timm
    except ImportError:
        return None
    try:
        base = timm.create_model('vit_tiny_patch16_224',
                                 pretrained=True, num_classes=_NUM_CLASSES)
    except Exception as e:
        print(f'[{_BENCHMARK_NAME}] timm.create_model raised '
              f'{type(e).__name__}: {e}', file=sys.stderr)
        return None
    return _UpsampleToVit(base, target_size=224)


def _try_load_torchvision() -> 'nn.Module | None':
    try:
        from torchvision.models import vit_b_16, ViT_B_16_Weights
    except Exception as e:
        print(f'[{_BENCHMARK_NAME}] torchvision ViT import raised '
              f'{type(e).__name__}: {e}', file=sys.stderr)
        return None
    try:
        base = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    except Exception as e:
        print(f'[{_BENCHMARK_NAME}] torchvision vit_b_16 weight load '
              f'raised {type(e).__name__}: {e}', file=sys.stderr)
        return None
    # Replace the 1000-way classifier head with a 10-way head.
    in_features = base.heads.head.in_features
    base.heads.head = nn.Linear(in_features, _NUM_CLASSES)
    return _UpsampleToVit(base, target_size=224)


class _TinyViTCifar(nn.Module):
    """Tiny CIFAR-sized ViT for smoke fallback when no pretrained backbone
    is available. Trains from scratch on CIFAR (low accuracy, but enough
    to exercise the verification pipeline end-to-end).
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.patch_size = 4
        d = 64
        n_tokens = (32 // self.patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, d, kernel_size=self.patch_size,
                                     stride=self.patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens + 1, d))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=4,
                                           dim_feedforward=128,
                                           batch_first=True,
                                           activation='gelu')
        self.encoder = nn.TransformerEncoder(layer, num_layers=4)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.view(-1, 3, 32, 32)
        x = self.patch_embed(x)              # (B, d, h, w)
        x = x.flatten(2).transpose(1, 2)     # (B, h*w, d)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)       # (B, 1+h*w, d)
        x = x + self.pos_embed
        x = self.encoder(x)
        x = self.norm(x[:, 0])
        return self.head(x)


def _load_or_train_tiny_vit() -> nn.Module:
    """Load the smoke-fallback CIFAR-sized ViT, training-from-scratch for
    1 epoch on CIFAR-10 if no checkpoint is cached. Result is low-accuracy
    but enough for smoke verification of the pipeline.
    """
    _CACHE.mkdir(parents=True, exist_ok=True)
    fallback_ckpt = _CACHE / 'tiny_vit.pt'
    model = _TinyViTCifar()
    if fallback_ckpt.exists():
        model.load_state_dict(torch.load(str(fallback_ckpt),
                                          map_location='cpu',
                                          weights_only=False))
        model.eval()
        return model
    # Cheap 1-epoch fine-tune.
    print(f'[{_BENCHMARK_NAME}] training tiny CIFAR ViT (fallback) for 1 '
          f'epoch...', flush=True)
    import torchvision
    ds = torchvision.datasets.CIFAR10(str(_CACHE / 'data'), train=True,
                                       download=True)
    imgs = torch.tensor(ds.data, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    targets = torch.tensor(ds.targets, dtype=torch.long)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    bs = 256
    n = imgs.shape[0]
    perm = torch.randperm(n)
    for i in range(0, n, bs):
        idx = perm[i:i + bs]
        x = imgs[idx].to(device)
        y = targets[idx].to(device)
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
    model.cpu().eval()
    torch.save(model.state_dict(), str(fallback_ckpt))
    return model


def _maybe_finetune(model: nn.Module, epochs: int = 1) -> nn.Module:
    """If a fine-tuned checkpoint is cached, load it; otherwise run a
    cheap fine-tune on CIFAR-10 and cache. ``epochs=1`` is enough for the
    smoke + verify pipeline; the user can replace the cached checkpoint
    with a properly fine-tuned one before the full run.
    """
    _CACHE.mkdir(parents=True, exist_ok=True)
    if _FINETUNED_CKPT.exists():
        try:
            sd = torch.load(str(_FINETUNED_CKPT), map_location='cpu',
                             weights_only=False)
            model.load_state_dict(sd, strict=False)
            model.eval()
            return model
        except Exception:
            pass
    print(f'[{_BENCHMARK_NAME}] fine-tuning {epochs} epochs on CIFAR-10...',
          flush=True)
    import torchvision
    ds = torchvision.datasets.CIFAR10(str(_CACHE / 'data'), train=True,
                                       download=True)
    imgs = torch.tensor(ds.data, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    targets = torch.tensor(ds.targets, dtype=torch.long)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    bs = 64
    n = imgs.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            x = imgs[idx].to(device)
            y = targets[idx].to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
    model.cpu().eval()
    torch.save(model.state_dict(), str(_FINETUNED_CKPT))
    return model


def _load_model(prefer: str) -> tuple['nn.Module | None', str]:
    """Try the model variants in order. Returns (model, name) or (None, ''):
       prefer='timm' -> timm > torchvision > tiny
       prefer='torchvision' -> torchvision > timm > tiny
       prefer='tiny' -> only the tiny fallback
    """
    if prefer == 'tiny':
        return _load_or_train_tiny_vit(), 'tiny_vit_cifar10'

    if prefer == 'torchvision':
        m = _try_load_torchvision()
        if m is not None:
            return _maybe_finetune(m), 'torchvision_vit_b_16'
        m = _try_load_timm()
        if m is not None:
            return _maybe_finetune(m), 'timm_vit_tiny_patch16_224'
    else:  # default: try timm first
        m = _try_load_timm()
        if m is not None:
            return _maybe_finetune(m), 'timm_vit_tiny_patch16_224'
        m = _try_load_torchvision()
        if m is not None:
            return _maybe_finetune(m), 'torchvision_vit_b_16'

    # Last resort
    return _load_or_train_tiny_vit(), 'tiny_vit_cifar10'


# ---------------------------------------------------------------------------
# CIFAR-10 test data (same helper as resnet110 script)
# ---------------------------------------------------------------------------

def _load_cifar10_test(n: int):
    import torchvision
    cache = Path(os.path.expanduser('~/.cache/n2v_exp2_cifar10'))
    cache.mkdir(parents=True, exist_ok=True)
    ds = torchvision.datasets.CIFAR10(str(cache), train=False, download=True)
    imgs = ds.data[:n].astype(np.float32) / 255.0
    imgs = np.transpose(imgs, (0, 3, 1, 2)).reshape(imgs.shape[0], -1)
    labels = np.asarray(ds.targets[:n], dtype=np.int64)
    return imgs, labels


# ---------------------------------------------------------------------------
# Loader factory
# ---------------------------------------------------------------------------

def _make_loader(network, x: np.ndarray, true_label: int, idx: int):
    name = f'cifar10_test_{idx:04d}_label_{true_label}'

    def _load():
        with torch.no_grad():
            y = network(torch.as_tensor(
                x.reshape(1, -1), dtype=torch.float32)).cpu().numpy().flatten()
        c = int(np.argmax(y))
        spec = make_classification_robustness_spec(_NUM_CLASSES, c)
        lb, ub = linf_box_for_image(x, _EPS, lo=0.0, hi=1.0)
        return network, [(lb, ub)], spec, name

    return name, _load


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument('--prefer', choices=['timm', 'torchvision', 'tiny'],
                        default='timm',
                        help='Backbone preference (fallback chain applied).')
    args = parser.parse_args()

    network, model_name = _load_model(args.prefer)
    if network is None:
        print(f'[{_BENCHMARK_NAME}] TODO: could not load any ViT backbone. '
              f'Install timm (`pip install timm`) or check torchvision/'
              f'internet access.', file=sys.stderr)
        sys.exit(0)
    print(f'[{_BENCHMARK_NAME}] using model: {model_name}', flush=True)

    n = 2 if args.smoke else args.instances
    try:
        imgs, labels = _load_cifar10_test(n)
    except Exception as e:
        print(f'[{_BENCHMARK_NAME}] TODO: CIFAR-10 download/load failed: '
              f'{type(e).__name__}: {e}', file=sys.stderr)
        sys.exit(0)

    instances = [_make_loader(network, imgs[i], int(labels[i]), i)
                 for i in range(len(imgs))]

    suffix = '_smoke' if args.smoke else ''
    falsify_tag = '_falsify' if args.falsify_first else ''
    default = (_OUT_DIR / f'exp2_{_BENCHMARK_NAME}_ours_{model_name}'
               f'{falsify_tag}{suffix}.csv')
    out_csv = args.output_csv or default

    run_sweep(
        benchmark_name=_BENCHMARK_NAME,
        instances=instances,
        out_csv=out_csv,
        pipeline_kwargs=get_pipeline_kwargs(args.falsify_first),
        timeout_s=args.timeout,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
