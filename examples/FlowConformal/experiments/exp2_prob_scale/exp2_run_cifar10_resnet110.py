"""Exp 2: CIFAR-10 ResNet-110 (RS adv-trained) at ε=8/255 sweep.

Loads the Cohen-et-al randomized-smoothing pretrained ResNet-110 (any
sigma; we default to noise_0.25 since that's the closest to "clean +
adv-trained" available), then evaluates classification-robustness on
CIFAR-10 test images at L∞ ε=8/255.

Pretrained weights expected at:
  ``~/v/other/smoothing/models/cifar10/resnet110/noise_<sigma>/checkpoint.pth.tar``

If the weights file is missing, the script prints a TODO with the URL
and exits gracefully (no smoke output) so the rest of the sweep can
proceed.

Usage:
    cd /home/sasakis/v/tools/n2v
    /home/sasakis/miniconda3/envs/n2v/bin/python -u -m \\
        examples.FlowConformal.experiments.exp2_prob_scale.exp2_run_cifar10_resnet110 \\
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

from examples.FlowConformal.experiments.exp2_prob_scale._common import (
    add_common_args, get_pipeline_kwargs,
    linf_box_for_image, make_classification_robustness_spec, run_sweep,
)


_BENCHMARK_NAME = 'cifar10_resnet110'
_OUT_DIR = Path(__file__).parent / 'outputs'

_EPS = 8.0 / 255.0
_NUM_CLASSES = 10
_SMOOTHING_REPO = Path(os.path.expanduser('~/v/other/smoothing'))
_DEFAULT_SIGMA = '0.25'  # noise_0.25; can override via --sigma
_WEIGHTS_URL = ('https://drive.google.com/file/d/'
                '1h_TpbXm5haY5f-l4--IKylmdz6tvPoR4/view?usp=sharing')


# ---------------------------------------------------------------------------
# Network construction
# ---------------------------------------------------------------------------

def _build_resnet110() -> nn.Module:
    """Construct the same architecture used by the RS repo
    (``cifar_resnet110``: 18 BasicBlocks per stage, 16/32/64 channels)
    using a transcribed version of ``smoothing/code/archs/cifar_resnet.py``
    so we don't need the RS repo on PYTHONPATH.
    """
    import math

    def _conv3x3(c_in, c_out, stride=1):
        return nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride,
                         padding=1, bias=False)

    class BasicBlock(nn.Module):
        expansion = 1

        def __init__(self, inplanes, planes, stride=1, downsample=None):
            super().__init__()
            self.conv1 = _conv3x3(inplanes, planes, stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = _conv3x3(planes, planes)
            self.bn2 = nn.BatchNorm2d(planes)
            self.downsample = downsample

        def forward(self, x):
            residual = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)
            if self.downsample is not None:
                residual = self.downsample(x)
            out = out + residual
            out = self.relu(out)
            return out

    class ResNet(nn.Module):

        def __init__(self, depth=110, num_classes=10):
            super().__init__()
            assert (depth - 2) % 6 == 0
            n = (depth - 2) // 6
            self.inplanes = 16
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(16)
            self.relu = nn.ReLU(inplace=True)
            self.layer1 = self._make_layer(BasicBlock, 16, n)
            self.layer2 = self._make_layer(BasicBlock, 32, n, stride=2)
            self.layer3 = self._make_layer(BasicBlock, 64, n, stride=2)
            self.avgpool = nn.AvgPool2d(8)
            self.fc = nn.Linear(64, num_classes)
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    fan = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2.0 / fan))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

        def _make_layer(self, block, planes, blocks, stride=1):
            downsample = None
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            layers = [block(self.inplanes, planes, stride, downsample)]
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes))
            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    return ResNet(depth=110, num_classes=10)


class _NormalizeAndReshape(nn.Module):
    """Wrap a ResNet so it accepts flat ``(B, 3072)`` and applies the
    CIFAR-10 mean/std normalization the RS repo bakes into its model
    (``Sequential(NormalizeLayer, base_model)``).
    """
    _MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    _STD = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        self.register_buffer('_mean', self._MEAN.clone())
        self.register_buffer('_std', self._STD.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.view(-1, 3, 32, 32)
        x = (x - self._mean) / self._std
        return self.base(x)


def _load_pretrained(sigma: str) -> 'nn.Module | None':
    """Try to load Cohen-et-al RS pretrained ResNet-110 for CIFAR-10.
    Returns ``None`` if the checkpoint is missing — caller prints TODO.
    """
    ckpt = (_SMOOTHING_REPO / 'models' / 'cifar10' / 'resnet110' /
            f'noise_{sigma}' / 'checkpoint.pth.tar')
    if not ckpt.exists():
        return None

    base = _build_resnet110()
    wrapped = _NormalizeAndReshape(base)

    state = torch.load(str(ckpt), map_location='cpu', weights_only=False)
    sd = state.get('state_dict', state)

    # The RS repo wraps the model as `Sequential(NormalizeLayer,
    # DataParallel(resnet110))`. Strip "1.module." and "0." prefixes so
    # weights map onto our `base` directly. The normalization layer is
    # baked into our wrapper, so the "0." prefix entries (mean/std
    # buffers) are dropped.
    new_sd = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith('module.'):
            nk = nk[len('module.'):]
        if nk.startswith('1.module.'):
            nk = nk[len('1.module.'):]
        elif nk.startswith('1.'):
            nk = nk[len('1.'):]
        elif nk.startswith('0.'):
            continue  # NormalizeLayer's mean/std — provided by our wrapper
        new_sd[nk] = v

    missing, unexpected = base.load_state_dict(new_sd, strict=False)
    if unexpected:
        print(f'[{_BENCHMARK_NAME}] WARN: unexpected keys: {unexpected[:5]}',
              file=sys.stderr)
    wrapped.eval()
    return wrapped


# ---------------------------------------------------------------------------
# CIFAR-10 test data
# ---------------------------------------------------------------------------

def _load_cifar10_test(n: int):
    """Load up to ``n`` test images from CIFAR-10. Caches under
    ``~/.cache/n2v_exp2_cifar10``.

    Returns ``(images_np[n,3072], labels_np[n])`` with float32 in [0,1].
    """
    import torchvision
    cache = Path(os.path.expanduser('~/.cache/n2v_exp2_cifar10'))
    cache.mkdir(parents=True, exist_ok=True)
    ds = torchvision.datasets.CIFAR10(str(cache), train=False, download=True)
    # ds.data has shape (10000, 32, 32, 3) uint8.
    imgs = ds.data[:n].astype(np.float32) / 255.0  # [n, 32, 32, 3]
    imgs = np.transpose(imgs, (0, 3, 1, 2))         # [n, 3, 32, 32]
    imgs = imgs.reshape(imgs.shape[0], -1)           # [n, 3072]
    labels = np.asarray(ds.targets[:n], dtype=np.int64)
    return imgs, labels


# ---------------------------------------------------------------------------
# Per-instance loader factory
# ---------------------------------------------------------------------------

def _make_loader(network, x: np.ndarray, true_label: int, idx: int):
    name = f'cifar10_test_{idx:04d}_label_{true_label}'

    def _load():
        # Predicted class on the clean image (used by the spec); if it
        # differs from the true label, still verify wrt the predicted
        # class — that's the interesting "robustness around the model's
        # own decision" question.
        #
        # Device note: the RS process_one path calls ``net.to('cuda')``
        # in-place on the first instance, which moves ``network``'s
        # params to cuda. On subsequent instances the loader runs
        # before process_one re-checks device, so we must put the input
        # on whatever device the params are currently on. Without this,
        # the second-instance forward pass raises
        # ``RuntimeError: Expected all tensors to be on the same device``.
        try:
            net_device = next(network.parameters()).device
        except StopIteration:
            net_device = torch.device('cpu')
        with torch.no_grad():
            x_t = torch.as_tensor(
                x.reshape(1, -1), dtype=torch.float32, device=net_device)
            y = network(x_t).cpu().numpy().flatten()
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
    parser.add_argument(
        '--sigma', type=str, default=_DEFAULT_SIGMA,
        choices=['0.12', '0.25', '0.50', '1.00'],
        help='RS noise sigma for the pretrained checkpoint.')
    args = parser.parse_args()

    network = _load_pretrained(args.sigma)
    if network is None:
        print(f'[{_BENCHMARK_NAME}] TODO: RS pretrained weights not found at',
              file=sys.stderr)
        print(f'  {_SMOOTHING_REPO}/models/cifar10/resnet110/noise_{args.sigma}/'
              'checkpoint.pth.tar', file=sys.stderr)
        print(f'  download from {_WEIGHTS_URL}', file=sys.stderr)
        print(f'  (extract the gdrive `models` directory into '
              f'{_SMOOTHING_REPO}/)', file=sys.stderr)
        sys.exit(0)  # graceful skip

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
    default = (_OUT_DIR / f'exp2_{_BENCHMARK_NAME}_ours_sigma{args.sigma}'
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
