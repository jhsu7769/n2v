"""Sanity-check whether αβ-CROWN can verify the Cohen-RS ResNet-110
when loaded via PyTorch-native model_defs (bypassing the ONNX export
path that errored on every config we tried).

Three steps:

1. Load the Cohen RS checkpoint (``Sequential(NormalizeLayer,
   DataParallel(resnet))``), unwrap, strip the DataParallel "module."
   prefix, save the bare-ResNet state_dict to αβ-CROWN's expected
   location (``alpha-beta-CROWN/complete_verifier/models/
   cifar10_resnet/resnet110_cohen_rs.pth``).

2. Write an αβ-CROWN config that:
   * uses our newly-registered ``cifar_resnet110_cohen_rs`` model_def
   * normalizes inputs via the Cohen RS mean/std (αβ-CROWN handles
     normalization at the data-loading layer, not in the model graph)
   * uses the cifar100 vnncomp24 solver knobs (bs=1024, beta-CROWN)
   * skips PGD (we want pure sound-verification cost)

3. Run αβ-CROWN on a single CIFAR-10 test image with ε = 8/255.

If this works, the issue is our ONNX export, not αβ-CROWN's
auto_LiRPA. If it fails with the same shape mismatch, it's a genuine
auto_LiRPA limitation on this architecture.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import torch


_ABCROWN_REPO = Path(os.path.expanduser('~/v/other/alpha-beta-CROWN'))
_ABCROWN_PYTHON = Path(os.path.expanduser(
    '~/miniconda3/envs/alpha-beta-crown/bin/python'))
_ABCROWN_ENTRY = _ABCROWN_REPO / 'complete_verifier' / 'abcrown.py'
_MODEL_DST = (
    _ABCROWN_REPO / 'complete_verifier' / 'models' / 'cifar10_resnet'
    / 'resnet110_cohen_rs.pth'
)
_CONFIG_DST = (
    _ABCROWN_REPO / 'complete_verifier' / 'exp_configs' / 'tutorial_examples'
    / 'cohen_rs_resnet110.yaml'
)
_COHEN_CHECKPOINT = Path(os.path.expanduser(
    '~/v/other/smoothing/models/cifar10/resnet110/noise_0.25/checkpoint.pth.tar'))


def step1_convert_checkpoint() -> None:
    """Unwrap the Cohen RS checkpoint to a bare-ResNet state_dict."""
    if not _COHEN_CHECKPOINT.exists():
        raise FileNotFoundError(
            f'Cohen RS checkpoint missing at {_COHEN_CHECKPOINT}')

    state = torch.load(str(_COHEN_CHECKPOINT), map_location='cpu',
                        weights_only=False)
    sd = state.get('state_dict', state)

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
            # NormalizeLayer params (mean/std) — αβ-CROWN handles
            # normalization via config, so drop these.
            continue
        new_sd[nk] = v

    _MODEL_DST.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new_sd, str(_MODEL_DST))
    print(f'[1/3] Converted checkpoint → {_MODEL_DST}')
    print(f'      keys: {len(new_sd)} (sample: '
          f'{list(new_sd.keys())[:3]})')


_CONFIG_YAML = """\
# Sanity-check: αβ-CROWN on Cohen RS ResNet-110 via PyTorch-native loading.
# Bypasses ONNX entirely. If this works, our ONNX export was the issue;
# if it fails the same way, the problem is auto_LiRPA's bound prop on
# this 110-layer architecture.

model:
  name: cifar_resnet110_cohen_rs
  path: models/cifar10_resnet/resnet110_cohen_rs.pth

data:
  dataset: CIFAR
  mean: [0.4914, 0.4822, 0.4465]
  std: [0.2023, 0.1994, 0.2010]
  start: 0
  end: 1   # one instance is enough for the sanity check

specification:
  norm: .inf
  epsilon: 0.03137254901960784   # 8/255

attack:
  pgd_order: skip   # pure sound verification

solver:
  batch_size: 32         # tiny — alpha tensors are O(batch × width × depth)
  alpha-crown:
    share_alphas: true   # share α params across layers to cut memory
    iteration: 50
  beta-crown:
    iteration: 20
    lr_beta: 0.05

bab:
  timeout: 600
  branching:
    method: kfsb
    candidates: 3
    reduceop: max
"""


def step2_write_config() -> None:
    _CONFIG_DST.parent.mkdir(parents=True, exist_ok=True)
    _CONFIG_DST.write_text(_CONFIG_YAML)
    print(f'[2/3] Wrote config → {_CONFIG_DST}')


def step3_run() -> int:
    print(f'[3/3] Running αβ-CROWN ...')
    cmd = [
        str(_ABCROWN_PYTHON), str(_ABCROWN_ENTRY),
        '--config', str(_CONFIG_DST),
    ]
    env = os.environ.copy()
    env.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    proc = subprocess.run(
        cmd, cwd=str(_ABCROWN_REPO / 'complete_verifier'),
        env=env,
    )
    return proc.returncode


def main() -> int:
    step1_convert_checkpoint()
    step2_write_config()
    return step3_run()


if __name__ == '__main__':
    sys.exit(main())
