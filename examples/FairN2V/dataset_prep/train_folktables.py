"""
Train + export the folktables (ACSIncome) classifiers for the FairN2V
`folktables` profile: FT-1, FT-2, FT-3 (small / medium / large, mirroring the
adult AC sizes; numbered like GC-1/2/3).

Unlike adult/german/bank (whose nets shipped with NNV), there is no upstream
model here, so we train our own and export ONNX in the same shape the pipeline
expects: a feed-forward ReLU net ending in Softmax (the loader strips it for
Star reachability), dynamic batch dim, opset 13.

Two conventions are matched to the existing datasets:

  * class_type='min'. The NNV nets predict via argmin (the true class gets the
    SMALLEST output). We reproduce that by training on the NEGATED logits, so
    the correct class is driven to the minimum. argmin(softmax) == argmin(logits)
    so the Softmax tail stays consistent (and is stripped before reachability).

  * normalization must match the adapter. At verification time the
    DatasetAdapter min-max normalizes using stats derived from the SAME npz X we
    load here, so we recompute those exact stats and train on the normalized
    data -- the inputs the net trains on then match what verification feeds it.

Labels use y[:,0] (1 == income >$50k). Run make_folktables_npz.py first.

USAGE (from examples/FairN2V/, after make_folktables_npz.py):
    python dataset_prep/train_folktables.py
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

SEED = 0
EPOCHS = 60
BATCH = 256
LR = 1e-3

# This script lives in examples/FairN2V/dataset_prep/; anchor on the FairN2V
# root so data/ and models/ resolve regardless of cwd.
fairn2v_root = Path(__file__).resolve().parent.parent
data_path = fairn2v_root / 'data' / 'folktables_data.npz'
models_dir = fairn2v_root / 'models'

# name -> hidden layer widths (output=2 fixed): small / medium / large. The
# input width is inferred from the npz (13 for the one-hot-race representation),
# so the same nets serve both the `folktables` (sex) and `folktables_race`
# profiles -- same noun, different fairness verb.
ARCHITECTURES = {
    'FT-1': [16, 8],
    'FT-2': [50],
    'FT-3': [100, 100],
}


def build_net(hidden, input_dim):
    """Feed-forward ReLU classifier ending in Softmax (stripped before reach)."""
    layers = []
    prev = input_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers += [nn.Linear(prev, 2), nn.Softmax(dim=1)]
    return nn.Sequential(*layers)


def normalized_data():
    """Load the npz and apply the EXACT same min-max normalization the adapter
    uses at verification time, so train inputs == verify inputs."""
    data = np.load(data_path)
    X = data['X'].astype(np.float64)          # (n_samples, n_features)
    y0 = data['y'][:, 0].astype(np.int64)     # class index (1 == >$50k)

    min_v = X.min(axis=0)
    max_v = X.max(axis=0)
    var = max_v - min_v > 0
    min_v[~var] = 0.0
    max_v[~var] = 1.0
    Xn = (X - min_v) / (max_v - min_v)
    return Xn.astype(np.float32), y0


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    models_dir.mkdir(exist_ok=True)

    Xn, y0 = normalized_data()
    X_t = torch.from_numpy(Xn)
    y_t = torch.from_numpy(y0)
    n, input_dim = X_t.shape

    for name, hidden in ARCHITECTURES.items():
        torch.manual_seed(SEED)  # same init regime per model for reproducibility
        net = build_net(hidden, input_dim)
        opt = torch.optim.Adam(net.parameters(), lr=LR)
        loss_fn = nn.CrossEntropyLoss()

        # Train on the logits (net without its Softmax tail). class_type='min':
        # we want the TRUE class to be the argMIN, so train CE on the NEGATED
        # logits -- this pushes the correct class to the smallest output.
        logits_net = nn.Sequential(*list(net.children())[:-1])

        net.train()
        for epoch in range(EPOCHS):
            perm = torch.randperm(n)
            for s in range(0, n, BATCH):
                idx = perm[s:s + BATCH]
                opt.zero_grad()
                logits = logits_net(X_t[idx])
                loss = loss_fn(-logits, y_t[idx])   # negated -> argmin convention
                loss.backward()
                opt.step()

        # Accuracy with the SAME rule verification uses: argmin of logits vs y[:,0].
        net.eval()
        with torch.no_grad():
            pred = logits_net(X_t).argmin(dim=1)
            acc = (pred == y_t).float().mean().item()

        # Export the full net (with Softmax tail) -> matches AC/GC/BM ONNX shape.
        onnx_path = models_dir / f'{name}.onnx'
        dummy = torch.zeros(1, input_dim, dtype=torch.float32)
        torch.onnx.export(
            net, dummy, str(onnx_path),
            input_names=['input'], output_names=['output'],
            dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
            opset_version=13,
        )
        print(f"{name}: hidden={hidden}  accuracy={acc:.4f}  -> {onnx_path.name}")


if __name__ == '__main__':
    main()
