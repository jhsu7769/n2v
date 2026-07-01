"""Train + export insurance network for FairN2V."""
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


SEED = 0
EPOCHS = 60
BATCH = 256
LR = 1e-3

fairn2v_root = Path(__file__).resolve().parent.parent
data_path = fairn2v_root / "data" / "medcost_data.npz"
models_dir = fairn2v_root / "models"

ARCHITECTURES = {
    "MC-1": [16, 8],
    "MC-2": [50],
    "MC-3": [100, 100]
}

def normalize_data() -> tuple[np.ndarray, np.ndarray]:
    data = np.load(data_path)
    X = data["X"].astype(np.float32)
    y0 = data["y"][:, 0].astype(np.float32)

    min_v = X.min(axis=0)
    max_v = X.max(axis=0)
    var = max_v - min_v > 0
    min_v[~var] = 0.0
    max_v[~var] = 1.0
    Xn = (X - min_v) / (max_v - min_v)
    return Xn.astype(np.float32), y0


def build_net(hidden: list[int], input_dim: int):
    """Linear ReLU -> Linear(prev, 1). No softmax."""
    layers = []
    prev = input_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers += [nn.Linear(prev, 1)]
    return nn.Sequential(*layers)


def main():
    Xn, y = normalize_data()
    input_dim = Xn.shape[1]

    y_mean = float(y.mean())
    y_std = float(y.std())
    z = (y - y_mean) / y_std  # Standardize

    X_t = torch.from_numpy(Xn)  # (n,9)
    z_t = torch.from_numpy(z).reshape(-1, 1)  # (n,1)

    for name, hidden in ARCHITECTURES.items():
        torch.manual_seed(SEED)
        net = build_net(hidden, input_dim)
        opt = torch.optim.Adam(net.parameters(), lr=LR)
        loss_fn = nn.MSELoss()

        net.train()
        for epoch in range(EPOCHS):
            perm = torch.randperm(X_t.shape[0])  # Shuffle each epoch
            for s in range(0, X_t.shape[0], BATCH):
                idx = perm[s:s + BATCH]
                opt.zero_grad()
                pred = net(X_t[idx])
                loss = loss_fn(pred, z_t[idx])
                loss.backward()
                opt.step()

        descale = nn.Linear(1,1)
        with torch.no_grad():
            descale.weight.copy_(torch.tensor([[y_std]]))  # (1,1)
            descale.bias.copy_(torch.tensor([y_mean]))  # (1,)
        for p in descale.parameters():
            p.requires_grad_(False)

        export_net = nn.Sequential(net, descale)
        export_net.eval()
        with torch.no_grad():
            pred_dollars = export_net(X_t).squeeze(1)  # (n,)
            mae = (pred_dollars - torch.from_numpy(y)).abs().mean().item()

        models_dir.mkdir(exist_ok=True)
        onnx_path = models_dir / f"{name}.onnx"
        dummy = torch.zeros(1, input_dim, dtype=torch.float32)
        torch.onnx.export(
                export_net, dummy, str(onnx_path),
                input_names=["input"], output_names=["output"],
                dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                opset_version=13,
                dynamo=False,
        )
        print(f"{name}: hidden={hidden}  MAE=${mae:,.0f}  -> {onnx_path.name}")


if __name__ == "__main__":
    main()

