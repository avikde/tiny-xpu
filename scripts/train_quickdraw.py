"""
Train a 3-layer MLP on a subset of Google QuickDraw and export to ONNX.

Downloads CLASSES numpy-bitmap files from the QuickDraw public dataset
(28×28 uint8 images, same layout as MNIST), trains a PyTorch MLP, exports
to ONNX, then applies dynamic int8 quantization for the TinyXPU EP.

Usage:
    python scripts/train_quickdraw.py [--epochs N] [--samples N]

Outputs (written to the repo root):
    quickdraw.onnx       float32 ONNX model
    quickdraw-int8.onnx  dynamically quantized (QInt8 weights)
"""

import argparse
import os
import sys
import urllib.request

import numpy as np
import onnx
import torch
import torch.nn as nn
import torch.optim as optim
from onnxruntime.quantization import QuantType, quantize_dynamic
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

CLASSES = [
    "circle",
    "square",
    "triangle",
    "star",
    "arrow",
    "line",
    "zigzag",
    "spiral",
    "cross",
    "lightning",
]

QUICKDRAW_URL = (
    "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{}.npy"
)


def download_class(name: str, cache_dir: str, samples_per_class: int) -> np.ndarray:
    """Download (or load from cache) one QuickDraw class, return first N samples."""
    path = os.path.join(cache_dir, f"{name}.npy")
    if not os.path.exists(path):
        url = QUICKDRAW_URL.format(name.replace(" ", "%20"))
        print(f"  Downloading {name} … ", end="", flush=True)
        urllib.request.urlretrieve(url, path)
        print("done")
    data = np.load(path, mmap_mode="r")
    return data[:samples_per_class].copy()


def load_dataset(cache_dir: str, samples_per_class: int):
    os.makedirs(cache_dir, exist_ok=True)
    X_list, y_list = [], []
    for label, name in enumerate(CLASSES):
        imgs = download_class(name, cache_dir, samples_per_class)  # (N, 784) uint8
        X_list.append(imgs)
        y_list.append(np.full(len(imgs), label, dtype=np.int64))
    X = np.concatenate(X_list).astype(np.float32) / 255.0  # normalise to [0,1]
    y = np.concatenate(y_list)
    return X, y


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class QuickDrawMLP(nn.Module):
    """Three-layer MLP: 784 → 512 → 256 → N_classes."""

    def __init__(self, n_classes: int = len(CLASSES)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        correct += (logits.argmax(1) == y_batch).sum().item()
        total += len(y_batch)
    return total_loss / total, correct / total


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            correct += (model(X_batch).argmax(1) == y_batch).sum().item()
            total += len(y_batch)
    return correct / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--samples",
        type=int,
        default=5000,
        help="samples per class (train+test combined)",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.join(script_dir, "..")
    cache_dir = os.path.join(repo_root, ".quickdraw_cache")

    print(f"Classes     : {CLASSES}")
    print(f"Samples/cls : {args.samples}  (80/20 train/test split)")
    print(f"Epochs      : {args.epochs}")
    print()

    # ---- data ------------------------------------------------------------------
    print("Loading data …")
    X, y = load_dataset(cache_dir, args.samples)
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    print(f"  Train: {len(X_train)}  Test: {len(X_test)}")

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=512)

    # ---- model -----------------------------------------------------------------
    device = torch.device("cpu")
    model = QuickDrawMLP(n_classes=len(CLASSES)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print("\nTraining …")
    for epoch in range(1, args.epochs + 1):
        loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        test_acc = evaluate(model, test_loader, device)
        print(
            f"  Epoch {epoch:3d}/{args.epochs}  loss={loss:.4f}"
            f"  train={train_acc:.3f}  test={test_acc:.3f}"
        )

    # ---- export to ONNX --------------------------------------------------------
    float_path = os.path.join(repo_root, "quickdraw.onnx")
    print(f"\nExporting float32 model → {float_path}")
    dummy = torch.zeros(1, 784, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        float_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=13,
    )
    print("  Done.")

    # ---- dynamic int8 quantization ---------------------------------------------
    int8_path = os.path.join(repo_root, "quickdraw-int8.onnx")
    print(f"Quantizing (dynamic QInt8) → {int8_path}")
    quantize_dynamic(float_path, int8_path, weight_type=QuantType.QInt8)
    print("  Done.")

    # Confirm the quantized model loads correctly
    m = onnx.load(int8_path)
    matmuls = [n for n in m.graph.node if n.op_type == "MatMulInteger"]
    relus = [n for n in m.graph.node if n.op_type == "Relu"]
    print(f"\nQuantized graph: {len(matmuls)} MatMulInteger, {len(relus)} Relu node(s)")
    print("Ready to run with TinyXPU EP via:  python scripts/run_quickdraw.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
