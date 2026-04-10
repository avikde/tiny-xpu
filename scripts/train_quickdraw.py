"""
Train a 3-layer integer MLP on a subset of Google QuickDraw and export to ONNX.

The network is co-designed with the 64×64 systolic array so that no dimension
requires tiling:

    Input  : 28×28 bitmap downsampled to 8×8 → 64 features
    FC1    : 64 → 64  + ReLU
    FC2    : 64 → 32  + ReLU
    FC3    : 32 → 10  (logits)

Static post-training quantization (PTQ) is applied using torch.ao.quantization.
The exported ONNX model uses QLinearMatMul nodes so the TinyXPU EP can drive the
systolic array + requantization stage as a single hardware operation with no
float32 between layers.

Usage:
    python scripts/train_quickdraw.py [--epochs N] [--samples N]

Outputs (repo root):
    quickdraw.onnx       float32 model
    quickdraw-int8.onnx  statically quantized (QLinearMatMul)
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
from torch.ao.quantization import (
    get_default_qconfig,
    prepare,
    convert,
    fuse_modules,
    QuantStub,
    DeQuantStub,
)
from torch.utils.data import DataLoader, TensorDataset
from onnxruntime.quantization import QuantType, quantize_dynamic

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

CLASSES = [
    "circle", "square", "triangle", "star", "arrow",
    "line", "zigzag", "spiral", "cross", "lightning",
]

QUICKDRAW_URL = (
    "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{}.npy"
)

INPUT_DIM = 64   # 8×8 downsampled


def resize_8x8(imgs_flat: np.ndarray) -> np.ndarray:
    """Resize N×784 (28×28) uint8 bitmaps to N×64 (8×8) via area averaging."""
    n = len(imgs_flat)
    imgs = imgs_flat.reshape(n, 28, 28).astype(np.float32)
    # 28×28 → 8×8: each 3.5×3.5 block → use simple strided slicing for speed
    # We use a proper resize via averaging over 4×4 non-overlapping blocks on a
    # 32×32 intermediate (pad 2 on each side, then 4×4 average pool).
    pad = np.pad(imgs, ((0, 0), (2, 2), (2, 2)), mode='edge')  # 32×32
    blocks = pad.reshape(n, 8, 4, 8, 4).mean(axis=(2, 4))      # 8×8
    return blocks.reshape(n, INPUT_DIM).astype(np.float32) / 255.0


def download_class(name: str, cache_dir: str, samples: int) -> np.ndarray:
    path = os.path.join(cache_dir, f"{name}.npy")
    if not os.path.exists(path):
        url = QUICKDRAW_URL.format(name.replace(" ", "%20"))
        print(f"  Downloading {name} … ", end="", flush=True)
        urllib.request.urlretrieve(url, path)
        print("done")
    data = np.load(path, mmap_mode="r")
    return data[:samples].copy()


def load_dataset(cache_dir: str, samples_per_class: int):
    os.makedirs(cache_dir, exist_ok=True)
    X_list, y_list = [], []
    for label, name in enumerate(CLASSES):
        imgs = download_class(name, cache_dir, samples_per_class)
        X_list.append(resize_8x8(imgs))
        y_list.append(np.full(len(imgs), label, dtype=np.int64))
    return np.concatenate(X_list), np.concatenate(y_list)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class QuickDrawMLP(nn.Module):
    """64 → 64 → 32 → 10  (fits exactly in a 64×64 systolic array, no tiling)."""

    def __init__(self):
        super().__init__()
        self.quant   = QuantStub()
        self.fc1     = nn.Linear(INPUT_DIM, 64)
        self.relu1   = nn.ReLU()
        self.fc2     = nn.Linear(64, 32)
        self.relu2   = nn.ReLU()
        self.fc3     = nn.Linear(32, len(CLASSES))
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        x = self.dequant(x)
        return x


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_b), y_b)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            total_loss += loss.item() * len(y_b)
            correct += (model(X_b).argmax(1) == y_b).sum().item()
            total += len(y_b)
    return total_loss / total, correct / total


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_b, y_b in loader:
            correct += (model(X_b.to(device)).argmax(1) == y_b.to(device)).sum().item()
            total += len(y_b)
    return correct / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",  type=int, default=15)
    parser.add_argument("--samples", type=int, default=5000,
                        help="samples per class (train+test combined)")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root  = os.path.join(script_dir, "..")
    cache_dir  = os.path.join(repo_root, ".quickdraw_cache")

    print(f"Classes     : {CLASSES}")
    print(f"Input dim   : {INPUT_DIM}  (8×8 downsampled from 28×28)")
    print(f"Architecture: {INPUT_DIM} → 64 → 32 → {len(CLASSES)}")
    print(f"Samples/cls : {args.samples}  (80/20 split)")
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
    X_test,  y_test  = X[split:], y[split:]
    print(f"  Train: {len(X_train)}  Test: {len(X_test)}\n")

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds  = TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test))
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=512)

    # ---- float training --------------------------------------------------------
    device    = torch.device("cpu")
    model     = QuickDrawMLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print("Training float model …")
    for epoch in range(1, args.epochs + 1):
        loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        te_acc = evaluate(model, test_loader, device)
        print(f"  Epoch {epoch:3d}/{args.epochs}  loss={loss:.4f}"
              f"  train={tr_acc:.3f}  test={te_acc:.3f}")

    # ---- export float ONNX -----------------------------------------------------
    float_path = os.path.join(repo_root, "quickdraw.onnx")
    print(f"\nExporting float32 model → {float_path}")
    # Export the plain (non-quantized) model; remove QuantStub/DeQuantStub wrappers
    # by passing the inner forward logic directly.
    class _FloatOnly(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.net = nn.Sequential(m.fc1, m.relu1, m.fc2, m.relu2, m.fc3)
        def forward(self, x):
            return self.net(x)

    float_model = _FloatOnly(model).eval()
    dummy = torch.zeros(1, INPUT_DIM)
    torch.onnx.export(
        float_model, dummy, float_path,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=13,
    )
    print("  Done.")

    # ---- static post-training quantization -------------------------------------
    print("\nApplying static PTQ …")
    model.eval()

    # Fuse Linear+ReLU pairs so they quantize jointly
    model_fused = QuickDrawMLP()
    model_fused.load_state_dict(model.state_dict())
    model_fused.eval()
    fuse_modules(model_fused, [["fc1", "relu1"], ["fc2", "relu2"]], inplace=True)

    # Configure per-tensor symmetric int8 quantization (matches hardware)
    model_fused.qconfig = get_default_qconfig("x86")
    prepare(model_fused, inplace=True)

    # Calibration: run training subset through the prepared model
    print("  Calibrating on training set …")
    model_fused.eval()
    with torch.no_grad():
        for X_b, _ in train_loader:
            model_fused(X_b)

    # Convert to quantized model (inserts QLinearMatMul / quantized ops)
    model_quant = convert(model_fused, inplace=False)
    model_quant.eval()

    quant_acc = evaluate(model_quant, test_loader, device)
    print(f"  Quantized test accuracy: {quant_acc:.3f}")

    # ---- export quantized ONNX -------------------------------------------------
    int8_path = os.path.join(repo_root, "quickdraw-int8.onnx")
    print(f"\nExporting quantized model → {int8_path}")
    # torch.ao.quantization's convert() produces a model with QuantizedLinear ops.
    # torch.onnx.export maps these to QLinearMatMul in the ONNX graph.
    dummy_quant = torch.zeros(1, INPUT_DIM)
    torch.onnx.export(
        model_quant, dummy_quant, int8_path,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=13,
    )
    print("  Done.")

    # ---- inspect exported graph ------------------------------------------------
    m = onnx.load(int8_path)
    qlmm  = [n for n in m.graph.node if n.op_type == "QLinearMatMul"]
    relus = [n for n in m.graph.node if n.op_type == "Relu"]
    print(f"\nQuantized graph: {len(qlmm)} QLinearMatMul, {len(relus)} Relu node(s)")
    print("Run with:  python scripts/run_quickdraw.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
