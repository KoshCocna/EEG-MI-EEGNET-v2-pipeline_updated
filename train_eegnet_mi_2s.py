#!/usr/bin/env python3
"""Train EEGNet v2 on MI-friendly epoch CSVs and save results to result/.

Expected dataset layout:
data/
  left/ right/ up/ down/ zoomIn/ zoomOut/
Each CSV: timestamp_sec, ch0..ch23 (2s @ 250Hz => 500 rows (default))

Outputs (auto-created):
result/
  confusion_matrix.png
  learning_curves.png
  best_model.pt
  metrics.json

Quick start:
  python generate_fake_mi_epochs_300.py --out data --per_label 300
  python train_eegnet_mi.py
"""

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

LABELS = ["left","right","up","down","zoomIn","zoomOut"]
SR = 250
EPOCH_SEC = 2.0
N_SAMPLES = int(SR * EPOCH_SEC)
N_CH = 24

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def butter_bandpass_filtfilt(x: np.ndarray, sr: int, low: float, high: float, order: int = 4) -> np.ndarray:
    """Optional bandpass for MI (8–30Hz). If SciPy isn't available, returns input."""
    try:
        from scipy.signal import butter, filtfilt
    except Exception:
        return x
    nyq = 0.5 * sr
    b, a = butter(order, [low/nyq, high/nyq], btype="band")
    y = np.zeros_like(x)
    for c in range(x.shape[0]):
        y[c] = filtfilt(b, a, x[c]).astype(np.float32)
    return y

def per_channel_zscore(x: np.ndarray) -> np.ndarray:
    m = x.mean(axis=1, keepdims=True)
    s = x.std(axis=1, keepdims=True) + 1e-8
    return ((x - m) / s).astype(np.float32)

def load_one_csv(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    cols = [f"ch{i}" for i in range(N_CH)]
    X = df[cols].to_numpy(dtype=np.float32).T  # (ch, time)
    if X.shape[1] != N_SAMPLES:
        if X.shape[1] < N_SAMPLES:
            pad = N_SAMPLES - X.shape[1]
            X = np.pad(X, ((0,0),(0,pad)), mode="edge")
        else:
            X = X[:, :N_SAMPLES]
    return X

def build_index(data_root: Path):
    files, y = [], []
    for li, lab in enumerate(LABELS):
        folder = data_root / lab
        if not folder.exists():
            continue
        for p in sorted(folder.glob("*.csv")):
            files.append(p)
            y.append(li)
    return files, np.array(y, dtype=np.int64)

class EEGCSVSet(Dataset):
    def __init__(self, files, y, bandpass=True, zscore=True):
        self.files = files
        self.y = y
        self.bandpass = bandpass
        self.zscore = zscore

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        X = load_one_csv(self.files[idx])
        if self.bandpass:
            X = butter_bandpass_filtfilt(X, SR, 8.0, 30.0, order=4)
        if self.zscore:
            X = per_channel_zscore(X)
        Xt = torch.from_numpy(X).unsqueeze(0)  # (1, ch, time)
        yt = torch.tensor(self.y[idx], dtype=torch.long)
        return Xt, yt

class EEGNetV2(nn.Module):
    """EEGNet v2-ish (PyTorch) for (B, 1, C, T) input."""
    def __init__(self, n_ch=24, n_samples=500, n_classes=6, F1=8, D=2, F2=16, kern_length=64, sep_kern=16, dropout=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, F1, kernel_size=(1, kern_length), padding=(0, kern_length//2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        self.depthwise = nn.Conv2d(F1, F1*D, kernel_size=(n_ch, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1*D)
        self.act = nn.ELU()
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.drop1 = nn.Dropout(dropout)

        self.sep_depth = nn.Conv2d(F1*D, F1*D, kernel_size=(1, sep_kern), padding=(0, sep_kern//2), groups=F1*D, bias=False)
        self.sep_point = nn.Conv2d(F1*D, F2, kernel_size=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.drop2 = nn.Dropout(dropout)

        # infer feature dim
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_ch, n_samples)
            feat = self.forward_features(dummy)
            feat_dim = feat.shape[1]
        self.classifier = nn.Linear(feat_dim, n_classes)

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)

        x = self.depthwise(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.sep_depth(x)
        x = self.sep_point(x)
        x = self.bn3(x)
        x = self.act(x)
        x = self.pool2(x)
        x = self.drop2(x)

        return torch.flatten(x, start_dim=1)

    def forward(self, x):
        return self.classifier(self.forward_features(x))

def train_one_epoch(model, loader, opt, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(X)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss/total, correct/total

@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    ys, ps = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = F.cross_entropy(logits, y)
        total_loss += loss.item() * y.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        ys.append(y.cpu().numpy())
        ps.append(pred.cpu().numpy())
    return total_loss/total, correct/total, np.concatenate(ys), np.concatenate(ps)

def plot_learning_curves(hist, out_path: Path):
    epochs = np.arange(1, len(hist["train_loss"])+1)
    plt.figure()
    plt.plot(epochs, hist["train_loss"], label="train_loss")
    plt.plot(epochs, hist["val_loss"], label="val_loss")
    plt.plot(epochs, hist["train_acc"], label="train_acc")
    plt.plot(epochs, hist["val_acc"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_confusion(cm, labels, out_path: Path, title="Confusion Matrix"):
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick = np.arange(len(labels))
    plt.xticks(tick, labels, rotation=45, ha="right")
    plt.yticks(tick, labels)

    thresh = cm.max()/2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{cm[i,j]:d}", ha="center", va="center",
                     color="white" if cm[i,j] > thresh else "black")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data")
    ap.add_argument("--result_dir", type=str, default="result")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sr", type=int, default=250, help="Sampling rate (Hz)")
    ap.add_argument("--epoch_sec", type=float, default=2.0, help="Epoch length (seconds)")
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--test_size", type=float, default=0.15)
    ap.add_argument("--val_size", type=float, default=0.15)
    ap.add_argument("--no_bandpass", action="store_true")
    ap.add_argument("--no_zscore", action="store_true")
    args = ap.parse_args()

    # Update global sampling config (must match your data)
    global SR, EPOCH_SEC, N_SAMPLES
    SR = int(args.sr)
    EPOCH_SEC = float(args.epoch_sec)
    N_SAMPLES = int(SR * EPOCH_SEC)

    set_seed(args.seed)

    data_root = Path(args.data_root)
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    files, y = build_index(data_root)
    if len(files) == 0:
        raise SystemExit(f"No CSV files found under: {data_root.resolve()}")

    idx_all = np.arange(len(files))
    tr_idx, te_idx = train_test_split(idx_all, test_size=args.test_size, random_state=args.seed, stratify=y)
    y_tr = y[tr_idx]
    val_rel = args.val_size / (1.0 - args.test_size)
    tr_idx, va_idx = train_test_split(tr_idx, test_size=val_rel, random_state=args.seed, stratify=y_tr)

    tr_files = [files[i] for i in tr_idx]
    va_files = [files[i] for i in va_idx]
    te_files = [files[i] for i in te_idx]

    tr_y = y[tr_idx]
    va_y = y[va_idx]
    te_y = y[te_idx]

    bandpass = not args.no_bandpass
    zscore = not args.no_zscore

    tr_set = EEGCSVSet(tr_files, tr_y, bandpass=bandpass, zscore=zscore)
    va_set = EEGCSVSet(va_files, va_y, bandpass=bandpass, zscore=zscore)
    te_set = EEGCSVSet(te_files, te_y, bandpass=bandpass, zscore=zscore)

    tr_loader = DataLoader(tr_set, batch_size=args.batch, shuffle=True, num_workers=0)
    va_loader = DataLoader(va_set, batch_size=args.batch, shuffle=False, num_workers=0)
    te_loader = DataLoader(te_set, batch_size=args.batch, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EEGNetV2(n_ch=N_CH, n_samples=N_SAMPLES, n_classes=len(LABELS), dropout=0.5).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val = float("inf")
    best_path = result_dir / "best_model.pt"
    wait = 0

    for ep in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, tr_loader, opt, device)
        va_loss, va_acc, _, _ = eval_model(model, va_loader, device)

        hist["train_loss"].append(float(tr_loss))
        hist["val_loss"].append(float(va_loss))
        hist["train_acc"].append(float(tr_acc))
        hist["val_acc"].append(float(va_acc))

        print(f"Epoch {ep:03d}/{args.epochs} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | val loss {va_loss:.4f} acc {va_acc:.3f}")

        if va_loss < best_val - 1e-4:
            best_val = va_loss
            wait = 0
            torch.save(model.state_dict(), best_path)
        else:
            wait += 1
            if wait >= args.patience:
                print(f"Early stopping at epoch {ep} (best val loss {best_val:.4f})")
                break

    model.load_state_dict(torch.load(best_path, map_location=device))
    te_loss, te_acc, y_true, y_pred = eval_model(model, te_loader, device)

    print("\nTest results")
    print(f"  loss: {te_loss:.4f}")
    print(f"  acc : {te_acc:.3f}\n")
    print(classification_report(y_true, y_pred, target_names=LABELS, digits=3))

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion(cm, LABELS, result_dir / "confusion_matrix.png")
    plot_learning_curves(hist, result_dir / "learning_curves.png")

    metrics = {
        "labels": LABELS,
        "test_loss": float(te_loss),
        "test_acc": float(te_acc),
        "confusion_matrix": cm.tolist(),
        "train_size": int(len(tr_set)),
        "val_size": int(len(va_set)),
        "test_size": int(len(te_set)),
        "device": str(device),
        "bandpass_8_30": bool(bandpass),
        "zscore": bool(zscore),
        "seed": int(args.seed),
    }
    (result_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nSaved:")
    print(f"  {(result_dir/'confusion_matrix.png').resolve()}")
    print(f"  {(result_dir/'learning_curves.png').resolve()}")
    print(f"  {best_path.resolve()}")
    print(f"  {(result_dir/'metrics.json').resolve()}")

if __name__ == "__main__":
    main()
