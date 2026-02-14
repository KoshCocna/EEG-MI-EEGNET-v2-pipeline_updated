#!/usr/bin/env python3
"""
Watch a folder for newly created EEG epoch CSVs (2s window), run EEGNet v2 inference once per file,
and send the predicted command to Unity via UDP.

Expected CSV format:
timestamp_sec, ch0..ch23
- 2s @ 250Hz -> 500 rows recommended (but script will pad/crop)

How it works:
- Poll folder for *.csv
- Sort by file creation time (or filename if you prefer)
- For each new file:
    - wait until file size is stable (avoid partial write)
    - load -> preprocess -> infer -> smoothing(optional) -> send UDP
    - mark file as processed (no duplicates)

Run:
  python realtime_infer_to_unity_udp_2s_watchfolder.py --in_dir incoming_epochs --bandpass
"""

import argparse
import json
import time
import socket
from pathlib import Path
from collections import deque, Counter

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F


LABELS = ["left", "right", "up", "down", "zoomIn", "zoomOut"]
N_CH = 24


# ----------------------------
# Preprocess
# ----------------------------
def butter_bandpass_filtfilt(x: np.ndarray, sr: int, low: float, high: float, order: int = 4) -> np.ndarray:
    try:
        from scipy.signal import butter, filtfilt
    except Exception:
        return x
    nyq = 0.5 * sr
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    y = np.zeros_like(x)
    for c in range(x.shape[0]):
        y[c] = filtfilt(b, a, x[c]).astype(np.float32)
    return y

def per_channel_zscore(x: np.ndarray) -> np.ndarray:
    m = x.mean(axis=1, keepdims=True)
    s = x.std(axis=1, keepdims=True) + 1e-8
    return ((x - m) / s).astype(np.float32)

def load_epoch_csv(path: Path, n_samples: int) -> np.ndarray:
    df = pd.read_csv(path)
    cols = [f"ch{i}" for i in range(N_CH)]
    X = df[cols].to_numpy(dtype=np.float32).T  # (ch, time)

    # pad/crop
    if X.shape[1] != n_samples:
        if X.shape[1] < n_samples:
            pad = n_samples - X.shape[1]
            X = np.pad(X, ((0, 0), (0, pad)), mode="edge")
        else:
            X = X[:, :n_samples]
    return X


# ----------------------------
# EEGNet v2
# ----------------------------
class EEGNetV2(nn.Module):
    def __init__(
        self,
        n_ch: int,
        n_samples: int,
        n_classes: int,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kern_length: int = 64,
        sep_kern: int = 16,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(1, F1, kernel_size=(1, kern_length), padding=(0, kern_length // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        self.depthwise = nn.Conv2d(F1, F1 * D, kernel_size=(n_ch, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.act = nn.ELU()
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.drop1 = nn.Dropout(dropout)

        self.sep_depth = nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, sep_kern), padding=(0, sep_kern // 2), groups=F1 * D, bias=False)
        self.sep_point = nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.drop2 = nn.Dropout(dropout)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_ch, n_samples)
            feat_dim = self.forward_features(dummy).shape[1]
        self.classifier = nn.Linear(feat_dim, n_classes)

    def forward_features(self, x):
        x = self.bn1(self.conv1(x))
        x = self.drop1(self.pool1(self.act(self.bn2(self.depthwise(x)))))
        x = self.drop2(self.pool2(self.act(self.bn3(self.sep_point(self.sep_depth(x))))))
        return torch.flatten(x, start_dim=1)

    def forward(self, x):
        return self.classifier(self.forward_features(x))


# ----------------------------
# Optional smoothing (N-of-M + threshold + cooldown)
# ----------------------------
class CommandSmoother:
    def __init__(self, labels, M=5, N=4, threshold=0.70, cooldown=0.0):
        self.labels = labels
        self.q = deque(maxlen=M)
        self.N = N
        self.threshold = threshold
        self.cooldown = cooldown
        self.last_sent_time = 0.0

    def update(self, pred_idx: int, max_prob: float):
        now = time.time()
        raw_cmd = self.labels[pred_idx]
        self.q.append(pred_idx)

        if max_prob < self.threshold:
            return None, raw_cmd

        if self.cooldown > 0 and (now - self.last_sent_time) < self.cooldown:
            return None, raw_cmd

        cnt = Counter(self.q)
        best_idx, best_count = cnt.most_common(1)[0]
        if best_count >= self.N:
            cmd = self.labels[best_idx]
            self.last_sent_time = now
            return cmd, raw_cmd

        return None, raw_cmd


# ----------------------------
# File stability check (avoid reading half-written CSV)
# ----------------------------
def wait_until_file_stable(path: Path, checks: int = 4, interval: float = 0.2) -> bool:
    """
    Returns True if size stays the same for `checks` times.
    """
    last = -1
    same = 0
    for _ in range(checks):
        try:
            sz = path.stat().st_size
        except FileNotFoundError:
            return False
        if sz == last and sz > 0:
            same += 1
        else:
            same = 0
        last = sz
        time.sleep(interval)
    return same >= (checks - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, default="incoming_epochs", help="Folder where CSV files appear")
    ap.add_argument("--model_path", type=str, default="result/best_model.pt", help="Trained model checkpoint")
    ap.add_argument("--host", type=str, default="127.0.0.1", help="Unity UDP host")
    ap.add_argument("--port", type=int, default=5005, help="Unity UDP port")
    ap.add_argument("--poll", type=float, default=0.2, help="Folder polling interval (sec)")
    ap.add_argument("--sr", type=int, default=250)
    ap.add_argument("--epoch_sec", type=float, default=2.0)
    ap.add_argument("--bandpass", action="store_true")
    ap.add_argument("--no_zscore", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.70)
    ap.add_argument("--use_smoother", action="store_true", help="Use N-of-M voting (good if multiple files/sec)")
    ap.add_argument("--M", type=int, default=5)
    ap.add_argument("--N", type=int, default=4)
    ap.add_argument("--cooldown", type=float, default=0.0, help="Cooldown after sending a command (sec)")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    in_dir.mkdir(parents=True, exist_ok=True)

    n_samples = int(args.sr * args.epoch_sec)

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path.resolve()} (train first)")

    # UDP
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    target = (args.host, args.port)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EEGNetV2(n_ch=N_CH, n_samples=n_samples, n_classes=len(LABELS)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    smoother = CommandSmoother(LABELS, M=args.M, N=args.N, threshold=args.threshold, cooldown=args.cooldown)

    processed = set()  # stores processed file paths (string)

    print(f"[INFO] Watching folder: {in_dir.resolve()}")
    print(f"[INFO] Expect window: {args.epoch_sec}s @ {args.sr}Hz -> {n_samples} samples")
    print(f"[INFO] UDP -> Unity: {target[0]}:{target[1]}")
    print("[INFO] Drop CSVs into the folder. Ctrl+C to stop.\n")

    try:
        while True:
            # list csv files
            files = sorted(
                in_dir.glob("*.csv"),
                key=lambda p: p.stat().st_ctime  # creation time order
            )

            new_files = [p for p in files if str(p.resolve()) not in processed]

            for p in new_files:
                # wait until file write completed
                ok = wait_until_file_stable(p)
                if not ok:
                    continue  # skip for now; will be retried next loop

                # load & preprocess
                X = load_epoch_csv(p, n_samples=n_samples)
                if args.bandpass:
                    X = butter_bandpass_filtfilt(X, args.sr, 8.0, 30.0, order=4)
                if not args.no_zscore:
                    X = per_channel_zscore(X)

                Xt = torch.from_numpy(X).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,24,500)

                with torch.no_grad():
                    probs = F.softmax(model(Xt), dim=1).squeeze(0).cpu().numpy()
                pred_idx = int(np.argmax(probs))
                max_prob = float(np.max(probs))
                raw_cmd = LABELS[pred_idx]

                # decision
                if args.use_smoother:
                    cmd, _ = smoother.update(pred_idx, max_prob)
                    cmd_to_send = cmd if cmd is not None else "none"
                else:
                    cmd_to_send = raw_cmd if max_prob >= args.threshold else "none"

                msg = {
                    "cmd": cmd_to_send,
                    "raw_pred": raw_cmd,
                    "score": max_prob,
                    "probs": [float(x) for x in probs],
                    "timestamp": time.time(),
                    "file": p.name,
                }

                sock.sendto(json.dumps(msg).encode("utf-8"), target)
                print(f"[PRED] {p.name} -> cmd={cmd_to_send:7s} (raw={raw_cmd:7s}, score={max_prob:.3f})")

                # mark processed (prevents duplicates)
                processed.add(str(p.resolve()))

            time.sleep(args.poll)

    except KeyboardInterrupt:
        print("\n[INFO] Stopped.")
    finally:
        sock.close()


if __name__ == "__main__":
    main()

