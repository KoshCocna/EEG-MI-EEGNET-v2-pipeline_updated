#!/usr/bin/env python3
"""Generate MI-friendly synthetic EEG epoch CSVs (2s @ 250Hz) for 6 classes.

Folder layout:
data/
  left/ right/ up/ down/ zoomIn/ zoomOut/
Each contains epoch_MI_<label>_0001.csv ... epoch_MI_<label>_0300.csv

Label meanings (MI-friendly):
- left:    left hand imagery (stronger ERD contralateral ~ C4/FC6)
- right:   right hand imagery (stronger ERD contralateral ~ C3/FC5)
- up:      both hands imagery (bilateral ERD)
- down:    feet imagery (midline ERD ~ CZ/FZ)
- zoomIn:  tongue/jaw imagery (more frontal/midline activity)
- zoomOut: rest (stronger mu/beta power, minimal ERD)

CSV format:
timestamp_sec, ch0..ch23
"""

import argparse
from pathlib import Path
import numpy as np

CHANNEL_NAMES = ["FP1","FP2","F3","F4","C3","C4","FC5","FC6","O1","O2","F7","F8","T7","T8","P7","P8","AFZ","CZ","FZ","PZ","FPZ","OZ","AF3","AF4"]
LABELS = ["left","right","up","down","zoomIn","zoomOut"]

def band_noise(n, sr, scale=1.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    x = rng.standard_normal(n)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, 1/sr)
    w = 1.0/np.maximum(freqs, 0.5)
    X *= w
    y = np.fft.irfft(X, n)
    y = y / (np.std(y) + 1e-8)
    return (y * scale).astype(np.float32)

def make_epoch(label, sr, n_samples, rng):
    t = (np.arange(n_samples, dtype=np.float32) / sr)

    mu   = np.sin(2*np.pi*10*t).astype(np.float32)
    beta = np.sin(2*np.pi*20*t).astype(np.float32)
    alpha_occ = np.sin(2*np.pi*11*t).astype(np.float32)

    idx = {name:i for i,name in enumerate(CHANNEL_NAMES)}
    motor_left  = [idx["C3"], idx["FC5"]]
    motor_right = [idx["C4"], idx["FC6"]]
    motor_mid   = [idx["CZ"], idx["FZ"]]
    frontal     = [idx["F3"], idx["F4"], idx["FZ"], idx["AFZ"], idx["FPZ"], idx["F7"], idx["F8"]]
    occip       = [idx["O1"], idx["O2"], idx["OZ"]]

    erd_mask = (t >= 1.0) & (t <= 3.0)
    env_strong = np.ones_like(t); env_strong[erd_mask] = 0.35
    env_mild   = np.ones_like(t); env_mild[erd_mask]   = 0.65

    X = np.zeros((24, n_samples), dtype=np.float32)
    for c in range(24):
        X[c] = band_noise(n_samples, sr, scale=8.0, rng=rng)

    mu_amp = np.full(24, 3.0, dtype=np.float32)
    beta_amp = np.full(24, 1.2, dtype=np.float32)
    occ_amp = np.zeros(24, dtype=np.float32); occ_amp[occip] = 2.0

    if label == "zoomOut":  # rest
        mu_amp[motor_left + motor_right + motor_mid] = 5.0
        beta_amp[motor_left + motor_right + motor_mid] = 1.5
        for c in range(24):
            X[c] += mu_amp[c]*mu + beta_amp[c]*beta
    else:
        if label == "left":
            mu_amp[motor_right] = 6.0
            mu_amp[motor_left]  = 4.0
            target = set(motor_right)
        elif label == "right":
            mu_amp[motor_left]  = 6.0
            mu_amp[motor_right] = 4.0
            target = set(motor_left)
        elif label == "up":
            mu_amp[motor_left + motor_right] = 6.0
            target = set(motor_left + motor_right)
        elif label == "down":
            mu_amp[motor_mid] = 6.0
            target = set(motor_mid)
        elif label == "zoomIn":
            mu_amp[frontal] = 4.5
            beta_amp[frontal] = 2.5
            target = set(frontal[:3] + motor_mid)
        else:
            target = set(motor_mid)

        for c in range(24):
            e = env_strong if c in target else env_mild
            X[c] += mu_amp[c]*mu*e + beta_amp[c]*beta

    for c in occip:
        X[c] += occ_amp[c]*alpha_occ

    X += rng.normal(0, 0.2, size=(24,1)).astype(np.float32)
    return t, X

def write_csv(path: Path, t: np.ndarray, X: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "timestamp_sec," + ",".join([f"ch{i}" for i in range(24)])
    M = np.concatenate([t.reshape(-1,1), X.T], axis=1)
    np.savetxt(path, M, delimiter=",", header=header, comments="", fmt="%.4f")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data", help="Output root folder (default: data)")
    ap.add_argument("--per_label", type=int, default=300, help="Epoch CSV count per label")
    ap.add_argument("--sr", type=int, default=250, help="Sampling rate")
    ap.add_argument("--epoch_sec", type=float, default=2.0, help="Epoch length (seconds)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    args = ap.parse_args()

    out_root = Path(args.out)
    n_samples = int(args.sr * args.epoch_sec)
    rng = np.random.default_rng(args.seed)

    total = 0
    for lab in LABELS:
        folder = out_root / lab
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(1, args.per_label + 1):
            f = folder / f"epoch_MI_{lab}_{i:04d}.csv"
            if f.exists() and not args.overwrite:
                continue
            t, X = make_epoch(lab, args.sr, n_samples, rng)
            write_csv(f, t, X)
            total += 1
    print(f"Done. Wrote {total} files under: {out_root.resolve()}")

if __name__ == "__main__":
    main()
