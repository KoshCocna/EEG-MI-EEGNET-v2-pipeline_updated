# MI-EEGNet-v2 Digital Twin BCI System

Motor Imagery 기반 6-Class EEG 분류 및 Unity 디지털트윈 제어 시스템

---

## 1. Project Overview

본 프로젝트는 Motor Imagery(MI) EEG 신호를 이용하여  
6개의 명령을 분류하고, 해당 명령을 Unity 기반 디지털트윈 환경에서 실시간으로 실행하는 시스템입니다.

Pipeline:

EEG Data → EEGNet v2 → Command Prediction → UDP → Unity Digital Twin

---

## 2. Command Classes

| Label     | Motor Imagery Task |
|-----------|-------------------|
| left      | Left hand imagery |
| right     | Right hand imagery |
| up        | Both hands imagery |
| down      | Feet imagery |
| zoomIn    | Tongue / jaw imagery |
| zoomOut   | Rest |

---

## 3. Hardware Configuration

- Device: Laxtha QEEG-64FX
- Channels: 24
- Sampling Rate: 250 Hz
- Epoch Length: 2 seconds
- Samples per epoch: 500
- Frequency Band: 8–30 Hz (μ / β rhythm)

---

## 4. Channel Order (ch0 → ch23)

FP1, FP2, F3, F4,
C3, C4, FC5, FC6,
O1, O2, F7, F8,
T7, T8, P7, P8,
AFZ, CZ, FZ, PZ,
FPZ, OZ, AF3, AF4


---

## 5. Project Structure

mi_eegnet_v2_pipeline/
│
├── train_eegnet_mi_2s.py
├── realtime_infer_to_unity_udp_2s_watchfolder.py
│
├── data/ # training dataset (offline)
│
├── incoming_epochs/ # realtime input folder (online)
│
├── result/
│ ├── best_model.pt
│ ├── confusion_matrix.png
│ ├── learning_curves.png
│ └── metrics.json
│
└── README.md


---

## 6. Environment Setup

Create conda environment:

conda create -n eeg-dl python=3.10
conda activate eeg-dl
conda install numpy=1.26 -c conda-forge
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install scipy pandas matplotlib scikit-learn


---

## 7. Training Procedure (Offline Stage)

### Step 1 – Prepare dataset

Place labeled epoch CSV files inside:

data/
├── left/
├── right/
├── up/
├── down/
├── zoomIn/
└── zoomOut/


Each CSV format:

timestamp_sec, ch0, ch1, ..., ch23


(500 rows for 2s @ 250Hz)

---

### Step 2 – Train EEGNet v2

python train_eegnet_mi_2s.py


Output:

result/best_model.pt
result/confusion_matrix.png
result/learning_curves.png
result/metrics.json


---

## 8. Realtime Digital Twin Control (Online Stage)

### Step 1 – Launch Unity

- Add UDP Receiver script
- Listen on port 5005
- Assign controlled object (cube or end-effector)

---

### Step 2 – Prepare Realtime Input Folder

Ensure folder exists:

incoming_epochs/


When subject clicks a button,
the system must save a CSV file into this folder:

Example:

incoming_epochs/epoch_0001.csv
incoming_epochs/epoch_0002.csv
...


Each file contains 2 seconds of EEG recorded before click.

---

### Step 3 – Run Realtime Inference

python realtime_infer_to_unity_udp_2s_watchfolder.py
--in_dir incoming_epochs
--bandpass


Behavior:

- Watches folder continuously
- Detects new CSV files
- Ensures file writing is complete
- Runs inference once per file
- Sends JSON command via UDP
- Prevents duplicate processing

---

## 9. UDP Message Format

Example message sent to Unity:

{
"cmd": "left",
"raw_pred": "left",
"score": 0.83,
"probs": [...],
"timestamp": 1739451201.123,
"file": "epoch_0003.csv"
}


---

## 10. Decision Logic

Realtime classification includes:

- Softmax probability threshold
- Optional N-of-M voting
- Optional cooldown control
- Reject low-confidence predictions

This prevents jitter and unstable robot motion.

---

## 11. System Workflow Summary

Offline:

EEG Recording → Epoch Segmentation → Training → Model Save


Online:

Button Click → 2s EEG CSV → Folder Watcher → EEGNet Inference → UDP → Unity Motion


---

## 12. Research Notes

- Subject-specific training recommended
- Re-calibration required each session
- 2s window improves response speed
- 4s window may improve classification accuracy
- Future: real-time LSL streaming instead of file-based trigger

---

## 13. Author

Kanye Kim  
BCI · EEG Signal Processing · Deep Learning · Digital Twin Control

---

## 14. License

Research / Educational Use
