# Leyenda – Deliverable 1: Photo vs non-Photo

Binary image classification for TouNum: Photo (label 1) against Painting, Schematics, Sketch and Text (label 0).
The main deliverable is `deliverable1.ipynb`: self-contained, about 22 min on an NVIDIA GPU (22.1 min on an RTX 5060 Laptop in the last run).

## Repo structure

- `deliverable1.ipynb`: final notebook. It does not import `utils.py`, does not read `protocol/manifest.csv` or `LEYENDA_DATA_DIR`: it reads `data/Datasets`, saves the MobileNetV2 weights to `models/final_pretrained.keras` (not versioned) and recomputes inventory, duplicates, split and leakage check itself (seed 42).
- `00_data_prep.ipynb`: builds `protocol/manifest.csv`. Reads `LEYENDA_DATA_DIR` if set, otherwise `data/Datasets`.
- `01_mlp.ipynb` to `04_pretrained.ipynb`: development notebooks. They read `protocol/manifest.csv` and import `utils.py`; they read the data from `data/Datasets` only.
- `utils.py`: shared helpers (`load_images`, `class_weights`, `evaluate`, `plot_curves`).
- `protocol/manifest.csv`: one row per image (`drop_reason`, `split`).
- `reports/`: written by `deliverable1.ipynb`: `final_val_summary.csv`, `final_test_results.csv` and 7 PNG figures (MLP and CNN architecture, baseline curves, variant curves, pretrained curves, confusion matrix, error by origin); `reports/0*` (development) is not versioned.
- `requirements.txt`, `requirements-lock.txt` (exact versions of the final run; this file does not exist yet).

## Data

`data/Datasets/<origin>/` at the repo root, not versioned: 41,399 files in 5 folders (Painting, Photo, Schematics, Sketch, Text).
Dropped, never deleted: 696 exact duplicates (same md5), 401 images with a shortest side under 32 px (all Schematics). Kept: 40,302, split 70/15/15 stratified by origin (28,211 / 6,045 / 6,046), no md5 in two splits.
The md5 of `protocol/manifest.csv` is `fe4b4a4407369bdfdfc15700576a0515`.

## Setup

1. Verified stack: Python 3.12, TensorFlow 2.21 (stable), Keras 3.15.1. Final-run stack: TensorFlow 2.22.0-rc0, Keras 3.16.0.dev (`requirements-lock.txt` is not created yet).
2. `sudo apt install graphviz`, then `pip install -r requirements.txt pydot` (needed by `plot_model`).
3. On the 3.12 stack, set `LD_LIBRARY_PATH` to the pip `nvidia/cusolver/lib` folder (command below), otherwise no GPU is found.
4. Internet is needed once to download the ImageNet weights of MobileNetV2.
5. Every TensorFlow notebook stops with a message if no GPU is found.

## How to run

Open the notebooks from the repo root (paths are relative). Run `00_data_prep.ipynb`, then optionally `01` to `04` (`02_cnn.ipynb` reads `reports/test_results.csv` written by `01`), then `deliverable1.ipynb`.
Seed 42 everywhere. Two executions give slightly different numbers (GPU non-determinism): see the Limits cell of `deliverable1.ipynb`.

## Branches

Everyone works on their own branch (e.g. `maxime/deliverable-1`); never push to `main`.

## Environment

Maxime's stack (WSL2 Ubuntu 26.04, RTX 5060 Laptop GPU 8 GB, driver 616.92): Python 3.14.4, TensorFlow 2.22.0rc0, keras-nightly 3.16.0.dev2026100504, CUDA 12.9 libraries from pip (`nvidia-*-cu12`). The notebooks print the exact library versions they ran with (first code cell).

```bash
python3 -m venv uv-bootstrap && uv-bootstrap/bin/pip install uv
uv-bootstrap/bin/uv venv --python 3.12 .venv312
uv-bootstrap/bin/uv pip install --python .venv312/bin/python "tensorflow[and-cuda]" pandas scikit-learn pillow matplotlib ipykernel
# TF 2.21.0 does not search nvidia/cusolver/lib: without this line, no GPU is found.
export LD_LIBRARY_PATH=$PWD/.venv312/lib/python3.12/site-packages/nvidia/cusolver/lib
```

GPU check (both stacks warn that kernels for compute capability 12.0a are JIT-compiled from PTX, so the first epoch is slower):

```python
import tensorflow as tf
print(tf.__version__, tf.config.list_physical_devices("GPU"))   # expect one GPU
```

A `.keras` model saved in one stack loads in the other, with identical predictions (tested both ways).
