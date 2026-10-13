# Leyenda – Deliverable 1: Photo vs non-Photo

Binary image classification for TouNum: Photo (label 1) against Painting, Schematics, Sketch and Text (label 0).
The main deliverable is `deliverable1.ipynb`: self-contained, about 29 min of training and evaluation on an NVIDIA GPU (29.4 min on an RTX 5060 Laptop in the last run).

## Repo structure

- `deliverable1.ipynb`: final notebook. It does not import `utils.py`, does not read `protocol/manifest.csv` or `LEYENDA_DATA_DIR`: it reads `data/Datasets`, saves the MobileNetV2 weights to `models/final_pretrained.keras` (not versioned) and recomputes inventory, duplicates, split and leakage check itself (seed 42).
- `00_data_prep.ipynb`: builds `protocol/manifest.csv`. Reads `LEYENDA_DATA_DIR` if set, otherwise `data/Datasets`.
- `notebooks/d1_dev/`: development notebooks of D1, `00_data_prep.ipynb` to `04_pretrained.ipynb`. They read `protocol/manifest.csv` and import `src/utils.py`; they read the data from `data/Datasets` only. Their first cell goes back to the repo root, so they can be opened from anywhere.
- `src/`: shared code. `config.py` (image size, batch size, seed, noise level, paths), `data.py` (D2: photo split read from the manifest, `tf.data` pipeline with Gaussian noise), `utils.py` (D1 helpers: `load_images`, `class_weights`, `evaluate`, `plot_curves`).
- `tests/`: `python -m tests.test_data` (run from the repo root).
- `deliverable2.ipynb`: final notebook of D2, denoising with a convolutional auto-encoder and a U-Net (loading, noise, models, training, PSNR/SSIM comparison, case study, analysis). Open it from the repo root. Training takes about 15 min on a GPU; the weights are written to `models/d2_*.weights.h5` (not versioned) and reloaded by the notebook if they exist (set `RETRAIN = True` to train again).
- `protocol/manifest.csv`: one row per image (`drop_reason`, `split`).
- `reports/`: written by `deliverable1.ipynb`: `final_val_summary.csv`, `final_test_results.csv` and 7 PNG figures (MLP and CNN architecture, baseline curves, variant curves, pretrained curves, confusion matrix, error by origin); `reports/0*` (development) is not versioned.
- `requirements.txt`: the dependencies to install (see Setup).
- `requirements.lock.blackwell.txt`: exact versions frozen on one machine with an NVIDIA Blackwell GPU (RTX 50 series). Only for that kind of GPU; everybody else uses `requirements.txt`.

## Data

**Where to get it:** https://file.legionera.fr/s/jAFr/datascience (download the archive and unzip it at the repo root so that the folder below exists; `data/` is never committed).

Expected structure of `data/` (the folder names are the class names, and are read as is):

```
data/
└── Datasets/
    ├── Painting/    (painting_00001.jpg, ...)
    ├── Photo/       (photo_0001.jpg, ...)
    ├── Schematics/
    ├── Sketch/
    └── Text/
```

`data/Datasets/<origin>/` at the repo root, not versioned: 41,399 files in 5 folders (Painting, Photo, Schematics, Sketch, Text).
Dropped, never deleted: 696 exact duplicates (same md5), 401 images with a shortest side under 32 px (all Schematics). Kept: 40,302, split 70/15/15 stratified by origin (28,211 / 6,045 / 6,046), no md5 in two splits.
The md5 of `protocol/manifest.csv` is `fe4b4a4407369bdfdfc15700576a0515`.

## Model weights

The trained weights (`models/final_pretrained.keras`, about 21 MB) are not versioned. Download them from Google Drive: TODO_DRIVE_LINK, then place the file in `models/` (create the folder if needed). `deliverable1.ipynb` does not reload them yet: it retrains and overwrites this file on every run.

## Setup

1. Verified stack: Python 3.12, TensorFlow 2.21.0 (stable), Keras 3.15.1 (also the stack of the last run). The exact versions of every package are listed in `requirements.lock.blackwell.txt` (for Blackwell GPUs).
2. `sudo apt install graphviz` (system binary needed by `plot_model`), then `pip install -r requirements.txt` (`pydot` is included).
3. On the 3.12 stack, set `LD_LIBRARY_PATH` to the pip `nvidia/cusolver/lib` folder (command below), otherwise no GPU is found.
4. Internet is needed once to download the ImageNet weights of MobileNetV2.
5. Every TensorFlow notebook stops with a message if no GPU is found.

## How to install and run

```bash
git clone <repo-url> && cd leyenda
python3.12 -m venv .venv && source .venv/bin/activate
sudo apt install graphviz
pip install -r requirements.txt
# GPU on pip: pip install "tensorflow[and-cuda]==2.21.0", then the LD_LIBRARY_PATH line of the Environment section
# put the data in data/Datasets/ (see Data), then:
jupyter lab deliverable1.ipynb     # Run > Restart Kernel and Run All Cells (about 29 min)
```

`deliverable1.ipynb` must be opened from the repo root (its paths are relative). Run `notebooks/d1_dev/00_data_prep.ipynb`, then optionally `01` to `04` (`02_cnn.ipynb` reads `reports/test_results.csv` written by `01`), then `deliverable1.ipynb`.
Seed 42 everywhere. Two executions give slightly different numbers (GPU non-determinism): see the Limits cell of `deliverable1.ipynb`.

## Branches

Everyone works on their own branch (e.g. `maxime/deliverable-1`); never push to `main`.

## Environment

Maxime's stack (WSL2 Ubuntu 26.04, RTX 5060 Laptop GPU 8 GB, driver 616.92): Python 3.12, TensorFlow 2.21.0, Keras 3.15.1, numpy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1, CUDA 12.9 libraries from pip (`nvidia-*-cu12`). The notebooks print the exact library versions they ran with (first code cell).

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
