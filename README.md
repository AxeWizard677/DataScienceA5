# Leyenda: photo classification, denoising and captioning (TouNum)

Binary image classification for TouNum: Photo (label 1) against Painting, Schematics, Sketch and Text (label 0).
The main deliverable is `deliverable1.ipynb`: self-contained. Final model: MobileNetV2 fine-tuned at 224x224 (section 7.2), **99.07% test accuracy, 99.05% balanced accuracy**. It runs on a 2 GB GPU: about 35 min when the improved models are reloaded from `models/` (`REUSE = True`), about 2 h when everything is trained (GRID T4-2Q, 2 GB).

## Repo structure

- `deliverable1.ipynb`: final notebook. It does not import `utils.py`, does not read `protocol/manifest.csv` or `LEYENDA_DATA_DIR`: it reads `data/Datasets`, saves the MobileNetV2 weights to `models/final_pretrained.keras` (not versioned) and recomputes inventory, duplicates, split and leakage check itself (seed 42).
- `00_data_prep.ipynb`: builds `protocol/manifest.csv`. Reads `LEYENDA_DATA_DIR` if set, otherwise `data/Datasets`.
- `notebooks/d1_dev/`: development notebooks of D1, `00_data_prep.ipynb` to `04_pretrained.ipynb`. They read `protocol/manifest.csv` and import `src/utils.py`; they read the data from `data/Datasets` only. Their first cell goes back to the repo root, so they can be opened from anywhere.
- `src/`: shared code. `config.py` (image size, batch size, seed, noise level, paths), `data.py` (D2: photo split read from the manifest, `tf.data` pipeline with Gaussian noise), `utils.py` (D1 helpers: `load_images`, `class_weights`, `evaluate`, `plot_curves`).
- `tests/`: `python -m tests.test_data` and `python -m tests.test_coco` (run from the repo root, no real data needed).
- `deliverable2.ipynb`: final notebook of D2, denoising with a convolutional auto-encoder and a U-Net (loading, noise, models, training, PSNR/SSIM comparison, case study, analysis), then a U-Net trained on several noise levels and a residual variant (section 8), further improvements at 128x128 (section 9: learning-rate schedule, scan-like damage, a noise gate), denoising at the original resolution (sections 10-13: training on crops of the original photos, test-time augmentation, an L1 + SSIM loss, one model for Gaussian noise and scan-like damage, and a larger U-Net in 13.2), the final pipeline (section 14), a test on 40 photos never seen, downloaded from picsum.photos into `data/unseen_random/` (section 15, needs internet the first time), and the saved final model with `denoise_folder` to clean a folder of photos (section 16, `models/d2_final_unet.keras` and `models/d2_final_config.json`). Open it from the repo root. Training all the models takes about 1 h 30 on a 2 GB GPU, a full run with every model reloaded about 2 h (most of it is the evaluation at full resolution); the weights are written to `models/d2_*.weights.h5` (not versioned) and reloaded by the notebook if they exist (set `RETRAIN = True` to train again).
- `deliverable3.ipynb`: final notebook of D3, image captioning on MS COCO 2017 with a pretrained classification CNN and an RNN trained by us. Sections 1-10: a first model (InceptionV3 features, 12,000 images). Section 11: improvements one change at a time, chosen on validation (all of COCO split 70/15/15, ConvNeXt-Large features, beam search without repeated words, SCST reinforcement learning on CIDEr-D). Section 12: the final model (architecture, pre-processing, CNN and RNN code, training curves) tested on the whole 15 % test split, 18,493 photos: **BLEU-4 0.301, CIDEr-D 1.124** (a human caption against the 4 others: 0.195 / 0.873). Section 13: the whole workflow D1 (`deliverable1.ipynb`) -> D2 (`deliverable2.ipynb`) -> D3 on images given by folder, link or upload. Section 14: analysis. **Section 15, last cell: change `INPUT_FOLDER` to the evaluation dataset.** To run only the workflow: run the first code cell, the cell of 11.3 that defines `decode`, then section 13 and the last cell (the weights are in `models/`, COCO is not needed). Re-running every section needs COCO (`src/coco.py` downloads it) and recomputes the features (31 GB per encoder).
- `pipeline.ipynb`: the improved workflow alone (improved D1 with categories, improved D2, final D3), ready in a few seconds for a live demo: images by folder (`demo_images/`, `my_photos/`), link or upload; the last cell takes the evaluation folder (`INPUT_FOLDER`) and sorts the images by category.
- `demo_images/`: 7 test images (5 photos clean, noisy, scan-damaged or blurred, a sketch, a diagram), used by the demo and last cells.
- **Improved versions, made with the time left:** `deliverable1_improved.ipynb` (own ResNet, MobileNetV2 improved one change at a time, 224x224, 99.07 % test accuracy; model `models/d1_improved_final.keras`, reports in `reports/improved/`), `deliverable2_improved.ipynb` (restoration at the original resolution, larger U-Net, test-time augmentation, gate, final model `models/d2_final_unet.keras`), `deliverable3_improved.ipynb` (same captioner, workflow with the improved D1, which also gives the category of rejected images and accepts damaged photos after a repair by D2, and the improved D2).
- `protocol/manifest.csv`: one row per image (`drop_reason`, `split`).
- `reports/`: written by `deliverable1.ipynb`: `final_val_summary.csv`, `final_test_results.csv`, `final_pretrained_improvements.csv` (section 7.2), `final_cost.csv` (weights, training time, speed) and PNG figures (MLP and CNN architecture, baseline curves, variant curves, ResNet curves, pretrained curves, improvements, confusion matrix, error by origin); `reports/0*` (development) is not versioned.
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

The final weights needed by the workflows are versioned in `models/` (forced past `.gitignore`): `final_pretrained.keras` (D1), `d2_unet_multi_*.weights.h5` (D2), `d3_final_captioner.keras` + `d3_final_vocab.json` + `d3_final_config.json` (D3), and for the improved versions `d1_improved_final.*`, `d1_category_*`, `d2_final_unet.keras`, `d2_final_config.json`. The other training weights are not versioned.


The trained weights are not versioned. `models/final_pretrained.keras` (final model, 224x224) and `models/final_pretrained.json` (its image size and threshold) are written at the end of `deliverable1.ipynb`. The improved models (`models/d1_resnet.keras`, `d1_p3*.keras`, `d1_p4.keras`, `d1_p5*.keras`, with their `_history.json`) are reloaded when `REUSE = True` and retrained when `REUSE = False` or missing. Download link: TODO_DRIVE_LINK.

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
