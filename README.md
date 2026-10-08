# Leyenda – Deliverable 1: Photo vs non-Photo

Binary image classification (TouNum): Photo (label 1) against Painting, Schematics, Sketch and Text (label 0).

## Repo structure

- `00_data_prep.ipynb`: inventory, duplicates, size threshold, split, leakage check. Output: `protocol/manifest.csv`.
- `01_mlp.ipynb`, `02_cnn.ipynb`: MLP and CNN baselines, Photo vs the 4 other origins (and Photo vs Text only).
- `03_improvements.ipynb`, (`04_pretrained.ipynb`), then `livrable1.ipynb` (final assembly): planned, not written yet.
- `utils.py`: helpers shared by notebooks 01 and later (`load_images`, `class_weights`, `evaluate`, `plot_curves`). `livrable1.ipynb` will copy them inline instead of importing.
- `protocol/manifest.csv`: one row per image (path, origin, label, size, md5, drop_reason, split).
- `reports/`: CSV tables and PNG figures. `cache/`: ignored by git.
- `requirements.txt`, `.gitignore`.

## How to run

1. Put the images in `data/Datasets/<origin>/` (5 folders: Painting, Photo, Schematics, Sketch, Text). `00_data_prep.ipynb` also reads `LEYENDA_DATA_DIR` if set.
2. Open the notebooks from the repo root (paths are relative), with a kernel from one of the environments below.
3. Run `00_data_prep.ipynb` first (about 1 min), then the other notebooks in numeric order. Notebooks 01 and later need a GPU and stop with a message otherwise.
4. `02_cnn.ipynb` reads the MLP and decision-tree rows from `reports/test_results.csv`, written by `01_mlp.ipynb`: run 01 before 02.

## Data protocol

- 41,399 files. Dropped (marked in `drop_reason`, never deleted): 696 `exact_duplicate` (same md5, first path in alphabetical order kept), 401 `too_small` (shortest side under 32 px, all Schematics). 0 `unreadable`, 0 `not_an_image`, 0 `label_conflict`.
- The 32 px threshold (`MIN_SIDE_PX`) is proposed by the author after visual inspection, to be validated by the group.
- Kept: 40,302. Split 70/15/15 after cleaning, stratified by origin, seed 42: 28,211 train, 6,045 val, 6,046 test. No md5 is in two splits.
- Test on the natural distribution: 24.79 % Photo, majority-class rate 0.7521.
- Regenerating the manifest gives the same file: md5 `fe4b4a4407369bdfdfc15700576a0515`.

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

## Not verified

- The command used to install the author's stack.
- The lower bounds in `requirements.txt`: only the versions printed by the notebooks (stack above) and the 3.12 stack were run.
- Running the notebooks on the 3.12 stack (only a GPU benchmark and the `.keras` exchange were run), including `LD_LIBRARY_PATH` in a VS Code kernel.
- Native Windows or macOS.
- That the pre-release versions of the author's stack (TensorFlow 2.22.0rc0, keras-nightly) stay downloadable from PyPI.
