# DataScienceA5 – Leyenda (TouNum)

Automatic photo filtering, denoising and captioning pipeline for TouNum's digitized archives.

| Deliverable | Notebook | Due |
|---|---|---|
| 1. Binary classification (photo vs. other) | `notebooks/01_classification.ipynb` | end of week 3 |
| 2. Denoising (convolutional autoencoder) | `notebooks/02_denoising.ipynb` | end of week 4 |
| 3. Captioning (CNN encoder + RNN decoder, MS COCO) | `notebooks/03_captioning.ipynb` | end of week 5 |

## Setup (one command)

Requires [uv](https://docs.astral.sh/uv/). It installs Python 3.12 itself, so no system Python or sudo is needed.

```bash
uv sync                                      # exact versions from uv.lock
uv run python -m ipykernel install --user --name leyenda --display-name "Python 3.12 (Leyenda)"
```

In VS Code, select the **Python 3.12 (Leyenda)** kernel.

Optional system package for architecture diagrams (`keras.utils.plot_model`):
`sudo apt install graphviz` (Linux) / `winget install graphviz` (Windows).

### What changed from the workshop environment
- **Reproducible:** versions are pinned in `uv.lock` rather than a loose `pip install`.
- **Keras 3 native:** no `tf-keras` / `TF_USE_LEGACY_KERAS`. Custom losses are written as Keras 3 layers or losses.
- **GPU:** `tensorflow[and-cuda]` on Linux. `leyenda.setup()` enables memory growth (the vGPU only has 2 GB) and works around a TF 2.21 wheel bug that silently disables the GPU (missing cusolver path).
- **Experiment tracking** for the training report (see below).

## Notebook template

```python
import leyenda                      # import FIRST (pre-loads CUDA libs before TensorFlow)
leyenda.setup(seed=42)              # seeds + GPU config
import tensorflow as tf, keras
```

## Training report: failures → improvements

Each training attempt is recorded as a `Run`, failures included:

```python
run = leyenda.Run("d1_classifier", "baseline_cnn",
                  hypothesis="A 3-block CNN separates photos from scanned text",
                  config={"lr": 1e-3, "batch_size": 32, "augmentation": False})
history = model.fit(train_ds, validation_data=val_ds, epochs=30,
                    callbacks=run.callbacks(early_stopping=False))
run.finish(model, history, metrics=model.evaluate(test_ds, return_dict=True),
           verdict="failure",   # baseline | failure | improvement | best
           lesson="val_loss rises after epoch 5 (overfitting) -> add augmentation + dropout")
```

Then, in the report section of the notebook:

```python
leyenda.compare_runs("d1_classifier")          # table of all attempts
leyenda.plot_runs("d1_classifier", "val_loss")  # overlaid curves (dashed = failure)
print(leyenda.training_journal("d1_classifier"))  # hypothesis -> result -> lesson, also saved to reports/
```

Each run is saved in `runs/<deliverable>/NN_<name>/` (config, history.csv, curves.png, model.png, summary.json, TensorBoard logs).
Live view: `uv run tensorboard --logdir runs`.

## Layout

```
data/raw, data/processed   datasets (git-ignored)
notebooks/                 one notebook per deliverable
src/leyenda/               shared code (env setup, tracking, and later data loading/models)
runs/                      experiment logs (checkpoints and TensorBoard logs are git-ignored)
models/                    final exported models (git-ignored)
reports/                   training journals + figures for the presentation
```
