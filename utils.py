"""Helpers shared by the notebooks (code copied from 01_mlp.ipynb, plus plot_curves)."""
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             confusion_matrix, precision_recall_fscore_support)

DATA_DIR, CACHE = Path("data/Datasets"), Path("cache")
METRICS = ["precision", "recall", "f1", "balanced_acc", "pr_auc", "accuracy"]

def _load_one(path, size):
    with Image.open(DATA_DIR / path) as im:
        return np.asarray(im.convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.uint8)

def load_images(df, size=128):
    """RGB uint8 array (n, size, size, 3), cached in cache/ (key = list of paths + size)."""
    key = hashlib.md5(("|".join(df["path"]) + f"|{size}").encode()).hexdigest()[:12]
    fp = CACHE / f"images_{size}_{len(df)}_{key}.npy"
    if fp.exists():
        return np.load(fp)
    with ThreadPoolExecutor(8) as ex:
        X = np.stack(list(ex.map(lambda p: _load_one(p, size), df["path"])))
    np.save(fp, X)
    return X

def class_weights(y_train):
    """Balanced weights n / (2 * n_c), from training labels only."""
    n, n1 = len(y_train), int(np.sum(y_train))
    return {0: n / (2 * (n - n1)), 1: n / (2 * n1)}

def evaluate(y_true, proba, origin, threshold=0.5):
    """Photo-class metrics at a fixed threshold, plus error rate by origin."""
    y_true, proba = np.asarray(y_true), np.asarray(proba).ravel()
    y_pred = (proba >= threshold).astype(int)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[1], zero_division=0)
    err = pd.Series(y_pred != y_true, index=np.asarray(origin)).groupby(level=0).mean()
    return {"precision": p[0], "recall": r[0], "f1": f[0],
            "balanced_acc": balanced_accuracy_score(y_true, y_pred),
            "pr_auc": average_precision_score(y_true, proba),
            "accuracy": (y_pred == y_true).mean(),
            "cm": confusion_matrix(y_true, y_pred, labels=[0, 1]),
            "error_by_origin": err}

def plot_curves(history, title="", axes=None):
    """Train/validation loss and accuracy per epoch from a Keras history dict."""
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    epochs = np.arange(1, len(history["loss"]) + 1)
    for ax, m in zip(axes, ["loss", "accuracy"]):
        ax.plot(epochs, history[m], color="#2a78d6", lw=2, marker="o", ms=4, label="train")
        ax.plot(epochs, history[f"val_{m}"], color="#eb6834", lw=2, marker="o", ms=4, label="validation")
        ax.set(title=f"{title} - {m}".strip(" -"), xlabel="epoch")
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    return axes
