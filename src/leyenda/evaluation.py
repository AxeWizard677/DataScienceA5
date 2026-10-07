"""Evaluation helpers: per-source accuracy, confusion matrix, error gallery."""

import numpy as np
import pandas as pd


def predict(model, ds) -> np.ndarray:
    return model.predict(ds, verbose=0).ravel()


def per_category_report(df: pd.DataFrame, probs: np.ndarray, threshold: float = 0.5) -> pd.DataFrame:
    """Accuracy per *source category*: shows which kinds of non-photo images fool the model.
    For non-photo categories this is the fraction correctly rejected; for Photo, the recall."""
    out = df[["category", "label"]].copy()
    out["pred"] = (probs >= threshold).astype(int)
    out["correct"] = out["pred"] == out["label"]
    rep = out.groupby("category").agg(n=("correct", "size"), accuracy=("correct", "mean"),
                                      predicted_photo=("pred", "mean"))
    return rep.sort_values("accuracy")


def binary_metrics(labels, probs, threshold: float = 0.5) -> dict:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    pred = (np.asarray(probs) >= threshold).astype(int)
    return {"accuracy": accuracy_score(labels, pred), "precision": precision_score(labels, pred, zero_division=0),
            "recall": recall_score(labels, pred), "f1": f1_score(labels, pred),
            "roc_auc": roc_auc_score(labels, probs)}


def plot_confusion(df: pd.DataFrame, probs: np.ndarray, threshold: float = 0.5, title: str = ""):
    """Rows = true source category, columns = predicted class. More informative than a 2x2 matrix."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    pred = np.where(probs >= threshold, "photo", "other")
    table = pd.crosstab(df["category"].values, pred, normalize="index")[["other", "photo"]]
    fig, ax = plt.subplots(figsize=(5, 0.6 * len(table) + 1.2))
    sns.heatmap(table, annot=True, fmt=".1%", cmap="Blues", vmin=0, vmax=1, cbar=False, ax=ax)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true source")
    ax.set_title(title)
    fig.tight_layout()
    plt.show()


def plot_errors(df: pd.DataFrame, probs: np.ndarray, threshold: float = 0.5, n: int = 12, size: float = 2.2):
    """The most confident mistakes: what the model gets wrong and how badly."""
    import matplotlib.pyplot as plt
    from PIL import Image

    d = df.assign(prob=probs)
    d["confidence_wrong"] = np.where(d["label"] == 1, 1 - d["prob"], d["prob"])
    wrong = d[(d["prob"] >= threshold) != (d["label"] == 1)].nlargest(n, "confidence_wrong")
    if wrong.empty:
        print("No errors.")
        return
    cols = min(6, len(wrong))
    rows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * size, rows * (size + 0.3)), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (_, r) in zip(axes.ravel(), wrong.iterrows()):
        with Image.open(r["path"]) as im:
            ax.imshow(im.convert("RGB"))
        ax.set_title(f"{r['category']}\nP(photo)={r['prob']:.2f}", fontsize=9)
    fig.tight_layout()
    plt.show()


def plot_architecture(model, save_to=None, show: bool = True, title: str | None = None):
    """Block diagram of the model's top-level layers (output shape + parameter count), drawn with
    matplotlib so it works without the Graphviz system program that keras.utils.plot_model needs.
    Nested sub-models (augmentation pipeline, pre-trained MobileNetV2) appear as a single block."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    colors = {"InputLayer": "#d9d9d9", "Conv2D": "#9ecae1", "MaxPooling2D": "#c6dbef",
              "BatchNormalization": "#fdd0a2", "Activation": "#fee6ce", "Dropout": "#fcbba1",
              "Dense": "#a1d99b", "Flatten": "#e5f5e0", "GlobalAveragePooling2D": "#e5f5e0",
              "Functional": "#bcbddc", "Sequential": "#dadaeb", "Rescaling": "#f0f0f0",
              "Resizing": "#f0f0f0"}
    rows = []
    for layer in model.layers:
        kind = type(layer).__name__
        try:
            shape = tuple(layer.output.shape[1:])
        except Exception:
            shape = ""
        rows.append((layer.name, kind, shape, layer.count_params()))

    h = 0.62
    fig, ax = plt.subplots(figsize=(7.5, 0.5 + h * len(rows)))
    for i, (name, kind, shape, params) in enumerate(rows):
        y = len(rows) - i - 1
        ax.add_patch(FancyBboxPatch((0.05, y + 0.12), 0.9, 0.76, boxstyle="round,pad=0.01",
                                    fc=colors.get(kind, "#f7f7f7"), ec="#555555", lw=0.8))
        ax.text(0.08, y + 0.5, f"{kind}", va="center", fontsize=9, weight="bold")
        ax.text(0.42, y + 0.5, f"{name}", va="center", fontsize=8, color="#444444")
        ax.text(0.92, y + 0.5, f"{shape}   {params:,} p", va="center", ha="right", fontsize=8)
        if i < len(rows) - 1:
            ax.annotate("", xy=(0.5, y + 0.1), xytext=(0.5, y - 0.1 + 0.22),
                        arrowprops=dict(arrowstyle="<-", color="#555555", lw=0.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, len(rows))
    ax.axis("off")
    ax.set_title(title or f"{model.name}: {model.count_params():,} parameters "
                          f"({sum(int(w.numpy().size) for w in model.trainable_weights):,} trainable)",
                 fontsize=10)
    fig.tight_layout()
    if save_to:
        fig.savefig(save_to, dpi=110)
    if show:
        plt.show()
    else:
        plt.close(fig)
