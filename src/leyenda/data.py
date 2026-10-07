"""Dataset indexing, auditing, splitting and tf.data pipelines."""

import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .env import DATA_DIR

RAW_DIR = DATA_DIR / "raw"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}


def _inspect(path: Path) -> dict:
    from PIL import Image

    info = {"path": str(path), "bytes": path.stat().st_size}
    info["md5"] = hashlib.md5(path.read_bytes()).hexdigest()
    try:
        with Image.open(path) as im:
            info.update(width=im.width, height=im.height, mode=im.mode, format=im.format)
            im.verify()  # detects truncated / corrupt files
        info["readable"] = True
    except Exception as exc:
        info.update(readable=False, error=type(exc).__name__)
    return info


def build_index(root: Path = RAW_DIR, positive: str = "Photo", workers: int = 16) -> pd.DataFrame:
    """One row per image: category (= source folder), binary label, size, mode, hash, readability."""
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS)
    with ThreadPoolExecutor(workers) as pool:
        rows = list(pool.map(_inspect, paths))
    df = pd.DataFrame(rows)
    df["category"] = [Path(p).relative_to(root).parts[0] for p in df["path"]]
    df["label"] = (df["category"] == positive).astype(int)  # 1 = photo, 0 = other
    df["ext"] = [Path(p).suffix.lower() for p in df["path"]]
    return df


def clean_index(df: pd.DataFrame, min_side: int = 32) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop unreadable files, images smaller than `min_side` px (extraction debris: colour-bar strips,
    1-px slivers, which carry no information once resized) and exact duplicates. Duplicates whose
    copies sit in different categories are ambiguous labels, so every copy is dropped; otherwise the
    first copy is kept. Returns (clean_df, removed_df with a `reason` column)."""
    removed = [df[~df["readable"]].assign(reason="unreadable")]
    ok = df[df["readable"]]
    tiny = ok[["width", "height"]].min(axis=1) < min_side
    removed.append(ok[tiny].assign(reason=f"smaller than {min_side}px"))
    ok = ok[~tiny]
    n_cats = ok.groupby("md5")["category"].nunique()
    conflict = ok["md5"].isin(n_cats[n_cats > 1].index)
    removed.append(ok[conflict].assign(reason="duplicate across categories"))
    ok = ok[~conflict]
    dup = ok.duplicated("md5", keep="first")
    removed.append(ok[dup].assign(reason="duplicate within category"))
    return ok[~dup].reset_index(drop=True), pd.concat(removed, ignore_index=True)


def split(df: pd.DataFrame, val: float = 0.15, test: float = 0.15, seed: int = 42) -> pd.DataFrame:
    """Stratified by *category* (not just label) so every source, e.g. paintings, is represented
    in the same proportion in train / val / test."""
    from sklearn.model_selection import train_test_split

    train_df, rest = train_test_split(df, test_size=val + test, stratify=df["category"], random_state=seed)
    val_df, test_df = train_test_split(rest, test_size=test / (val + test), stratify=rest["category"],
                                       random_state=seed)
    return pd.concat([train_df.assign(split="train"), val_df.assign(split="val"),
                      test_df.assign(split="test")]).reset_index(drop=True)


def train_eval_subset(train_df: pd.DataFrame, frac: float = 0.12, seed: int = 0) -> pd.DataFrame:
    """Fixed per-source sample of the training set, evaluated *without* augmentation, so training error
    is measured under the same conditions as validation error (needed for the bias/variance analysis)."""
    return train_df.groupby("category", group_keys=False).sample(frac=frac, random_state=seed)


def class_weights(labels) -> dict:
    """Inverse-frequency weights so the minority class (photos) counts as much as the rest."""
    labels = np.asarray(labels)
    counts = np.bincount(labels)
    return {i: len(labels) / (len(counts) * c) for i, c in enumerate(counts)}


def make_dataset(df: pd.DataFrame, img_size: int = 160, batch_size: int = 64, shuffle: bool = False,
                 cache: str | Path | None = None, seed: int = 42):
    """tf.data pipeline: decode (jpg/png, grayscale/RGBA -> RGB) -> resize -> uint8 batches.

    Images stay in [0, 255]; scaling is done inside each model so the saved model accepts raw images.
    `cache` (a file path) stores decoded+resized images on disk after the first epoch, which avoids
    re-decoding multi-megapixel paintings every epoch.
    """
    import tensorflow as tf

    def load(path, label):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img = tf.image.resize(img, (img_size, img_size), antialias=True)
        return tf.cast(tf.clip_by_value(img, 0, 255), tf.uint8), label

    ds = tf.data.Dataset.from_tensor_slices((df["path"].values, df["label"].values.astype("float32")))
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    if cache:
        # key the cache on the exact file list + size: a stale cache built from other images would
        # otherwise be silently reused and misalign predictions with labels
        key = hashlib.md5(("\n".join(df["path"]) + f"|{img_size}").encode()).hexdigest()[:10]
        cache = Path(f"{cache}_{key}")
        cache.parent.mkdir(parents=True, exist_ok=True)
        ds = ds.cache(str(cache))
    if shuffle:
        ds = ds.shuffle(4096, seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def show_samples(df: pd.DataFrame, n: int = 6, size: int = 2.2, seed: int = 0):
    """Grid with n random images per category."""
    import matplotlib.pyplot as plt
    from PIL import Image

    cats = sorted(df["category"].unique())
    fig, axes = plt.subplots(len(cats), n, figsize=(n * size, len(cats) * size))
    for r, cat in enumerate(cats):
        sample = df[df["category"] == cat].sample(n, random_state=seed)
        for c, path in enumerate(sample["path"]):
            ax = axes[r, c]
            with Image.open(path) as im:
                ax.imshow(im.convert("RGB"))
            ax.set_xticks([]), ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(cat, fontsize=12)
    fig.tight_layout()
    plt.show()
