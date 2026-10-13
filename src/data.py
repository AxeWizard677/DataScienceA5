"""D2 (denoising) data: paths, split, and tf.data with Gaussian noise.

Principle: the dataset is never held in RAM. We work on the LIST OF PATHS, and
tf.data reads, resizes and adds noise to the images batch by batch.
Usage from the repo root: `from src.data import make_dataset`.
"""
from pathlib import Path

import pandas as pd
import tensorflow as tf

from src.config import BATCH_SIZE, DATA_DIR, IMG_SIZE, MANIFEST_PATH, NOISE_STD, SEED

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def list_image_paths(root):
    """Sorted list of image paths (jpg/jpeg/png) under `root`, subfolders included.

    Used for folders that are not in the manifest (e.g. the unseen demo dataset).
    """
    # sorted(): the file-system order is not guaranteed; without sorting the order
    # (and so the fixed val/test noise) would differ from one machine to another.
    paths = [str(p) for p in Path(root).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(paths)


def photo_splits(manifest_path=MANIFEST_PATH, data_dir=DATA_DIR):
    """(train, val, test) lists of photo paths, read from the project manifest.

    The split is the one computed once for D1 (70/15/15, checked for duplicate
    leakage). Reusing it means an image never trains one model and tests another,
    which matters when D3 chains the classifier and the denoiser. It is done on
    PATHS, so it is always before any noise is added.
    """
    df = pd.read_csv(manifest_path)
    # Photos only (label 1), and only the images D1 kept (drop_reason is empty).
    df = df[(df["label"] == 1) & df["drop_reason"].isna()]
    # sorted(): same order on every machine, so the fixed val/test noise is reproducible.
    return tuple(
        sorted(str(Path(data_dir) / p) for p in df.loc[df["split"] == s, "path"])
        for s in ("train", "val", "test")
    )


def add_noise(img, std, seed=None):
    """Add N(0, std) Gaussian noise to `img` (float32 in [0, 1]), then clip to [0, 1].

    seed=None: different noise at every call (training).
    seed=int or tensor of 2 integers: deterministic noise (test set), using the
    "stateless" TensorFlow operations: same seed -> exactly the same noise.
    """
    if seed is None:
        noise = tf.random.normal(tf.shape(img), stddev=std)
    else:
        seed = tf.convert_to_tensor(seed)
        if seed.shape.rank == 0:       # a single integer -> (seed, 0) pair required by TF
            seed = tf.stack([seed, tf.zeros_like(seed)])
        noise = tf.random.stateless_normal(tf.shape(img), seed=seed, stddev=std)
    # The clip keeps values valid: the model sees the same range as with a clean input.
    return tf.clip_by_value(img + noise, 0.0, 1.0)


def _read_image(path):
    """Read a file, force RGB, resize to IMG_SIZE x IMG_SIZE, return uint8."""
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    # antialias=True: avoids aliasing when large photos are shrunk.
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], antialias=True)
    img.set_shape([IMG_SIZE, IMG_SIZE, 3])  # decode_image does not know the shape in advance
    # Stored as uint8 (project rule): 4x less memory than float32 in the buffers.
    return tf.cast(tf.round(tf.clip_by_value(img, 0.0, 255.0)), tf.uint8)


def make_dataset(paths, batch_size=BATCH_SIZE, noisy=True, noise_std=NOISE_STD,
                 training=False, seed=SEED):
    """tf.data.Dataset of (noisy_image, clean_image) pairs, float32 in [0, 1].

    - training=True: shuffled order and noise regenerated at every epoch (the model
      never sees the same noisy image twice: free augmentation).
    - training=False: no shuffle, FIXED noise (derived from `seed` and the batch
      number), so two models are compared on exactly the same noisy images.
    - noisy=False: the input is the clean image (useful to test the reading or
      to measure a reference without noise).
    """
    ds = tf.data.Dataset.from_tensor_slices([str(p) for p in paths])
    if training:
        # We shuffle the paths (light), not the images: little memory.
        # reshuffle_each_iteration=True: new order at every epoch.
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(_read_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)

    def to_pair(batch_index, batch_u8):
        # Conversion to float32 [0, 1] only here, on a batch of `batch_size` images.
        clean = tf.cast(batch_u8, tf.float32) / 255.0
        if not noisy:
            return clean, clean
        if training:
            return add_noise(clean, noise_std), clean
        # Test/val: different seed per batch, otherwise every batch would get the same noise.
        # Holds as long as the order and the batch size do not change (no shuffle here).
        return add_noise(clean, noise_std, seed=tf.stack([tf.constant(seed, tf.int64), batch_index])), clean

    ds = ds.enumerate().map(to_pair, num_parallel_calls=tf.data.AUTOTUNE)
    # prefetch: prepares the next batch while the GPU works on the current one.
    return ds.prefetch(tf.data.AUTOTUNE)
