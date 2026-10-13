"""Tests for src/data.py on fake images (no real data needed).

Run from the repo root:  python -m tests.test_data
(also works with pytest if you install it: python -m pytest tests)
"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.config import IMG_SIZE
from src.data import list_image_paths, make_dataset, photo_splits

N_IMAGES, BATCH = 20, 8


def _fake_photos(folder):
    """Create N_IMAGES jpg files of different sizes (to check the resize as well)."""
    rng = np.random.default_rng(0)
    for i in range(N_IMAGES):
        h, w = rng.integers(40, 300, size=2)
        Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8)).save(Path(folder) / f"p{i:02d}.jpg")


def test_photo_splits_from_manifest():
    """Fake manifest: only kept photos are returned, split as written, no overlap."""
    rows = [  # (path, label, drop_reason, split)
        ("Photo/a.jpg", 1, None, "train"), ("Photo/b.jpg", 1, None, "train"),
        ("Photo/c.jpg", 1, None, "val"), ("Photo/d.jpg", 1, None, "test"),
        ("Photo/e.jpg", 1, "exact_duplicate", None),   # dropped by D1: must not appear
        ("Text/f.jpg", 0, None, "train"),               # not a photo: must not appear
    ]
    with tempfile.TemporaryDirectory() as folder:
        manifest = Path(folder) / "manifest.csv"
        pd.DataFrame(rows, columns=["path", "label", "drop_reason", "split"]).to_csv(manifest, index=False)
        train, val, test = photo_splits(manifest, data_dir="root")
    assert [Path(p).name for p in train] == ["a.jpg", "b.jpg"]
    assert [Path(p).name for p in val] == ["c.jpg"] and [Path(p).name for p in test] == ["d.jpg"]
    assert not (set(train) & set(val) | set(train) & set(test) | set(val) & set(test))


def test_shapes_range_and_deterministic_test_noise():
    with tempfile.TemporaryDirectory() as folder:
        _fake_photos(folder)
        paths = list_image_paths(folder)
        assert len(paths) == N_IMAGES

        noisy, clean = next(iter(make_dataset(paths, BATCH)))
        assert noisy.shape == (BATCH, IMG_SIZE, IMG_SIZE, 3) and clean.shape == noisy.shape
        for t in (noisy, clean):
            assert 0.0 <= float(t.numpy().min()) and float(t.numpy().max()) <= 1.0
        assert not np.array_equal(noisy.numpy(), clean.numpy())  # noise was actually added

        # Test set: two independent calls give the same noisy images, batch by batch.
        a = [n.numpy() for n, _ in make_dataset(paths, BATCH, training=False)]
        b = [n.numpy() for n, _ in make_dataset(paths, BATCH, training=False)]
        assert len(a) == len(b) == 3  # 8 + 8 + 4 images
        assert all(np.array_equal(x, y) for x, y in zip(a, b))

        # Training: the noise changes from one epoch to the next.
        train_ds = make_dataset(paths, BATCH, training=True)
        e1 = next(iter(train_ds))[0].numpy()
        e2 = next(iter(train_ds))[0].numpy()
        assert not np.array_equal(e1, e2)


if __name__ == "__main__":
    test_photo_splits_from_manifest()
    test_shapes_range_and_deterministic_test_noise()
    print("OK: all tests pass")
