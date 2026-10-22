"""Tests for src/coco.py on fake COCO annotations (no download needed).

Run from the repo root:  python -m tests.test_coco
(also works with pytest if you install it: python -m pytest tests)
"""
import json
import tempfile
from pathlib import Path

from src.coco import coco_subsets, image_path, load_captions


def _fake_annotations(coco_dir, n_train=30, n_val=12):
    """captions_train2017.json and captions_val2017.json in the COCO format, ids in shuffled order."""
    folder = Path(coco_dir) / "annotations"
    folder.mkdir(parents=True)
    for split, n, first_id in [("train2017", n_train, 1000), ("val2017", n_val, 5000)]:
        ids = [first_id + (7 * i) % n for i in range(n)]   # not sorted, like a real JSON could be
        images = [{"id": i, "file_name": f"{i:012d}.jpg", "coco_url": f"http://x/{split}/{i:012d}.jpg"} for i in ids]
        annotations = [{"image_id": i, "id": 10 * i + k, "caption": f" caption {k} of {i} "} for i in ids for k in range(5)]
        images.append({"id": first_id + n, "file_name": "no_caption.jpg", "coco_url": "http://x/none.jpg"})
        (folder / f"captions_{split}.json").write_text(json.dumps({"images": images, "annotations": annotations}))


def test_load_captions():
    with tempfile.TemporaryDirectory() as coco_dir:
        _fake_annotations(coco_dir)
        images = load_captions("train2017", coco_dir)
    assert [im["id"] for im in images] == sorted(im["id"] for im in images)   # sorted by id
    assert all(len(im["captions"]) == 5 for im in images)                    # grouped by image
    assert images[0]["captions"][0] == f"caption 0 of {images[0]['id']}"      # stripped
    assert "no_caption.jpg" not in {im["file_name"] for im in images}         # images without caption dropped


def test_subsets_disjoint_and_reproducible():
    with tempfile.TemporaryDirectory() as coco_dir:
        _fake_annotations(coco_dir)
        train, val, test = coco_subsets(10, 5, 4, seed=42, coco_dir=coco_dir)
        again = coco_subsets(10, 5, 4, seed=42, coco_dir=coco_dir)
        other = coco_subsets(10, 5, 4, seed=0, coco_dir=coco_dir)
        path = image_path(test[0], coco_dir)
    assert (len(train), len(val), len(test)) == (10, 5, 4)
    assert not {im["id"] for im in train} & {im["id"] for im in val}
    assert {im["split"] for im in train + val} == {"train2017"} and {im["split"] for im in test} == {"val2017"}
    assert [[im["id"] for im in s] for s in (train, val, test)] == [[im["id"] for im in s] for s in again]
    assert [im["id"] for im in train] != [im["id"] for im in other[0]]   # the seed matters
    assert path == str(Path(coco_dir) / "val2017" / test[0]["file_name"])


if __name__ == "__main__":
    test_load_captions()
    test_subsets_disjoint_and_reproducible()
    print("OK: all tests pass")
