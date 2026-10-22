"""Tests for src/coco.py on a fake mini COCO (no real data needed).

Run from the repo root:  python -m tests.test_coco
"""
import json
import tempfile
import zipfile
from pathlib import Path

from src.coco import (ANNOTATIONS_ZIP, CAPTIONS_MEMBER, IMAGES_ZIP, choose_subset,
                      extract_images, image_filename, load_captions)

# Three images; image 7 has two captions, image 3 has one with spaces and a line break around it.
ANNOTATIONS = [
    {"image_id": 7, "caption": "A dog on a sofa."},
    {"image_id": 3, "caption": "  A red bus.\n"},
    {"image_id": 7, "caption": "A dog sleeping."},
    {"image_id": 12, "caption": "Two birds."},
]


def _fake_coco(folder):
    """Write a fake annotations zip and a fake images zip (the 'images' are just bytes)."""
    with zipfile.ZipFile(Path(folder) / ANNOTATIONS_ZIP, "w") as z:
        z.writestr(CAPTIONS_MEMBER, json.dumps({"annotations": ANNOTATIONS}))
    with zipfile.ZipFile(Path(folder) / IMAGES_ZIP, "w") as z:
        for image_id in (3, 7, 12):
            z.writestr("train2014/" + image_filename(image_id), b"image-%d" % image_id)


def test_image_filename():
    assert image_filename(270070) == "COCO_train2014_000000270070.jpg"


def test_load_captions_groups_by_image():
    with tempfile.TemporaryDirectory() as folder:
        _fake_coco(folder)
        captions = load_captions(folder)
    assert captions == {7: ["A dog on a sofa.", "A dog sleeping."], 3: ["A red bus."], 12: ["Two birds."]}


def test_subset_is_reproducible_nested_and_independent_of_input_order():
    ids = list(range(1000))
    small, big = choose_subset(ids, 20), choose_subset(ids, 50)
    assert len(small) == 20 and len(set(small)) == 20
    assert big[:20] == small                                       # scaling up keeps the first images
    assert choose_subset(ids, 20) == small                         # same seed, same result
    assert choose_subset(list(reversed(ids)), 20) == small         # the order of the input does not matter
    assert choose_subset(ids, 20, seed=1) != small                 # another seed, another subset


def test_extract_images_copies_only_the_subset_and_can_run_twice():
    with tempfile.TemporaryDirectory() as folder:
        _fake_coco(folder)
        paths = extract_images([12, 3], folder)
        assert [Path(p).name for p in paths] == [image_filename(12), image_filename(3)]
        assert Path(paths[0]).read_bytes() == b"image-12"
        assert sorted(p.name for p in (Path(folder) / "train2014").iterdir()) == sorted(Path(p).name for p in paths)
        assert extract_images([12, 3], folder) == paths             # second run: nothing breaks


if __name__ == "__main__":
    test_image_filename()
    test_load_captions_groups_by_image()
    test_subset_is_reproducible_nested_and_independent_of_input_order()
    test_extract_images_copies_only_the_subset_and_can_run_twice()
    print("OK: all tests pass")
