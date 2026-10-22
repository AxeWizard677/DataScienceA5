"""D3 (captioning) data: MS COCO 2017 captions and a reproducible image subset.

The full COCO 2017 is 18 GB of images. We only need a subset, so we read the
caption annotations (one JSON file per split) and download the chosen images one
by one from the official server (each annotation gives the URL of its image).
Usage from the repo root: `from src.coco import coco_subsets, download_images`.
"""
import json
import urllib.request
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from src.config import COCO_DIR, N_COCO_TEST, N_COCO_TRAIN, N_COCO_VAL, SEED

ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def download_annotations(coco_dir=COCO_DIR):
    """Download and unzip the caption annotations (253 MB zip, once). Returns the annotations folder."""
    folder = Path(coco_dir) / "annotations"
    if (folder / "captions_train2017.json").exists() and (folder / "captions_val2017.json").exists():
        return folder
    Path(coco_dir).mkdir(parents=True, exist_ok=True)
    archive = Path(coco_dir) / "annotations_trainval2017.zip"
    if not archive.exists():
        print("Downloading the COCO 2017 annotations (253 MB)...")
        urllib.request.urlretrieve(ANNOTATIONS_URL, archive)
    with zipfile.ZipFile(archive) as z:
        # The zip also has the object and keypoint annotations: only the captions are needed.
        z.extractall(coco_dir, members=[m for m in z.namelist() if "captions_" in m])
    return folder


def load_captions(split, coco_dir=COCO_DIR):
    """List of images of a COCO split ("train2017" or "val2017"), sorted by image id.

    Each image is a dict: id, file_name, url, captions (list of 5 to 7 strings).
    """
    data = json.loads((download_annotations(coco_dir) / f"captions_{split}.json").read_text())
    captions = defaultdict(list)
    for a in data["annotations"]:
        captions[a["image_id"]].append(a["caption"].strip())
    images = [{"id": im["id"], "split": split, "file_name": im["file_name"], "url": im["coco_url"],
               "captions": captions[im["id"]]} for im in data["images"] if captions[im["id"]]]
    # Sorted by id: the JSON order is not a promise, the sampling below must not depend on it.
    return sorted(images, key=lambda im: im["id"])


def coco_subsets(n_train=N_COCO_TRAIN, n_val=N_COCO_VAL, n_test=N_COCO_TEST, seed=SEED, coco_dir=COCO_DIR):
    """(train, val, test) lists of images, drawn at random with a fixed seed.

    train and val come from COCO train2017 and never share an image; test comes
    from COCO val2017, a separate set of images, so the test captions are never
    used to build the vocabulary or to choose anything.
    """
    rng = np.random.default_rng(seed)
    pool = load_captions("train2017", coco_dir)
    chosen = rng.choice(len(pool), n_train + n_val, replace=False)
    train = [pool[i] for i in sorted(chosen[:n_train])]
    val = [pool[i] for i in sorted(chosen[n_train:])]
    test_pool = load_captions("val2017", coco_dir)
    test = [test_pool[i] for i in sorted(rng.choice(len(test_pool), n_test, replace=False))]
    return train, val, test


def image_path(image, coco_dir=COCO_DIR):
    """Local path of an image of load_captions: data/coco/<split>/<file_name>."""
    return str(Path(coco_dir) / image["split"] / image["file_name"])


def _download_one(image, coco_dir, retries=3):
    path = Path(image_path(image, coco_dir))
    if path.exists() and path.stat().st_size > 0:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            # Written under a temporary name first: an interrupted download never leaves a broken jpg.
            tmp = path.with_suffix(".part")
            urllib.request.urlretrieve(image["url"], tmp)
            tmp.rename(path)
            return True
        except OSError:
            if attempt == retries - 1:
                raise
    return False


def download_images(images, coco_dir=COCO_DIR, workers=16):
    """Download the missing images (already present ones are skipped). Returns the local paths."""
    with ThreadPoolExecutor(workers) as ex:
        n_new = sum(ex.map(lambda im: _download_one(im, coco_dir), images))
    if n_new:
        print(f"{n_new} images downloaded, {len(images) - n_new} already on disk.")
    return [image_path(im, coco_dir) for im in images]
