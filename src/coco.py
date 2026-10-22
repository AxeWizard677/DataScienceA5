"""D3 (captioning) data: MS COCO 2014 captions read from the zips, and a seeded subset.

Principle: the 13.5 GB image zip is never extracted. We read the captions from the
small annotations zip, choose a subset of image ids, and copy only these images out of
the big zip. Usage from the repo root: `from src.coco import load_captions`.
"""
import json
import random
import zipfile
from pathlib import Path

from src.config import COCO_DIR, SEED

ANNOTATIONS_ZIP = "annotations_trainval2014.zip"
IMAGES_ZIP = "train2014.zip"
CAPTIONS_MEMBER = "annotations/captions_train2014.json"   # path of the file inside the zip


def image_filename(image_id):
    """COCO 2014 file name of an image: 'COCO_train2014_' + the id on 12 digits + '.jpg'."""
    return "COCO_train2014_%012d.jpg" % image_id


def load_captions(coco_dir=COCO_DIR):
    """{image_id: [caption, ...]} for every image of COCO 2014 train, read inside the zip."""
    # z.open reads the one file we need in the zip (66 MB), nothing is extracted on disk.
    with zipfile.ZipFile(Path(coco_dir) / ANNOTATIONS_ZIP) as z, z.open(CAPTIONS_MEMBER) as f:
        annotations = json.load(f)["annotations"]
    captions = {}
    for a in annotations:
        # An image has several captions (5 or a bit more): we group them by image id.
        # strip(): removes the spaces and line breaks around a caption, nothing else.
        captions.setdefault(a["image_id"], []).append(a["caption"].strip())
    return captions


def choose_subset(image_ids, n, seed=SEED):
    """The same `n` image ids for everybody: a seeded shuffle of all the ids, first `n` kept.

    The subset of size 2000 is the beginning of the subset of size 6000, so scaling up
    later only adds images and never replaces the ones already used (features, split).
    """
    # sorted(): the order of the JSON file is not guaranteed; sorting makes the result
    # depend only on the set of ids and on the seed.
    ids = sorted(image_ids)
    random.Random(seed).shuffle(ids)   # a private generator: does not touch the global random state
    return ids[:n]


def save_subset_ids(image_ids, coco_dir=COCO_DIR):
    """Write the chosen ids to coco_dir/subset_image_ids.json (a record, to compare between teammates)."""
    path = Path(coco_dir) / "subset_image_ids.json"
    path.write_text(json.dumps(list(image_ids)))
    return path


def extract_images(image_ids, coco_dir=COCO_DIR):
    """Copy only the images of `image_ids` from train2014.zip to coco_dir/train2014/.

    Returns the list of the image paths, in the order of `image_ids`.
    Images already copied are skipped, so the function can be run again.
    """
    out_dir = Path(coco_dir) / "train2014"
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(Path(coco_dir) / IMAGES_ZIP) as z:
        for image_id in image_ids:
            name = image_filename(image_id)
            dest = out_dir / name
            if not dest.exists():
                # Written under a temporary name then renamed: a copy interrupted in the middle
                # can never leave a half-written image that would be taken as valid next time.
                tmp = dest.with_name(name + ".part")
                tmp.write_bytes(z.read("train2014/" + name))
                tmp.replace(dest)
    return [str(out_dir / image_filename(i)) for i in image_ids]
