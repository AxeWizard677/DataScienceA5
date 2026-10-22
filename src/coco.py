"""D3 (captioning) data: MS COCO 2014 captions read from the zips, and a seeded subset of images.

The 13.5 GB image zip is never extracted: only the chosen images are copied out of it.
"""
import json
import random
import zipfile
from pathlib import Path

from src.config import COCO_DIR, SEED

ANNOTATIONS_ZIP = "annotations_trainval2014.zip"
IMAGES_ZIP = "train2014.zip"
CAPTIONS_MEMBER = "annotations/captions_train2014.json"   # path of the captions file inside the zip


def image_filename(image_id):
    """COCO 2014 file name of an image, e.g. COCO_train2014_000000270070.jpg."""
    return "COCO_train2014_%012d.jpg" % image_id


def load_captions(coco_dir=COCO_DIR):
    """{image_id: [caption, ...]} for the whole COCO 2014 train set, read inside the zip."""
    with zipfile.ZipFile(Path(coco_dir) / ANNOTATIONS_ZIP) as z, z.open(CAPTIONS_MEMBER) as f:
        annotations = json.load(f)["annotations"]
    captions = {}
    for a in annotations:
        # the file has one entry per caption: we group them by image
        captions.setdefault(a["image_id"], []).append(a["caption"].strip())
    return captions


def choose_subset(image_ids, n, seed=SEED):
    """The same `n` image ids for everybody: a seeded shuffle of all the ids, first `n` kept.

    A bigger subset starts with the smaller one, so scaling up only adds images.
    """
    ids = sorted(image_ids)           # sorted: the result must not depend on the order of the input
    random.Random(seed).shuffle(ids)
    return ids[:n]


def split_ids(image_ids):
    """(train, val, test) image ids, 80/10/10, from the ordered list given by choose_subset.

    The split is by image: all the captions of an image stay in the same set. The image at
    position i goes to val if i % 10 == 8, to test if i % 10 == 9, to train otherwise, so the
    images already used keep their set when the subset grows. Do not sort the list before.
    """
    train, val, test = [], [], []
    for position, image_id in enumerate(image_ids):
        if position % 10 == 8:
            val.append(image_id)
        elif position % 10 == 9:
            test.append(image_id)
        else:
            train.append(image_id)
    return train, val, test


def save_subset_ids(image_ids, coco_dir=COCO_DIR):
    """Write the ids to coco_dir/subset_image_ids.json, to compare the subset between teammates."""
    path = Path(coco_dir) / "subset_image_ids.json"
    path.write_text(json.dumps(list(image_ids)))
    return path


def extract_images(image_ids, coco_dir=COCO_DIR):
    """Copy the images of `image_ids` from train2014.zip to coco_dir/train2014/ and return their paths.

    Images already copied are skipped, so the function can be run again.
    """
    out_dir = Path(coco_dir) / "train2014"
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(Path(coco_dir) / IMAGES_ZIP) as z:
        for image_id in image_ids:
            name = image_filename(image_id)
            dest = out_dir / name
            if not dest.exists():
                # temporary name, then rename: an interrupted copy never leaves a broken image
                tmp = dest.with_name(name + ".part")
                tmp.write_bytes(z.read("train2014/" + name))
                tmp.replace(dest)
    return [str(out_dir / image_filename(i)) for i in image_ids]
