"""Shared parameters, defined in ONE place only."""

IMG_SIZE = 128      # side of the (square) images, RGB
BATCH_SIZE = 16     # 16 by default: some machines in the group only have 2 GB of VRAM
SEED = 42           # single seed for the split and for the test-set noise
NOISE_STD = 0.1     # standard deviation of the Gaussian noise, on images in [0, 1]
NOISE_RANGE = (0.05, 0.3)   # (low, high) of the noise levels when the models are trained on several levels
PHOTO_DIR = "data/Datasets/Photo"   # photo folder (not versioned)
DATA_DIR = "data/Datasets"          # root of the dataset; manifest paths are relative to it
MANIFEST_PATH = "protocol/manifest.csv"  # one row per image, with the split shared by D1/D2/D3

# D3 (captioning, MS COCO 2017)
COCO_DIR = "data/coco"      # annotations/ and the downloaded images (not versioned)
CAPTION_IMG_SIZE = 299      # input size of InceptionV3
N_COCO_TRAIN = 12000        # images drawn from COCO train2017 for training (5 captions each)
N_COCO_VAL = 1000           # other images of train2017, for validation (early stopping)
N_COCO_TEST = 1000          # images drawn from COCO val2017: never seen during development
