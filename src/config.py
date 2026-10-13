"""Shared parameters, defined in ONE place only (see CLAUDE.md)."""

IMG_SIZE = 128      # side of the (square) images, RGB
BATCH_SIZE = 16     # 16 by default: some machines in the group only have 2 GB of VRAM
SEED = 42           # single seed for the split and for the test-set noise
NOISE_STD = 0.1     # standard deviation of the Gaussian noise, on images in [0, 1]
PHOTO_DIR = "data/Datasets/Photo"   # photo folder (not versioned)
