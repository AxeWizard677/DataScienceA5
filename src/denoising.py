"""D2 (denoising): the convolutional auto-encoder models.

Usage from the repo root: `from src.denoising import build_autoencoder`.
"""
import keras

from src.config import IMG_SIZE, SEED


def build_autoencoder():
    """Baseline convolutional auto-encoder: noisy image in, clean image out.

    Encoder: 3 x (Conv 3x3 + ReLU + MaxPool 2) with 32-64-128 filters, which
    compresses 128x128x3 into a 16x16x128 "bottleneck".
    Decoder: 3 x (Conv 3x3 + ReLU + UpSampling 2), then a Conv with 3 filters and a
    sigmoid to get back an RGB image in [0, 1].
    """
    # Same seed as the rest of the project: same initial weights at every run.
    keras.utils.set_random_seed(SEED)

    model = keras.Sequential([
        # No Rescaling layer: make_dataset already gives float32 in [0, 1].
        keras.Input((IMG_SIZE, IMG_SIZE, 3)),

        # Encoder. padding="same" keeps the size, only the pooling halves it.
        keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
        keras.layers.MaxPooling2D(2),    # 128 -> 64
        keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
        keras.layers.MaxPooling2D(2),    # 64 -> 32
        keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
        keras.layers.MaxPooling2D(2),    # 32 -> 16 (bottleneck)

        # Decoder, mirror of the encoder (128-64-32 filters).
        # UpSampling2D repeats pixels (no weights), the Conv before it learns the content.
        keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
        keras.layers.UpSampling2D(2),    # 16 -> 32
        keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
        keras.layers.UpSampling2D(2),    # 32 -> 64
        keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
        keras.layers.UpSampling2D(2),    # 64 -> 128

        # Sigmoid: the output stays in [0, 1], like the clean target image.
        keras.layers.Conv2D(3, 3, padding="same", activation="sigmoid"),
    ], name="autoencoder")

    # MSE: pixel-by-pixel difference between the output and the clean image.
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model
