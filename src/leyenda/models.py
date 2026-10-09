"""Model architectures for Deliverable 1 (photo vs. other).

All models take raw uint8 RGB images (img_size x img_size x 3) and output P(photo) through a
sigmoid. Pre-processing (rescaling, augmentation) lives *inside* the model so the exported
.keras file can be used in production on raw images without extra code.
"""

import keras
from keras import layers, regularizers


def augmentation(seed: int = 42, grayscale: float = 0.0) -> keras.Sequential:
    """Only transformations that keep the class: a flipped/rotated/cropped photo is still a photo.
    (No colour jitter: colour statistics are one of the cues separating photos from paintings.)
    grayscale > 0 converts that fraction of training images to black & white, so the model cannot use
    "has colour => photo" as a shortcut and still recognises old black & white photographs."""
    steps = [
        layers.RandomFlip("horizontal", seed=seed),
        layers.RandomRotation(0.05, seed=seed),
        layers.RandomZoom(0.15, seed=seed),
        layers.RandomTranslation(0.1, 0.1, seed=seed),
    ]
    if grayscale:
        steps.append(layers.RandomGrayscale(factor=grayscale, seed=seed))
    return keras.Sequential(steps, name="augmentation")


def _head(x, dropout: float = 0.0):
    if dropout:
        x = layers.Dropout(dropout)(x)
    # float32 output keeps the sigmoid numerically stable under mixed precision
    return layers.Dense(1, activation="sigmoid", dtype="float32", name="p_photo")(x)


def mlp(img_size: int = 160, input_size: int = 64, hidden=(512, 128)) -> keras.Model:
    """E1 baseline: fully-connected network on flattened downsampled pixels (no spatial prior)."""
    inp = keras.Input((img_size, img_size, 3), name="image")
    x = layers.Resizing(input_size, input_size)(inp)
    x = layers.Rescaling(1 / 255)(x)
    x = layers.Flatten()(x)
    for units in hidden:
        x = layers.Dense(units, activation="relu")(x)
    return keras.Model(inp, _head(x), name="mlp")


def simple_cnn(img_size: int = 160, filters=(32, 64, 128, 256)) -> keras.Model:
    """E2/E3: plain CNN, no regularisation. Conv blocks extract local patterns (edges, textures),
    pooling makes them translation-tolerant; a dense layer combines them."""
    inp = keras.Input((img_size, img_size, 3), name="image")
    x = layers.Rescaling(1 / 255)(inp)
    for f in filters:
        x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    return keras.Model(inp, _head(x), name="simple_cnn")


def regularized_cnn(img_size: int = 160, filters=(32, 64, 128, 256), l2: float = 1e-4,
                    dropout: float = 0.4) -> keras.Model:
    """E4: same depth as simple_cnn plus the standard variance-reduction toolbox:
    data augmentation, batch normalisation, L2 weight decay, global average pooling
    (far fewer parameters than Flatten+Dense) and dropout."""
    reg = regularizers.l2(l2)
    inp = keras.Input((img_size, img_size, 3), name="image")
    x = augmentation()(inp)
    x = layers.Rescaling(1 / 255)(x)
    for f in filters:
        for _ in range(2):
            x = layers.Conv2D(f, 3, padding="same", use_bias=False, kernel_regularizer=reg)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    return keras.Model(inp, _head(x, dropout), name="regularized_cnn")


def transfer_mobilenet(img_size: int = 160, dropout: float = 0.3, trainable: bool = False, grayscale: float = 0.0):
    """E5/E6: MobileNetV2 pre-trained on ImageNet (1.4M natural photos) as a frozen feature extractor.
    Its features already encode 'natural image statistics' that separate photos from paintings.
    Returns (model, base) so the base can be partially unfrozen for fine-tuning."""
    base = keras.applications.MobileNetV2(input_shape=(img_size, img_size, 3), include_top=False,
                                          weights="imagenet")
    base.trainable = trainable
    inp = keras.Input((img_size, img_size, 3), name="image")
    x = augmentation(grayscale=grayscale)(inp)
    x = layers.Rescaling(1 / 127.5, offset=-1)(x)  # MobileNetV2 expects [-1, 1]
    x = base(x, training=False)  # keep BatchNorm in inference mode, even when fine-tuning
    x = layers.GlobalAveragePooling2D()(x)
    return keras.Model(inp, _head(x, dropout), name="mobilenetv2_transfer"), base


def unfreeze_top(base: keras.Model, n_layers: int = 40):
    """Fine-tuning: unfreeze only the last n layers (high-level, task-specific features);
    the first layers (generic edges/colours) stay frozen. BatchNorm layers stay frozen."""
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False


def compile_binary(model: keras.Model, lr: float = 1e-3, jit_compile: bool = False,
                   label_smoothing: float = 0.0) -> keras.Model:
    """Binary cross-entropy = negative log-likelihood of a Bernoulli output: the natural loss for a
    sigmoid classifier. Adam adapts the step size per parameter, a robust default for CNNs.

    jit_compile=False: XLA's convolution autotuner needs scratch GPU memory the 2 GB vGPU does not
    have ("Autotuning failed ... No valid config found"); plain cuDNN kernels work fine."""
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        # label_smoothing > 0 trains towards 1-e/2 and e/2 instead of 1 and 0: less over-confidence on
        # the mislabelled images found in the archives (photos filed as schematics/paintings)
        loss=keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing),
        metrics=["accuracy", keras.metrics.AUC(name="auc"),
                 keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall")],
        jit_compile=jit_compile,
    )
    return model


def finetune_mobilenet(img_size: int = 160, from_model: str = "", n_layers: int = 40) -> keras.Model:
    """E6: reload a trained frozen-MobileNetV2 model (E5) and unfreeze its top `n_layers`."""
    model = keras.models.load_model(from_model)
    base = next(l for l in model.layers if isinstance(l, keras.Model) and l.name.startswith("mobilenet"))
    unfreeze_top(base, n_layers)
    return model


# name -> builder, so experiments can be described as plain (JSON-serialisable) specs
BUILDERS = {
    "mlp": mlp,
    "simple_cnn": simple_cnn,
    "regularized_cnn": regularized_cnn,
    "mobilenet_frozen": lambda **kw: transfer_mobilenet(**kw)[0],
    "mobilenet_finetune": finetune_mobilenet,
}
