"""Train one experiment in an isolated process.

    python -m leyenda.train '<json spec>'

Why a separate process: the 2 GB vGPU cannot release memory between models inside one Python
session (TensorFlow's allocator keeps and fragments it, and the async allocator is not supported on
vGPUs), so a second model OOMs. A fresh process per experiment gets the whole GPU and frees it on exit.

The spec is a dict: name, hypothesis, deliverable, builder (key of models.BUILDERS), builder_kwargs,
epochs, lr, batch_size, img_size, train_sources (optional), early_stopping, patience, + free config.
Results land in runs/<deliverable>/NN_<name>/, including predictions.npz (val + train-eval probs).
"""

import json
import sys


def train(spec: dict):
    from . import setup
    setup(seed=spec.get("seed", 42), mixed_precision=True)

    import numpy as np
    import pandas as pd

    from . import models as M
    from .data import class_weights, make_dataset, train_eval_subset
    from .env import DATA_DIR
    from .evaluation import binary_metrics, predict
    from .tracking import Run

    img, cache = spec["img_size"], DATA_DIR / "processed" / "tfcache"
    data = pd.read_csv(DATA_DIR / "processed" / spec.get("split_file", "d1_split.csv"))
    train_df, val_df = data[data["split"] == "train"], data[data["split"] == "val"]
    tag = "all"
    if spec.get("train_sources"):
        train_df = train_df[train_df["category"].isin(spec["train_sources"])]
        tag = "-".join(sorted(spec["train_sources"]))
    train_eval_df = train_eval_subset(train_df)

    config = {k: v for k, v in spec.items() if k not in ("name", "hypothesis", "deliverable")}
    run = Run(spec["deliverable"], spec["name"], spec["hypothesis"], config)

    import keras

    point = run.resume_point()
    if point is not None:  # crashed / interrupted earlier: continue from the last completed epoch
        last_epoch, last_model = point
        model, initial_epoch = keras.models.load_model(last_model), last_epoch + 1  # weights + optimizer state
    else:
        model = M.BUILDERS[spec["builder"]](img_size=img, **spec.get("builder_kwargs", {}))
        M.compile_binary(model, spec["lr"], label_smoothing=spec.get("label_smoothing", 0.0))
        initial_epoch = 0
    train_ds = make_dataset(train_df, img, spec["batch_size"], shuffle=True, cache=cache / f"train_{tag}_{img}")
    val_ds = make_dataset(val_df, img, 64, cache=cache / f"val_{img}")
    train_eval_ds = make_dataset(train_eval_df, img, 64, cache=cache / f"train_eval_{tag}_{img}")

    history = model.fit(train_ds, validation_data=val_ds, epochs=spec["epochs"], initial_epoch=initial_epoch,
                        class_weight=class_weights(train_df["label"]),
                        callbacks=run.callbacks(early_stopping=spec.get("early_stopping", False),
                                                patience=spec.get("patience", 5),
                                                reduce_lr=spec.get("reduce_lr", False)),
                        verbose=2)
    import tensorflow as tf
    peak_gpu_mb = (tf.config.experimental.get_memory_info("GPU:0")["peak"] / 2**20
                   if tf.config.list_physical_devices("GPU") else None)
    if spec.get("early_stopping", False):
        model = run.best_model()  # restore the best epoch (equivalent of restore_best_weights, across restarts)

    p_val, p_train = predict(model, val_ds), predict(model, train_eval_ds)
    np.savez(run.dir / "predictions.npz", val=p_val, train_eval=p_train,
             val_paths=val_df["path"].values, train_eval_paths=train_eval_df["path"].values)
    metrics = {**{f"val_{k}": v for k, v in binary_metrics(val_df["label"], p_val).items()},
               **{f"train_{k}": v for k, v in binary_metrics(train_eval_df["label"], p_train).items()}}
    run.finish(model, history, metrics=metrics, show=False)
    if peak_gpu_mb is not None:  # peak GPU memory of this training segment
        summ = run.summary()
        summ["peak_gpu_mb"] = round(peak_gpu_mb, 1)
        (run.dir / "summary.json").write_text(json.dumps(summ, indent=2))
    print(f"[{run.dir.name}] done: val_f1={metrics['val_f1']:.4f} val_auc={metrics['val_roc_auc']:.4f}", flush=True)


if __name__ == "__main__":
    train(json.loads(sys.argv[1]))
