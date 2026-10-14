"""Efficiency benchmark of trained models: accuracy on a fixed image set, inference speed, memory, size.

    python -m leyenda.benchmark '<json>'            # on the GPU
    CUDA_VISIBLE_DEVICES="" python -m leyenda.benchmark '<json>'   # same models on the CPU

json = {"models": [{"label": ..., "run_dir": ...}, ...], "images_csv": ..., "out": ...}
Every model is scored on the same images (`images_csv`: path,label,category) so results are comparable.
"""

import json
import sys
import time
from pathlib import Path


def benchmark(spec: dict):
    from . import setup
    setup(verbose=False)

    import keras
    import numpy as np
    import pandas as pd
    import tensorflow as tf

    from .data import make_dataset
    from .evaluation import binary_metrics, predict

    gpu = bool(tf.config.list_physical_devices("GPU"))
    images = pd.read_csv(spec["images_csv"])
    rows = []
    for m in spec["models"]:
        run_dir = Path(m["run_dir"])
        cfg = json.loads((run_dir / "config.json").read_text())
        summ = json.loads((run_dir / "summary.json").read_text())
        size = cfg["img_size"]
        if gpu:
            tf.config.experimental.reset_memory_stats("GPU:0")
        model = keras.models.load_model(run_dir / "model.keras")

        probs = predict(model, make_dataset(images, size, 64))  # accuracy on the fixed image set
        batch = next(iter(make_dataset(images.head(64), size, 64)))[0]
        model.predict(batch, verbose=0)  # warm-up (graph building)
        n_rep = 20 if gpu else 3
        t0 = time.perf_counter()
        for _ in range(n_rep):
            model.predict(batch, verbose=0)
        img_per_s = n_rep * len(batch) / (time.perf_counter() - t0)

        rows.append({
            "label": m["label"], "run": f"{run_dir.parent.name}/{run_dir.name}", "img_size": size,
            "params": model.count_params(),
            "trainable_params": int(sum(np.prod(w.shape) for w in model.trainable_weights)),
            "file_mb": round((run_dir / "model.keras").stat().st_size / 1e6, 1),
            "epochs": summ["epochs_run"], "train_time_min": round(summ["train_time_s"] / 60, 1) if summ.get("train_time_s") else None,
            "train_peak_gpu_mb": summ.get("peak_gpu_mb"),
            "device": "GPU" if gpu else "CPU", "inference_img_per_s": round(img_per_s, 1),
            "inference_peak_gpu_mb": round(tf.config.experimental.get_memory_info("GPU:0")["peak"] / 2**20, 1) if gpu else None,
            **{f"fixed_{k}": v for k, v in binary_metrics(images["label"], probs).items()},
            "fixed_errors": int(((probs >= 0.5) != (images["label"] == 1)).sum()),
        })
        print(f"[{m['label']}] {rows[-1]['fixed_errors']} errors, {img_per_s:.0f} img/s on {rows[-1]['device']}", flush=True)
        keras.backend.clear_session()
    pd.DataFrame(rows).to_csv(spec["out"], index=False)


if __name__ == "__main__":
    benchmark(json.loads(sys.argv[1]))
