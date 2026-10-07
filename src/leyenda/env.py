"""Environment setup shared by every notebook: reproducibility and GPU configuration."""

import os
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RUNS_DIR = PROJECT_ROOT / "runs"
REPORTS_DIR = PROJECT_ROOT / "reports"
for _d in (DATA_DIR / "raw", DATA_DIR / "processed", MODELS_DIR, RUNS_DIR, REPORTS_DIR / "figures"):
    _d.mkdir(parents=True, exist_ok=True)  # git-ignored folders: create them on a fresh clone


def _preload_cuda_libs():
    """Work around a TF 2.21 Linux wheel bug: its RPATH omits nvidia/cusolver/lib, so the GPU
    is silently disabled ("Cannot dlopen some GPU libraries"). Loading cusolver globally before
    TensorFlow is imported makes it resolvable. No-op on CPU-only installs or other platforms."""
    import ctypes
    import importlib.util

    spec = importlib.util.find_spec("nvidia")
    if spec is None or not spec.submodule_search_locations:
        return
    nvidia = Path(list(spec.submodule_search_locations)[0])
    for lib in ("nvjitlink/lib/libnvJitLink.so.12", "cusparse/lib/libcusparse.so.12",
                "cublas/lib/libcublasLt.so.12", "cublas/lib/libcublas.so.12",
                "cusolver/lib/libcusolver.so.11"):
        if (nvidia / lib).exists():
            try:
                ctypes.CDLL(str(nvidia / lib), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


_preload_cuda_libs()  # must run before the first `import tensorflow`


def setup(seed: int = 42, mixed_precision: bool = False, gpu: bool = True, verbose: bool = True):
    """Call once at the top of each notebook, before building any model.

    - fixes every random seed so runs are comparable in the training report
    - enables GPU memory growth (the T4-2Q vGPU only has 2 GB, TF would grab it all otherwise)
    - optionally enables float16 mixed precision (~2x less memory for activations)
    - gpu=False hides the GPU from this process (e.g. a notebook kernel that delegates training to
      child processes, see leyenda.train, and must not hold GPU memory itself)
    """
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import numpy as np
    import tensorflow as tf
    import keras

    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)

    if not gpu:
        tf.config.set_visible_devices([], "GPU")
    gpus = tf.config.get_visible_devices("GPU")
    for gpu_dev in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu_dev, True)
        except RuntimeError:
            pass  # already initialised: setting must happen before first use

    if mixed_precision and gpus:
        keras.mixed_precision.set_global_policy("mixed_float16")

    if verbose:
        print(f"TensorFlow {tf.__version__} | Keras {keras.__version__} | seed={seed}")
        print(f"GPU: {[g.name for g in gpus] or 'none (CPU only)'}"
              f" | precision policy: {keras.mixed_precision.global_policy().name}")
    return gpus
