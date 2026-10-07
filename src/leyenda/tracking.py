"""Lightweight experiment tracking for the failure -> improvement training report.

Every training attempt is a `Run`, stored under runs/<deliverable>/<NN>_<name>/:

    config.json     hyper-parameters + the hypothesis being tested
    history.csv     per-epoch loss/metrics (written live by CSVLogger)
    summary.json    final metrics, verdict (failure / improvement / baseline) and lesson learned
    curves.png      train vs validation curves
    model.png       architecture diagram (needs Graphviz `dot` on PATH)
    tb/             TensorBoard logs

Typical use in a notebook:

    run = Run("d1_classifier", "baseline_cnn",
              hypothesis="A small 3-block CNN is enough to separate photos from scanned text",
              config={"lr": 1e-3, "batch_size": 32, "epochs": 20})
    history = model.fit(..., callbacks=run.callbacks())
    run.finish(model, history, metrics=model.evaluate(val_ds, return_dict=True))
    run.annotate("failure", lesson="Overfits after epoch 4 -> add augmentation + dropout")

    training_journal("d1_classifier")   # markdown story of all attempts, in order
"""

import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

from .env import RUNS_DIR

VERDICTS = ("pending", "baseline", "failure", "improvement", "best")


class Run:
    """A named training attempt. Re-creating a Run with the same name reuses its folder, so a
    notebook can be re-executed without retraining: check `run.done`, then `run.load_model()`."""

    def __init__(self, deliverable: str, name: str, hypothesis: str, config: dict | None = None):
        self.root = RUNS_DIR / deliverable
        self.root.mkdir(parents=True, exist_ok=True)
        existing = sorted(self.root.glob(f"[0-9][0-9]_{name}"))
        if existing:
            self.dir = existing[-1]
            index = int(self.dir.name[:2])
        else:
            index = len([p for p in self.root.iterdir() if p.is_dir()]) + 1
            self.dir = self.root / f"{index:02d}_{name}"
            self.dir.mkdir()
        self.name = name
        self.config = {
            "name": name,
            "index": index,
            "started": datetime.now().isoformat(timespec="seconds"),
            "hypothesis": hypothesis,
            **(config or {}),
        }
        if not self.done:
            (self.dir / "config.json").write_text(json.dumps(self.config, indent=2, default=str))

    @property
    def done(self) -> bool:
        """Training finished and the model was saved."""
        return (self.dir / "summary.json").exists() and (self.dir / "model.keras").exists()

    def load_model(self):
        import keras
        return keras.models.load_model(self.dir / "model.keras")

    def summary(self) -> dict:
        path = self.dir / "summary.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def annotate(self, verdict: str, lesson: str = ""):
        """Record the verdict once the results have been analysed (can be called on a reused run)."""
        if verdict not in VERDICTS:
            raise ValueError(f"verdict must be one of {VERDICTS}")
        summ = self.summary()
        summ.update(verdict=verdict, lesson=lesson)
        (self.dir / "summary.json").write_text(json.dumps(summ, indent=2))

    # ---- checkpointing -----------------------------------------------------------------------------
    @property
    def _ckpt(self):
        return self.dir / "checkpoints"

    def resume_point(self):
        """(last completed epoch index, path of the model saved at that epoch) or None if starting fresh."""
        state = self._ckpt / "state.json"
        if state.exists() and (self._ckpt / "last.keras").exists():
            return json.loads(state.read_text())["epoch"], self._ckpt / "last.keras"
        return None

    def callbacks(self, early_stopping: bool = False, patience: int = 5, monitor: str = "val_loss",
                  reduce_lr: bool = False):
        """Keras callbacks for a run. Checkpointing is always on (see _atomic_checkpoint):
        checkpoints/last.keras after every epoch (to resume after a crash) and checkpoints/best.keras
        whenever `monitor` improves. Resuming = load last.keras and call fit(initial_epoch=...)."""
        import shutil
        import keras

        self._ckpt.mkdir(exist_ok=True)
        point = self.resume_point()
        history_csv = self.dir / "history.csv"
        best = float("inf")
        if point is None:  # fresh start: drop anything left by an older attempt of this run
            history_csv.unlink(missing_ok=True)
            shutil.rmtree(self.dir / "tb", ignore_errors=True)
            (self._ckpt / "best.keras").unlink(missing_ok=True)
        else:
            epoch, _ = point
            print(f"[{self.dir.name}] resuming after epoch {epoch + 1} from checkpoints/last.keras", flush=True)
            if history_csv.exists():  # drop rows logged after the last checkpoint (they will be redone)
                hist = pd.read_csv(history_csv)
                hist = hist[hist["epoch"] <= epoch]
                hist.to_csv(history_csv, index=False)
                if monitor in hist and len(hist):
                    best = float(hist[monitor].min())  # best score from *before* the crash

        cbs = [
            keras.callbacks.CSVLogger(str(history_csv), append=True),  # continues across restarts
            keras.callbacks.TensorBoard(log_dir=str(self.dir / "tb")),
        ]
        if reduce_lr:  # halve the learning rate when the monitored loss stalls: damps noisy, oscillating training
            cbs.append(keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=3, min_lr=1e-5,
                                                         verbose=1))
        if early_stopping:
            # best weights are restored from checkpoints/best.keras after fit (works across restarts too)
            cbs.append(keras.callbacks.EarlyStopping(monitor=monitor, patience=patience))
        cbs.append(_atomic_checkpoint(self._ckpt, monitor, best))  # last: runs after CSVLogger has written
        return cbs

    def best_model(self):
        """Best model of this run so far (lowest monitored validation loss)."""
        import keras
        return keras.models.load_model(self._ckpt / "best.keras")

    def finish(self, model=None, history=None, metrics: dict | None = None,
               verdict: str = "pending", lesson: str = "", show: bool = True):
        if verdict not in VERDICTS:
            raise ValueError(f"verdict must be one of {VERDICTS}")
        # the CSV log covers every epoch even if training was resumed (History only has the last segment)
        hist = self.history().drop(columns=["epoch"], errors="ignore")
        if hist.empty and history is not None:
            hist = pd.DataFrame(history.history)

        summary = {
            "verdict": verdict,
            "lesson": lesson,
            "finished": datetime.now().isoformat(timespec="seconds"),
            "epochs_run": len(hist),
            "eval_metrics": {k: float(v) for k, v in (metrics or {}).items()},
        }
        if not hist.empty:
            summary["final_epoch"] = {k: float(v) for k, v in hist.iloc[-1].items()}
            if "val_loss" in hist:
                summary["best_val_loss_epoch"] = int(hist["val_loss"].idxmin()) + 1
                # train/val gap at the end = quick overfitting (variance) indicator
                summary["generalization_gap"] = float(hist["val_loss"].iloc[-1] - hist["loss"].iloc[-1])
        if model is not None:
            summary["params"] = int(model.count_params())
            model.save(self.dir / "model.keras")
            for f in ("last.keras", "state.json"):  # training complete: resume point no longer needed
                (self._ckpt / f).unlink(missing_ok=True)
            try:
                import keras
                keras.utils.plot_model(model, to_file=str(self.dir / "model.png"),
                                       show_shapes=True, show_layer_names=True, dpi=80)
            except Exception:  # Graphviz not installed: fall back to the matplotlib diagram
                from .evaluation import plot_architecture
                plot_architecture(model, save_to=self.dir / "model.png", show=False)

        (self.dir / "summary.json").write_text(json.dumps(summary, indent=2))
        if not hist.empty:
            plot_curves(hist, title=f"{self.dir.name} [{verdict}]",
                        save_to=self.dir / "curves.png", show=show)
        return summary

    def history(self) -> pd.DataFrame:
        path = self.dir / "history.csv"
        return pd.read_csv(path) if path.exists() else pd.DataFrame()


def plot_curves(hist: pd.DataFrame, title: str = "", save_to=None, show: bool = True):
    """Train vs validation curves for every metric in the history (loss, accuracy, ...)."""
    import matplotlib.pyplot as plt

    metrics = [c for c in hist.columns if not c.startswith("val_") and c not in ("epoch", "learning_rate")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 3.5), squeeze=False)
    epochs = range(1, len(hist) + 1)
    for ax, m in zip(axes[0], metrics):
        ax.plot(epochs, hist[m], label="train")
        if f"val_{m}" in hist:
            ax.plot(epochs, hist[f"val_{m}"], label="validation")
        ax.set_title(m)
        ax.set_xlabel("epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    if save_to:
        fig.savefig(save_to, dpi=110)
    if show:
        plt.show()
    else:
        plt.close(fig)


def _load_runs(deliverable: str):
    root = RUNS_DIR / deliverable
    for d in sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []:
        cfg = json.loads((d / "config.json").read_text())
        summ = json.loads((d / "summary.json").read_text()) if (d / "summary.json").exists() else {}
        yield d, cfg, summ


def compare_runs(deliverable: str) -> pd.DataFrame:
    """One row per run: hypothesis, verdict, key final metrics. Ready for display in the report."""
    rows = []
    for d, cfg, summ in _load_runs(deliverable):
        row = {"run": d.name, "hypothesis": cfg.get("hypothesis", ""),
               "verdict": summ.get("verdict", "unfinished"), "epochs": summ.get("epochs_run"),
               "params": summ.get("params"), "gap": summ.get("generalization_gap")}
        row.update({f"final_{k}": v for k, v in summ.get("final_epoch", {}).items()
                    if k.startswith("val_")})
        row.update({k: v for k, v in summ.get("eval_metrics", {}).items()})
        row["lesson"] = summ.get("lesson", "")
        rows.append(row)
    return pd.DataFrame(rows).set_index("run") if rows else pd.DataFrame()


def plot_runs(deliverable: str, metric: str = "val_loss", runs: list[str] | None = None,
              save_to=None):
    """Overlay one validation metric across runs: the core 'progress' figure of the report."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for d, _, summ in _load_runs(deliverable):
        if runs and d.name not in runs or not (d / "history.csv").exists():
            continue
        hist = pd.read_csv(d / "history.csv")
        if metric in hist:
            style = "--" if summ.get("verdict") == "failure" else "-"
            ax.plot(range(1, len(hist) + 1), hist[metric], style,
                    label=f"{d.name} ({summ.get('verdict', '?')})")
    ax.set_xlabel("epoch")
    ax.set_ylabel(metric)
    ax.set_title(f"{deliverable}: {metric} across experiments (dashed = failure)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_to:
        fig.savefig(save_to, dpi=110)
    plt.show()


def training_journal(deliverable: str, save: bool = True) -> str:
    """Chronological markdown log: hypothesis -> result -> verdict -> lesson, for every run."""
    lines = [f"# Training journal: {deliverable}", ""]
    for d, cfg, summ in _load_runs(deliverable):
        verdict = summ.get("verdict", "unfinished")
        lines.append(f"## {d.name}: **{verdict.upper()}**")
        lines.append(f"- **Hypothesis:** {cfg.get('hypothesis', '')}")
        params = {k: v for k, v in cfg.items() if k not in ("name", "index", "started", "hypothesis")}
        if params:
            lines.append(f"- **Config:** `{json.dumps(params, default=str)}`")
        if summ.get("eval_metrics"):
            lines.append("- **Evaluation:** " + ", ".join(f"{k}={v:.4f}" for k, v in summ["eval_metrics"].items()))
        if summ.get("generalization_gap") is not None:
            lines.append(f"- **Train/val loss gap:** {summ['generalization_gap']:.4f}"
                         f" (best val epoch {summ.get('best_val_loss_epoch')}/{summ.get('epochs_run')})")
        if summ.get("lesson"):
            lines.append(f"- **Lesson / next step:** {summ['lesson']}")
        lines.append("")
    text = "\n".join(lines)
    if save:
        from .env import REPORTS_DIR
        (REPORTS_DIR / f"journal_{deliverable}.md").write_text(text)
    return text


def _atomic_save(model, path: Path):
    """Write to a temporary file, then rename: a crash mid-write can never leave a corrupt checkpoint."""
    tmp = path.with_name(path.stem + ".tmp.keras")
    model.save(tmp)
    os.replace(tmp, path)


def _atomic_checkpoint(ckpt_dir: Path, monitor: str, best: float):
    """After every epoch: save last.keras (model + optimizer state) and state.json (epoch index);
    when the monitored metric improves, also save best.keras. All writes are atomic."""
    import keras

    class AtomicCheckpoint(keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.best = best

        def on_epoch_end(self, epoch, logs=None):
            value = (logs or {}).get(monitor)
            if value is not None and value < self.best:
                self.best = float(value)
                _atomic_save(self.model, ckpt_dir / "best.keras")
            _atomic_save(self.model, ckpt_dir / "last.keras")
            tmp = ckpt_dir / "state.json.tmp"
            tmp.write_text(json.dumps({"epoch": epoch, "best": self.best}))
            os.replace(tmp, ckpt_dir / "state.json")

    return AtomicCheckpoint()
