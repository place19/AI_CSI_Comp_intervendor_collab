"""MLflow logging: windowed-mean metrics + per-run artifacts + note."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow


class _Window:
    """Accumulates scalar values; on flush returns their mean and resets."""

    def __init__(self):
        self.sum: float = 0.0
        self.count: int = 0

    def add(self, v: float) -> None:
        self.sum += float(v)
        self.count += 1

    def flush(self) -> Optional[float]:
        if self.count == 0:
            return None
        m = self.sum / self.count
        self.sum = 0.0
        self.count = 0
        return m


class MLflowLogger:
    """Lightweight wrapper around mlflow that integrates with the Trainer callback API.

    Behaviour:
      • Starts (or attaches to) an mlflow run on `__enter__` / `start_run`.
      • Train-step metrics: accumulate; flush every `log_every_n_iters` steps as windowed means.
      • Validation metrics: log directly (no averaging).
      • Artifacts: log_artifact(path) → uploads to the active run.
      • Note: set_note(markdown) sets `mlflow.note.content` on the run.
    """

    def __init__(
        self,
        tracking_uri: str,
        experiment_name: str,
        log_every_n_iters: int,
        run_name: Optional[str] = None,
    ):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.log_every_n_iters = int(log_every_n_iters)
        self.run_name = run_name
        self._windows: dict[str, _Window] = defaultdict(_Window)
        self._run = None

    # ----- lifecycle -----

    def start_run(self) -> "MLflowLogger":
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        self._run = mlflow.start_run(run_name=self.run_name)
        return self

    def end_run(self) -> None:
        # Flush whatever's left, in case anything is hanging.
        for k, w in list(self._windows.items()):
            m = w.flush()
            if m is not None:
                mlflow.log_metric(k, m)
        mlflow.end_run()
        self._run = None

    def __enter__(self):
        return self.start_run()

    def __exit__(self, exc_type, exc, tb):
        self.end_run()
        return False

    # ----- logging API -----

    def log_step(self, step: int, metrics: Dict[str, float]) -> None:
        for k, v in metrics.items():
            self._windows[k].add(v)
        if self.log_every_n_iters > 0 and step % self.log_every_n_iters == 0:
            self._flush(step)

    def _flush(self, step: int) -> None:
        for k, w in list(self._windows.items()):
            m = w.flush()
            if m is not None:
                mlflow.log_metric(k, m, step=step)

    def log_val(self, step: int, metrics: Dict[str, float]) -> None:
        for k, v in metrics.items():
            mlflow.log_metric(k, v, step=step)

    def log_artifact(self, path: Path | str) -> None:
        mlflow.log_artifact(str(path))

    def set_note(self, markdown: str) -> None:
        mlflow.set_tag("mlflow.note.content", markdown)

    def log_params(self, params: Dict[str, Any]) -> None:
        # mlflow.log_params requires scalar/string values
        flat = {k: str(v) if not isinstance(v, (int, float, str, bool)) else v
                for k, v in params.items()}
        mlflow.log_params(flat)


class MLflowCallback:
    """Trainer callback that funnels train/val metrics through MLflowLogger."""

    def __init__(self, logger: MLflowLogger):
        self.logger = logger

    def on_train_begin(self, trainer): ...
    def on_train_end(self, trainer): ...
    def on_epoch_begin(self, trainer, epoch): ...
    def on_epoch_end(self, trainer, epoch, train_metrics): ...

    def on_train_step_end(self, trainer, step, metrics):
        self.logger.log_step(step, metrics)

    def on_val_end(self, trainer, epoch, val_metrics):
        # Validation runs once at the end of each epoch; the x-axis users care
        # about for val curves is the epoch number, not the iteration count.
        self.logger.log_val(int(epoch) + 1, val_metrics)
