import os
from pathlib import Path
import tempfile
from typing import Any, cast

from lightning.fabric.utilities.rank_zero import rank_zero_info, rank_zero_only, rank_zero_warn  # type: ignore
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback, DeviceStatsMonitor, EarlyStopping
from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning.pytorch.loggers import Logger, MLFlowLogger
from lightning.pytorch.trainer.states import TrainerFn
from lightning.pytorch.utilities.model_summary.model_summary import ModelSummary
from lightning.pytorch.utilities.types import STEP_OUTPUT
import mlflow
from mlflow import MlflowClient, MlflowException
from mlflow.entities.run import Run
from mlflow.entities.run_status import RunStatus
from mlflow.models import Model
from mlflow.pytorch import pickle_module as mlflow_pytorch_pickle_module
from torch.optim import Optimizer

from lit_mlflow.logger import DbxMLFlowLogger
from lit_mlflow.utils.dbx import get_databricks_tags


class MlFlowAutoCallback(Callback):
    def __init__(self, verbose: bool = True) -> None:
        self.supported_loggers = (MLFlowLogger, DbxMLFlowLogger)
        self.verbose = verbose
        self.logger: MLFlowLogger | DbxMLFlowLogger | None = None
        self.autologging_disabled = False

    @property
    def client(self) -> MlflowClient | None:
        if self.logger:
            return self.logger.experiment
        return None

    def _get_logger(self, loggers: list[Logger]) -> MLFlowLogger | DbxMLFlowLogger | None:
        if isinstance(loggers, list):
            if len(loggers) == 0:
                rank_zero_warn("Cannot log artifacts because Trainer has no logger.")
                return None
            else:
                for logger in loggers:
                    if isinstance(logger, self.supported_loggers):
                        return logger
                rank_zero_warn(
                    f"{self.__class__.__name__} does not support logging with {logger.__class__.__name__}."
                    f" Supported loggers are: {', '.join(str(x.__name__) for x in self.supported_loggers)}"
                )
            return None

    def _prevent_entry(self, trainer: "pl.Trainer") -> bool:
        return self.logger is None or not trainer.is_global_zero

    def _get_optimizer_name(self, optimizer: LightningOptimizer | Optimizer) -> str:
        return (
            str(optimizer._optimizer.__class__.__name__)
            if isinstance(optimizer, LightningOptimizer)
            else str(optimizer.__class__.__name__)
        )

    def _log_early_stop_params(self, early_stop_callback: EarlyStopping) -> None:
        """Logs early stopping configuration parameters to MLflow."""
        if self.logger is None:
            return None

        params = {
            p: getattr(early_stop_callback, p)
            for p in ["monitor", "mode", "patience", "min_delta", "stopped_epoch"]
            if hasattr(early_stop_callback, p)
        }
        self.logger.log_hyperparams(params)

    def _log_early_stop_metrics(self, early_stop_callback: EarlyStopping) -> None:
        """Logs early stopping behavior results (e.g. stopped epoch) as metrics to MLflow."""
        if self.logger is None:
            return None

        if early_stop_callback is None or early_stop_callback.stopped_epoch == 0:
            return None

        metrics: dict[str, float] = {
            "stopped_epoch": early_stop_callback.stopped_epoch,
            "restored_epoch": early_stop_callback.stopped_epoch - max(1, early_stop_callback.patience),
        }

        if hasattr(early_stop_callback, "best_score"):
            metrics["best_score"] = float(early_stop_callback.best_score)

        if hasattr(early_stop_callback, "wait_count"):
            metrics["wait_count"] = early_stop_callback.wait_count

        self.logger.log_metrics(metrics)

    def _resolve_early_stopping_callback(self, trainer: "pl.Trainer") -> EarlyStopping | None:
        if hasattr(trainer, "callbacks"):
            for callback in cast(list[Callback], trainer.callbacks):  # pyright: ignore[reportAttributeAccessIssue]
                if isinstance(callback, EarlyStopping):
                    return callback
        return None

    def _log_model_summary(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        summary = str(ModelSummary(pl_module, max_depth=-1))
        artifact_path = "model_summary.txt"
        if self.logger and self.logger._run_id and self.client:
            with tempfile.TemporaryDirectory(prefix="test", suffix="test", dir=Path.cwd()) as tmp_dir:
                with Path.open(Path(f"{tmp_dir}/{artifact_path}"), "w") as tmp_file_summary:
                    tmp_file_summary.write(summary)
                run_id = str(self.logger.run_id)
                self.client.log_artifacts(run_id, tmp_dir, artifact_path)

    def _log_model(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.logger and self.logger.run_id and self.client:
            rank_zero_info("Saving the model and uploading to MLFlow!")
            with tempfile.TemporaryDirectory(prefix="test", suffix="test", dir=Path.cwd()) as tmp_dir:
                local_path = Path(tmp_dir) / "model"
                artifact_path = "model"
                mlflow_model = Model(artifact_path=artifact_path, run_id=self.logger.run_id)
                mlflow.pytorch.save_model(
                    pytorch_model=pl_module,
                    path=local_path,
                    conda_env=None,
                    mlflow_model=mlflow_model,
                    code_paths=None,
                    pickle_module=mlflow_pytorch_pickle_module,
                    signature=None,
                    input_example=None,
                    requirements_file=None,
                    extra_files=None,
                    pip_requirements=None,
                    extra_pip_requirements=None,
                )
                self.client.log_artifacts(
                    run_id=self.logger.run_id,
                    local_dir=tmp_dir,
                    artifact_path=artifact_path,
                )
                try:
                    self.client._record_logged_model(run_id=self.logger.run_id, mlflow_model=mlflow_model)
                except MlflowException:
                    rank_zero_warn(
                        f"Logging model metadata to the tracking server {self.logger._tracking_uri} has failed"
                    )

            # info = mlflow_model.get_model_info()
            return None

    def _print_auto_logged_info(self) -> None:
        if self.logger and self.logger.run_id and self.client:
            run = mlflow.get_run(run_id=self.logger.run_id)
            if run:
                artifacts = [f.path for f in self.client.list_artifacts(run.info.run_id, "model")]
                tags = {k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.")}
                rank_zero_info(f"run_id: {run.info.run_id}")
                rank_zero_info(f"artifacts: {artifacts}")
                rank_zero_info(f"params: {run.data.params}")
                rank_zero_info(f"metrics: {run.data.metrics}")
                rank_zero_info(f"tags: {tags}")

    def _log_cluster_tags(self) -> None:
        tags = get_databricks_tags()
        if self.logger and self.logger.run_id and self.client:
            for tag, value in tags.items():
                self.client.set_tag(self.logger.run_id, key=tag, value=value)

    def _patch_device_stats_monitor(self, trainer: "pl.Trainer") -> None:
        def _patched_prefix_metric_keys(
            metrics_dict: dict[str, float], prefix: str, separator: str
        ) -> dict[str, float]:
            return {prefix + separator + k: v for k, v in metrics_dict.items()}

        def _patched_get_and_log_device_stats(self, trainer: "pl.Trainer", key: str) -> None:
            if not trainer._logger_connector.should_update_logs:
                return

            device = trainer.strategy.root_device
            if self._cpu_stats is False and device.type == "cpu":
                # cpu stats are disabled
                return

            device_stats = trainer.accelerator.get_device_stats(device)

            if self._cpu_stats and device.type != "cpu":
                # Don't query CPU stats twice if CPU is accelerator
                from lightning.pytorch.accelerators.cpu import get_cpu_stats

                device_stats.update(get_cpu_stats())

            for logger in trainer.loggers:
                separator = logger.group_separator
                prefixed_device_stats = _patched_prefix_metric_keys(device_stats, f"system/{key}", separator)
                logger.log_metrics(prefixed_device_stats, step=trainer.fit_loop.epoch_loop._batches_that_stepped)

        patched = False
        if hasattr(trainer, "callbacks"):
            for callback in cast(list[Callback], trainer.callbacks):  # pyright: ignore[reportAttributeAccessIssue]
                if isinstance(callback, DeviceStatsMonitor):
                    callback._get_and_log_device_stats = _patched_get_and_log_device_stats.__get__(
                        callback, DeviceStatsMonitor
                    )
                    patched = True
                    rank_zero_info("Device stats monitoring enabled!")

        if not patched:
            rank_zero_info("Device stats monitor has not been added to callbacks!")

        mlflow.enable_system_metrics_logging()

    @rank_zero_only
    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        """Called when fit, validate, test, predict, or tune begins."""
        if not self.autologging_disabled:
            rank_zero_info("Starting MLFlow Databricks logging!")
            rank_zero_info("Auto logging disabled!")
            mlflow.autolog(disable=True)
            self.autologging_disabled = True
        if trainer.is_global_zero:
            self.logger = self._get_logger(trainer.loggers)
            self._log_cluster_tags()

        self._patch_device_stats_monitor(trainer)

    @rank_zero_only
    def teardown(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        """Called when fit, validate, test, predict, or tune ends."""
        if self._prevent_entry(trainer):
            return None

        if stage == TrainerFn.TESTING:
            if self.logger and self.logger.run_id and self.client:
                pass
                # self.client.set_terminated(run_id=self.logger.run_id, status=RunStatus.to_string(RunStatus.FINISHED))

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when fit begins."""

    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when fit ends."""

    def on_sanity_check_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the validation sanity check starts."""

    def on_sanity_check_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the validation sanity check ends."""

    def on_train_batch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int
    ) -> None:
        """Called when the train batch begins."""

    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        """Called when the train batch ends.

        Note:
            The value ``outputs["loss"]`` here will be the normalized value w.r.t ``accumulate_grad_batches`` of the
            loss returned from ``training_step``.

        """

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the train epoch begins."""

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the train epoch ends.

        To access all batch outputs at the end of the epoch, you can cache step outputs as an attribute of the
        :class:`lightning.pytorch.core.LightningModule` and access them in this hook:

        .. code-block:: python

            class MyLightningModule(L.LightningModule):
                def __init__(self):
                    super().__init__()
                    self.training_step_outputs = []

                def training_step(self):
                    loss = ...
                    self.training_step_outputs.append(loss)
                    return loss


            class MyCallback(L.Callback):
                def on_train_epoch_end(self, trainer, pl_module):
                    # do something with all training_step outputs, for example:
                    epoch_mean = torch.stack(pl_module.training_step_outputs).mean()
                    pl_module.log("training_epoch_mean", epoch_mean)
                    # free up the memory
                    pl_module.training_step_outputs.clear()

        """
        if self.logger:
            metrics = {str(key): float(value) for key, value in trainer.callback_metrics.items()}
            self.logger.log_metrics(metrics, pl_module.current_epoch)

    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the val epoch begins."""

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the val epoch ends."""

    def on_test_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the test epoch begins."""

    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the test epoch ends."""

    def on_predict_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the predict epoch begins."""

    def on_predict_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the predict epoch ends."""

    def on_validation_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Called when the validation batch begins."""

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Called when the validation batch ends."""

    def on_test_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Called when the test batch begins."""

    def on_test_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Called when the test batch ends."""

    def on_predict_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Called when the predict batch begins."""

    def on_predict_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Called when the predict batch ends."""

    @rank_zero_only
    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the train begins."""
        if self._prevent_entry(trainer):
            return None

        if self.logger and self.logger.run_id and self.client:
            run_id = str(self.logger.run_id)

            self.client.set_tag(run_id=run_id, key="Mode", value="training")

            self.client.log_param(run_id=run_id, key="epochs", value=trainer.max_epochs)

            if hasattr(trainer, "optimizers"):
                for i, optimizer in enumerate(trainer.optimizers):
                    self.client.log_param(
                        self.logger.run_id, key=f"optimizer{i}_name", value=self._get_optimizer_name(optimizer)
                    )
                    if hasattr(optimizer, "defaults"):
                        self.client.log_param(
                            self.logger.run_id, key=f"optimizer{i}_defaults", value=str(optimizer.defaults)
                        )

            callback = self._resolve_early_stopping_callback(trainer)
            if callback:
                self._log_early_stop_params(callback)

    @rank_zero_only
    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the train ends."""
        if self._prevent_entry(trainer):
            return None

        callback = self._resolve_early_stopping_callback(trainer)
        if callback:
            self._log_early_stop_metrics(callback)

        self._log_model_summary(trainer, pl_module)

        self._log_model(trainer, pl_module)

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the validation loop begins."""

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the validation loop ends."""

    @rank_zero_only
    def on_test_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the test begins."""
        if self._prevent_entry(trainer):
            return None
        if self.logger and self.logger.run_id and self.client:
            self.client.set_tag(self.logger.run_id, key="Mode", value="testing")

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the test ends."""
        # originally, mlflow.autolog changes the mode to testing here, but we do it in on_test_start

    def on_predict_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the predict begins."""

    def on_predict_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when predict ends."""

    @rank_zero_only
    def on_exception(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", exception: BaseException) -> None:
        """Called when any trainer execution is interrupted by an exception."""
        if self._prevent_entry(trainer):
            return
        if self.logger and self.logger.run_id and self.client:
            self.client.set_terminated(run_id=self.logger.run_id, status=RunStatus.to_string(RunStatus.FAILED))
