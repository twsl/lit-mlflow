from collections.abc import Callable
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Literal, cast

from lightning.fabric.utilities.rank_zero import rank_zero_info, rank_zero_only, rank_zero_warn
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback, DeviceStatsMonitor, EarlyStopping
from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning.pytorch.loggers import Logger, MLFlowLogger
from lightning.pytorch.loggers.mlflow import _get_resolve_tags
from lightning.pytorch.trainer.states import TrainerFn
from lightning.pytorch.utilities.model_summary.model_summary import ModelSummary
from lightning.pytorch.utilities.types import STEP_OUTPUT
import mlflow
from mlflow import MlflowClient, MlflowException
from mlflow.entities.dataset import Dataset
from mlflow.entities.dataset_input import DatasetInput
from mlflow.entities.input_tag import InputTag
from mlflow.entities.run import Run
from mlflow.entities.run_status import RunStatus
from mlflow.models import Model, ModelSignature
from mlflow.pytorch import pickle_module as mlflow_pytorch_pickle_module
from mlflow.types import Schema, TensorSpec
from torch import Tensor
from torch.optim import Optimizer

from lit_mlflow.logger import DbxMLFlowLogger
from lit_mlflow.utils.dbx import get_databricks_tags


class MlFlowAutoCallback(Callback):
    def __init__(
        self,
        verbose: bool = True,
        patch_device_monitor: bool = True,
        serialization_format: Literal["pickle", "pt2"] = "pickle",
        input_example_fn: Callable[[Any], Tensor | tuple[Tensor, ...] | list[Tensor]] | None = None,
    ) -> None:
        """Initialize callback state and supported MLflow logger types.

        Args:
            verbose: Stored verbosity setting; it does not currently change behavior.
            patch_device_monitor: Whether to patch device statistics logging during setup.
            serialization_format: MLflow model serialization format.
            input_example_fn: Select the model's positional tensor inputs from a training batch for PT2 export.
        """
        if serialization_format not in ("pickle", "pt2"):
            raise ValueError("serialization_format must be 'pickle' or 'pt2'.")
        self.supported_loggers = (MLFlowLogger, DbxMLFlowLogger)
        self.verbose = verbose
        self.logger: MLFlowLogger | DbxMLFlowLogger | None = None
        self.autologging_disabled = False
        self.patch_device_monitor = patch_device_monitor
        self.serialization_format = serialization_format
        self.input_example_fn = input_example_fn
        self._input_example: tuple[Tensor, ...] | None = None

    @property
    def client(self) -> MlflowClient | None:
        """Get the selected logger's MLflow client, if a logger is available.

        Returns:
            The MLflow client, or None when no supported logger is selected.
        """
        if self.logger:
            return self.logger.experiment
        return None

    def _get_logger(self, loggers: list[Logger]) -> MLFlowLogger | DbxMLFlowLogger | None:
        """Select the first supported logger and warn about unsupported loggers.

        Args:
            loggers: Loggers registered with the trainer.

        Returns:
            The first supported MLflow logger, or None if none is found.
        """
        if isinstance(loggers, list):
            if len(loggers) == 0:
                rank_zero_warn("Cannot log artifacts because Trainer has no logger.")
                return None
            else:
                rank_zero_info(f" Supported loggers are: {', '.join(str(x.__name__) for x in self.supported_loggers)}")
                for logger in loggers:
                    if isinstance(logger, self.supported_loggers):
                        return logger
                    else:
                        rank_zero_warn(
                            f"{self.__class__.__name__} does not support logging with {logger.__class__.__name__}."
                        )
            return None

    def _prevent_entry(self, trainer: "pl.Trainer") -> bool:
        """Check whether logging should be skipped because of logger or rank state.

        Args:
            trainer: Trainer whose global rank is checked.

        Returns:
            True when no supported logger is selected or the trainer is not rank zero.
        """
        return self.logger is None or not trainer.is_global_zero

    def _get_optimizer(self, optimizer: LightningOptimizer | Optimizer) -> Optimizer:
        """Unwrap a Lightning optimizer, leaving ordinary optimizers unchanged.

        Args:
            optimizer: Lightning wrapper or underlying optimizer.

        Returns:
            The underlying optimizer.
        """
        return optimizer._optimizer if isinstance(optimizer, LightningOptimizer) else optimizer

    def _get_optimizer_name(self, optimizer: LightningOptimizer | Optimizer) -> str:
        """Get the class name of the underlying optimizer.

        Args:
            optimizer: Lightning wrapper or underlying optimizer.

        Returns:
            The optimizer class name.
        """
        opt = self._get_optimizer(optimizer)
        return opt.__class__.__name__

    def _get_optimizer_defaults(self, optimizer: LightningOptimizer | Optimizer) -> dict[str, Any]:
        """Get the underlying optimizer's default parameters.

        Args:
            optimizer: Lightning wrapper or underlying optimizer.

        Returns:
            The optimizer defaults, or an empty dictionary if unavailable.
        """
        opt = self._get_optimizer(optimizer)
        return opt.defaults if hasattr(opt, "defaults") else {}

    def _log_early_stop_params(self, early_stop_callback: EarlyStopping) -> None:
        """Log available early stopping configuration and stopped epoch as parameters.

        Args:
            early_stop_callback: Callback providing the values to log.
        """
        if self.logger is None:
            return None

        params = {
            p: getattr(early_stop_callback, p)
            for p in ["monitor", "mode", "patience", "min_delta", "stopped_epoch"]
            if hasattr(early_stop_callback, p)
        }
        self.logger.log_hyperparams(params)

    def _log_early_stop_metrics(self, early_stop_callback: EarlyStopping) -> None:
        """Log early stopping results when training stopped early.

        The restored epoch metric is calculated from the stopped epoch and patience;
        this method does not restore a checkpoint.

        Args:
            early_stop_callback: Callback providing the stopped epoch and metrics.
        """
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
        """Find the first early stopping callback registered with the trainer.

        Args:
            trainer: Trainer whose callbacks are inspected.

        Returns:
            The first early stopping callback, or None if absent.
        """
        if hasattr(trainer, "callbacks"):
            for callback in cast(list[Callback], trainer.callbacks):
                if isinstance(callback, EarlyStopping):
                    return callback
        return None

    def _log_model_summary(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Generate and upload the full-depth model summary as a text artifact.

        Args:
            trainer: Trainer invoking the callback; unused.
            pl_module: Model to summarize.
        """
        summary = str(ModelSummary(pl_module, max_depth=-1))
        artifact_path = "model_summary.txt"
        if self.logger and self.logger._run_id and self.client:
            with tempfile.TemporaryDirectory(prefix="test", suffix="test", dir=Path.cwd()) as tmp_dir:
                with Path.open(Path(f"{tmp_dir}/{artifact_path}"), "w") as tmp_file_summary:
                    tmp_file_summary.write(summary)
                run_id = str(self.logger.run_id)
                self.client.log_artifacts(run_id, tmp_dir, artifact_path)

    def _get_pt2_signature(self) -> ModelSignature | None:
        """Build a tensor signature from the captured PT2 input example.

        Returns:
            The model input signature, or None if no input example was captured.
        """
        if self._input_example is None:
            return None

        return ModelSignature(
            inputs=Schema(
                [
                    TensorSpec(
                        sample.numpy().dtype,
                        (-1, *sample.shape[1:]) if sample.ndim else (),
                        name=f"input_{index}" if len(self._input_example) > 1 else None,
                    )
                    for index, sample in enumerate(self._input_example)
                ]
            )
        )

    def _log_model(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Serialize and upload the model, then record its MLflow metadata.

        A warning is emitted if recording the model metadata fails.

        Args:
            trainer: Trainer invoking the callback; unused.
            pl_module: Model to serialize.
        """
        if self.logger and self.logger.run_id and self.client:
            rank_zero_info("Saving the model and uploading to MLFlow!")
            with tempfile.TemporaryDirectory(prefix="test", suffix="test", dir=Path.cwd()) as tmp_dir:
                local_path = Path(tmp_dir) / "model"
                artifact_path = "model"
                mlflow_model = Model(artifact_path=artifact_path, run_id=self.logger.run_id)
                serialization_format = self.serialization_format
                if serialization_format == "pt2" and self._input_example is None:
                    rank_zero_warn("PT2 export requires a captured tensor input; falling back to pickle.")
                    serialization_format = "pickle"
                training_modes = []
                try:
                    signature = self._get_pt2_signature() if serialization_format == "pt2" else None
                    input_example = None
                    if signature is not None and self._input_example is not None:
                        if len(self._input_example) == 1 and pl_module.device.type == "cpu":
                            input_example = self._input_example[0].numpy()
                        else:
                            input_example = tuple(sample.to(pl_module.device) for sample in self._input_example)
                        training_modes = [(module, module.training) for module in pl_module.modules()]
                        pl_module.eval()
                    mlflow.pytorch.save_model(
                        pytorch_model=pl_module,
                        path=local_path,
                        conda_env=None,
                        mlflow_model=mlflow_model,
                        code_paths=None,
                        pickle_module=mlflow_pytorch_pickle_module,
                        serialization_format=self.serialization_format,
                        signature=signature,
                        input_example=input_example,
                        extra_files=None,
                        pip_requirements=None,
                        extra_pip_requirements=None,
                    )
                except Exception as exc:
                    if serialization_format != "pt2":
                        raise
                    rank_zero_warn(f"PT2 model export failed; falling back to pickle. Reason: {exc}")
                    shutil.rmtree(local_path, ignore_errors=True)
                    serialization_format = "pickle"
                    mlflow_model = Model(artifact_path=artifact_path, run_id=self.logger.run_id)
                    mlflow.pytorch.save_model(
                        pytorch_model=pl_module,
                        path=local_path,
                        conda_env=None,
                        mlflow_model=mlflow_model,
                        code_paths=None,
                        pickle_module=mlflow_pytorch_pickle_module,
                        serialization_format=serialization_format,
                        signature=None,
                        input_example=None,
                        extra_files=None,
                        pip_requirements=None,
                        extra_pip_requirements=None,
                    )
                finally:
                    for module, training in training_modes:
                        module.training = training
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
        """Print the run ID, model artifacts, parameters, metrics, and non-MLflow tags."""
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
        """Collect Databricks cluster tags and attach them to the selected run."""
        tags = get_databricks_tags()

        if self.logger and self.logger.run_id and self.client:
            for tag, value in tags.items():
                self.client.set_tag(self.logger.run_id, key=tag, value=value)

    def _log_dataset_info(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Log the training dataset and dataloader class names as an MLflow input.

        Args:
            trainer: Trainer providing the data module and training dataloader.
            pl_module: Model passed by the callback hook; unused.
        """
        if hasattr(trainer, "datamodule") and trainer.datamodule:
            dm = trainer.datamodule.train_dataloader()  # ty: ignore[missing-attribute]
            dataset = dm.dataset
            if self.logger and self.logger.run_id and self.client:
                meta_ds = Dataset(
                    name=dataset.__class__.__name__,
                    digest="",
                    source=dm.__class__.__name__,
                    source_type="",
                )
                ds_input = DatasetInput(
                    dataset=meta_ds,
                    tags=[
                        InputTag("class", dataset.__class__.__name__),
                    ],
                )
                self.client.log_inputs(run_id=self.logger.run_id, datasets=[ds_input])

    def _patch_device_stats_monitor(self, trainer: "pl.Trainer") -> None:
        """Patch device monitors to log statistics under system-prefixed metric keys.

        Args:
            trainer: Trainer whose device statistics monitors are patched.
        """

        def _patched_prefix_metric_keys(
            metrics_dict: dict[str, float], prefix: str, separator: str
        ) -> dict[str, float]:
            """Prefix each metric key using the logger's separator.

            Args:
                metrics_dict: Device statistics to rename.
                prefix: Prefix to add to each key.
                separator: Separator between the prefix and original key.

            Returns:
                Device statistics with prefixed keys.
            """
            return {prefix + separator + k: v for k, v in metrics_dict.items()}

        def _patched_get_and_log_device_stats(self, trainer: "pl.Trainer", key: str) -> None:
            """Collect enabled device statistics and log them through each trainer logger.

            Args:
                self: Device statistics monitor receiving the patched method.
                trainer: Trainer providing the device, accelerator, and loggers.
                key: Hook-specific suffix for the system metric prefix.
            """
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
            for callback in cast(list[Callback], trainer.callbacks):
                if isinstance(callback, DeviceStatsMonitor):
                    callback._get_and_log_device_stats = _patched_get_and_log_device_stats.__get__(
                        callback, DeviceStatsMonitor
                    )
                    patched = True
                    rank_zero_info("Lightning device stats monitoring enabled!")

        if not patched:
            rank_zero_info("Lightning device stats monitor has not been added to callbacks!")

    @rank_zero_only
    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        """Disable MLflow autologging, configure logging, and mark the run as running.

        Setup also logs cluster and training dataset metadata and optionally patches
        device statistics monitors.

        Args:
            trainer: Trainer providing loggers, callbacks, and data.
            pl_module: Model passed to dataset metadata logging.
            stage: Lightning stage invoking setup; unused.
        """
        if not self.autologging_disabled:
            rank_zero_info("Starting MLFlow Databricks logging!")
            rank_zero_info("Default auto logging disabled!")
            mlflow.autolog(disable=True)
            self.autologging_disabled = True

        if trainer.is_global_zero:
            self.logger = self._get_logger(trainer.loggers)
            self._log_cluster_tags()
            self._log_dataset_info(trainer, pl_module)

        if self.patch_device_monitor:
            self._patch_device_stats_monitor(trainer)

        if not self._prevent_entry(trainer) and self.logger and self.logger.run_id and self.client:
            self.client.update_run(run_id=self.logger.run_id, status=RunStatus.to_string(RunStatus.RUNNING))

    @rank_zero_only
    def teardown(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        """Mark the run finished and terminate it when the fitting stage ends.

        Args:
            trainer: Trainer whose global rank is checked.
            pl_module: Model passed by the callback hook; unused.
            stage: Lightning stage that has ended.
        """
        if self._prevent_entry(trainer):
            return None

        if self.logger and self.logger.run_id and self.client:
            self.client.update_run(run_id=self.logger.run_id, status=RunStatus.to_string(RunStatus.FINISHED))

            if stage == TrainerFn.FITTING:
                self.client.set_terminated(run_id=self.logger.run_id, status=RunStatus.to_string(RunStatus.FINISHED))

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Do nothing when fitting begins.

        Args:
            trainer: Trainer invoking the hook.
            pl_module: Model being fitted.
        """
        self._input_example = None

    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Do nothing when fitting ends.

        Args:
            trainer: Trainer invoking the hook.
            pl_module: Model being fitted.
        """

    def on_sanity_check_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Do nothing when the validation sanity check starts.

        Args:
            trainer: Trainer invoking the hook.
            pl_module: Model being validated.
        """

    def on_sanity_check_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Do nothing when the validation sanity check ends.

        Args:
            trainer: Trainer invoking the hook.
            pl_module: Model being validated.
        """

    def on_train_batch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int
    ) -> None:
        """Capture the first model input when PT2 serialization is enabled.

        Args:
            trainer: Trainer invoking the hook.
            pl_module: Model being trained.
            batch: Current training batch.
            batch_idx: Index of the batch.
        """
        if self.serialization_format == "pt2" and self._input_example is None and not self._prevent_entry(trainer):
            if self.input_example_fn is not None:
                sample = self.input_example_fn(batch)
            else:
                sample = batch[0] if isinstance(batch, (tuple, list)) and batch else batch
            samples = (sample,) if isinstance(sample, Tensor) else sample
            if not isinstance(samples, (tuple, list)) or not samples or not all(isinstance(x, Tensor) for x in samples):
                raise ValueError("PT2 model saving requires tensor inputs; use input_example_fn to select them.")
            self._input_example = tuple(sample.detach().cpu().clone() for sample in samples)

    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        """Do nothing when a training batch ends.

        Args:
            trainer: Trainer invoking the hook.
            pl_module: Model being trained.
            outputs: Output from the training step.
            batch: Current training batch.
            batch_idx: Index of the batch.

        Note:
            When present, ``outputs["loss"]`` is normalized by ``accumulate_grad_batches``.
        """

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Do nothing when a training epoch begins.

        Args:
            trainer: Trainer invoking the hook.
            pl_module: Model being trained.
        """

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Log callback metrics as floats using the current epoch as the step.

        Args:
            trainer: Trainer providing the callback metrics.
            pl_module: Model providing the current epoch.
        """
        if self.logger:
            metrics = {str(key): float(value) for key, value in trainer.callback_metrics.items()}
            self.logger.log_metrics(metrics, pl_module.current_epoch)

    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Do nothing when a validation epoch begins.

        Args:
            trainer: Trainer invoking the hook.
            pl_module: Model being validated.
        """

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Do nothing when a validation epoch ends.

        Args:
            trainer: Trainer invoking the hook.
            pl_module: Model being validated.
        """

    def on_test_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Do nothing when a test epoch begins.

        Args:
            trainer: Trainer invoking the hook.
            pl_module: Model being tested.
        """

    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Do nothing when a test epoch ends.

        Args:
            trainer: Trainer invoking the hook.
            pl_module: Model being tested.
        """

    def on_predict_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Do nothing when a prediction epoch begins.

        Args:
            trainer: Trainer invoking the hook.
            pl_module: Model making predictions.
        """

    def on_predict_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Do nothing when a prediction epoch ends.

        Args:
            trainer: Trainer invoking the hook.
            pl_module: Model making predictions.
        """

    def on_validation_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Do nothing when a validation batch begins.

        Args:
            trainer: Trainer invoking the hook.
            pl_module: Model being validated.
            batch: Current validation batch.
            batch_idx: Index of the batch.
            dataloader_idx: Index of the validation dataloader.
        """

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Do nothing when a validation batch ends.

        Args:
            trainer: Trainer invoking the hook.
            pl_module: Model being validated.
            outputs: Output from the validation step.
            batch: Current validation batch.
            batch_idx: Index of the batch.
            dataloader_idx: Index of the validation dataloader.
        """

    def on_test_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Do nothing when a test batch begins.

        Args:
            trainer: Trainer invoking the hook.
            pl_module: Model being tested.
            batch: Current test batch.
            batch_idx: Index of the batch.
            dataloader_idx: Index of the test dataloader.
        """

    def on_test_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Do nothing when a test batch ends.

        Args:
            trainer: Trainer invoking the hook.
            pl_module: Model being tested.
            outputs: Output from the test step.
            batch: Current test batch.
            batch_idx: Index of the batch.
            dataloader_idx: Index of the test dataloader.
        """

    def on_predict_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Do nothing when a prediction batch begins.

        Args:
            trainer: Trainer invoking the hook.
            pl_module: Model making predictions.
            batch: Current prediction batch.
            batch_idx: Index of the batch.
            dataloader_idx: Index of the prediction dataloader.
        """

    def on_predict_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Do nothing when a prediction batch ends.

        Args:
            trainer: Trainer invoking the hook.
            pl_module: Model making predictions.
            outputs: Output from the prediction step.
            batch: Current prediction batch.
            batch_idx: Index of the batch.
            dataloader_idx: Index of the prediction dataloader.
        """

    @rank_zero_only
    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Tag the run for training and log epochs, optimizer defaults, and early stopping settings.

        Args:
            trainer: Trainer providing configuration, optimizers, and callbacks.
            pl_module: Model passed by the callback hook; unused.
        """
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
                    defaults = self._get_optimizer_defaults(optimizer)
                    for key, value in defaults.items():
                        self.client.log_param(self.logger.run_id, key=f"optimizer{i}_{key}", value=str(value))
                        # self.client.log_param(
                        #     self.logger.run_id, key=f"optimizer{i}_defaults", value=str(optimizer.defaults)
                        # )

            callback = self._resolve_early_stopping_callback(trainer)
            if callback:
                self._log_early_stop_params(callback)

    @rank_zero_only
    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Log early stopping results, the model summary, and the serialized model.

        Args:
            trainer: Trainer providing callbacks and the global rank.
            pl_module: Model to summarize and serialize.
        """
        if self._prevent_entry(trainer):
            return None

        callback = self._resolve_early_stopping_callback(trainer)
        if callback:
            self._log_early_stop_metrics(callback)

        self._log_model_summary(trainer, pl_module)

        try:
            self._log_model(trainer, pl_module)
        finally:
            self._input_example = None

    @rank_zero_only
    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Set the run's mode tag to validating when a logger is available.

        Args:
            trainer: Trainer invoking the hook; unused.
            pl_module: Model being validated; unused.
        """
        if self.logger and self.logger.run_id and self.client:
            run_id = str(self.logger.run_id)

            self.client.set_tag(run_id=run_id, key="Mode", value="validating")

    @rank_zero_only
    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Do nothing when validation ends.

        Args:
            trainer: Trainer invoking the hook.
            pl_module: Model being validated.
        """

    @rank_zero_only
    def on_test_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Set the run's mode tag to testing on global rank zero.

        Args:
            trainer: Trainer whose global rank is checked.
            pl_module: Model being tested; unused.
        """
        if self._prevent_entry(trainer):
            return None
        if self.logger and self.logger.run_id and self.client:
            self.client.set_tag(self.logger.run_id, key="Mode", value="testing")

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Do nothing when testing ends.

        Args:
            trainer: Trainer invoking the hook.
            pl_module: Model being tested.
        """
        # originally, mlflow.autolog changes the mode to testing here, but we do it in on_test_start

    def on_predict_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Do nothing when prediction begins.

        Args:
            trainer: Trainer invoking the hook.
            pl_module: Model making predictions.
        """

    def on_predict_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Do nothing when prediction ends.

        Args:
            trainer: Trainer invoking the hook.
            pl_module: Model making predictions.
        """

    @rank_zero_only
    def on_exception(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", exception: BaseException) -> None:
        """Mark the MLflow run as failed when trainer execution raises an exception.

        Args:
            trainer: Trainer whose global rank is checked.
            pl_module: Model passed by the callback hook; unused.
            exception: Exception that interrupted execution; unused.
        """
        self._input_example = None
        if self._prevent_entry(trainer):
            return
        if self.logger and self.logger.run_id and self.client:
            self.client.set_terminated(run_id=self.logger.run_id, status=RunStatus.to_string(RunStatus.FAILED))
