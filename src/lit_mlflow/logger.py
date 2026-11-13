import logging
import os
from typing import Any, Literal

from lightning.fabric.loggers.logger import Logger, rank_zero_experiment
from lightning.fabric.utilities.rank_zero import rank_zero_info, rank_zero_only, rank_zero_warn
from lightning.pytorch.loggers import MLFlowLogger
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.utils.databricks_utils import get_databricks_run_url, is_in_databricks_notebook

from lit_mlflow.utils.dbx import get_experiment_id, is_in_databricks, patch_dbx_credentials


class DbxMLFlowLogger(MLFlowLogger):
    def __init__(
        self,
        experiment_name: str = "lightning_logs",
        run_name: str | None = None,
        tracking_uri: str | None = mlflow.get_tracking_uri(),
        tags: dict[str, Any] | None = None,
        save_dir: str | None = "./mlruns",
        log_model: Literal[True, False, "all"] = False,
        prefix: str = "",
        artifact_location: str | None = None,
        run_id: str | None = None,
    ):
        """Initialize an MLflow logger and configure Databricks integration when available.

        In Databricks, suppress Py4J gateway logs and propagate notebook credentials
        before initializing the MLflow client. Outside Databricks, emit a warning.

        Args:
            experiment_name: Name of the MLflow experiment.
            run_name: Optional name for the MLflow run.
            tracking_uri: URI of the MLflow tracking server.
            tags: Tags to associate with the run.
            save_dir: Local directory used when no tracking URI is provided.
            log_model: Whether and when to log model checkpoints.
            prefix: Prefix for logged metric keys.
            artifact_location: Location for the experiment's artifacts.
            run_id: ID of an existing run to reuse.
        """
        if is_in_databricks() and not patch_dbx_credentials():
            rank_zero_warn("Could not patch Databricks credentials before creating the MLflow client.")

        super().__init__(
            experiment_name, run_name, tracking_uri, tags, save_dir, log_model, prefix, artifact_location, run_id
        )
        if not is_in_databricks():
            rank_zero_warn(f"You are running `{self.__class__.__name__}` outside of Databricks.")
        else:
            self._fix_logging()
            client = self.experiment
            if client:
                rank_zero_info("MLflow client created.")
            else:
                rank_zero_warn("Could not retrieve the MLflow client.")

    def _get_url(self) -> str | None:
        """Build the Databricks URL for this logger's run.

        Returns:
            The run URL, or None when no URL can be determined.
        """
        run_url = get_databricks_run_url(self._tracking_uri or "databricks", self.run_id or "")
        return run_url

    @rank_zero_only
    def _fix_logging(self) -> None:
        """Fix the logging level of the Py4J gateway to ERROR to prevent log spam."""
        logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)

    @property
    @rank_zero_experiment
    def experiment(self) -> MlflowClient:
        """Get the MLflow client, using the notebook ID as experiment ID when available.

        Returns:
            The MLflow client provided by the parent logger.
        """
        if is_in_databricks_notebook():
            self._experiment_id = get_experiment_id(None)
        return super().experiment

    @property
    def run_id(self) -> str | None:
        """Get the logger's run ID, preferring the active MLflow run.

        Returns:
            The active run ID, the logger's existing run ID, or None.
        """
        _ = self.experiment
        active_run = mlflow.active_run()
        if active_run:
            self._run_id = active_run.info.run_id or ""
        return self._run_id

    # @rank_zero_only
    # def finalize(self, status: str = "success") -> None:
    #     final = super().finalize(status)
    #     rank_zero_info(f"MLflow run URL: {self._get_url()}")
    #     return final
