import logging
import os
from typing import Any, Literal

from lightning.fabric.loggers.logger import Logger, rank_zero_experiment
from lightning.fabric.utilities.rank_zero import rank_zero_info, rank_zero_only, rank_zero_warn  # type: ignore
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
        tracking_uri: str | None = mlflow.get_tracking_uri(),  # os.getenv("MLFLOW_TRACKING_URI"),
        tags: dict[str, Any] | None = None,
        save_dir: str | None = "./mlruns",
        log_model: Literal[True, False, "all"] = False,
        prefix: str = "",
        artifact_location: str | None = None,
        run_id: str | None = None,
    ):
        super().__init__(
            experiment_name, run_name, tracking_uri, tags, save_dir, log_model, prefix, artifact_location, run_id
        )
        if not is_in_databricks():
            rank_zero_warn(f"You are running `{self.__class__.__name__}` outside of Databricks.")
        else:
            self._fix_logging()
            if patch_dbx_credentials():
                client = self.experiment
                active_run = mlflow.active_run()
                if client and active_run:
                    run_id = active_run.info.run_id or ""
                    run_url = get_databricks_run_url(tracking_uri or "databricks", run_id)
                    rank_zero_info(f"MLflow run URL: {run_url}")
                else:
                    rank_zero_warn("Could not retrieve the MLflow run.")
            else:
                rank_zero_warn("Could not patch Databricks credentials.")

    @rank_zero_only
    def _fix_logging(self) -> None:
        """Fix the logging level of the Py4J gateway to ERROR to prevent log spam."""
        logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)

    @property
    @rank_zero_experiment
    def experiment(self) -> MlflowClient:
        if is_in_databricks_notebook():
            self._experiment_id = get_experiment_id(None)
        return super().experiment
