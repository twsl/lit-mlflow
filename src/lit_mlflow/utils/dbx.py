import json
import os

from mlflow.utils.databricks_utils import (
    _get_property_from_spark_context,
    get_notebook_id,
    is_in_databricks_job,
    is_in_databricks_model_serving_environment,
    is_in_databricks_notebook,
)


def is_in_databricks() -> bool:
    """Check if the code is running in Databricks."""
    return is_in_databricks_notebook() or is_in_databricks_job() or is_in_databricks_model_serving_environment()


def patch_dbx_credentials() -> bool:
    """Patch/propagate the Databricks credentials to the environment variables, required for DDP.

    Returns:
        Returns True if the values have been patched.
    """
    if is_in_databricks_notebook():
        from databricks_cli.configure.provider import DatabricksConfig, get_config

        config: DatabricksConfig = get_config()
        if config:
            for k, v in config.__dict__.items():
                if v:
                    os.environ[f"DATABRICKS_{k.upper()}"] = str(v)
            return True
    return False


def get_experiment_id(default_name: str | None = None) -> str | None:
    """Returns the experiment Id if running in Databricks.

    Args:
        default_name: The optional default name to return if not in Databricks.

    Returns:
        The experiment Id.
    """
    if is_in_databricks_notebook():
        notebook_id = get_notebook_id()
        if notebook_id:
            return notebook_id
    return default_name


def get_databricks_tags() -> dict[str, str]:
    """Return a selection of Databricks cluster tags.

    Returns:
        The tags dictionary.
    """
    tags = {}
    if is_in_databricks():
        all_tags_str = _get_property_from_spark_context("spark.databricks.clusterUsageTags.clusterAllTags")
        if all_tags_str:
            tags_list = json.loads(all_tags_str)
            for tag in tags_list:
                tags[tag["key"]] = tags_list["value"]
        node_type = _get_property_from_spark_context("spark.databricks.clusterUsageTags.clusterNodeType")
        if node_type:
            tags["clusterNodeType"] = str(node_type)
        runtime_version = os.environ.get("DATABRICKS_RUNTIME_VERSION")
        if runtime_version:
            tags["runtimeVersion"] = str(runtime_version)
        num_worker = _get_property_from_spark_context("spark.databricks.clusterUsageTags.clusterWorkers")
        if num_worker:
            tags["numWorkers"] = int(num_worker)
    return tags
