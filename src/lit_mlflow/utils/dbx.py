import json
import os

from mlflow.utils.databricks_utils import (
    _get_property_from_spark_context,
    get_cluster_id,
    get_databricks_runtime_version,
    get_job_group_id,
    get_job_id,
    get_job_run_id,
    get_job_type,
    get_job_type_info,
    get_notebook_id,
    get_notebook_path,
    get_repl_id,
    get_workload_class,
    get_workload_id,
    is_databricks_connect,
    is_databricks_serverless,
    is_dbfs_fuse_available,
    is_in_databricks_job,
    is_in_databricks_model_serving_environment,
    is_in_databricks_notebook,
    is_in_databricks_repo,
    is_in_databricks_repo_notebook,
    is_in_databricks_runtime,
    is_in_databricks_serverless_runtime,
    is_in_databricks_shared_cluster_runtime,
    is_running_in_ipython_environment,
    is_uc_volume_fuse_available,
)


def is_in_databricks() -> bool:
    """Check for a Databricks notebook, job, or model serving environment.

    Returns:
        True when any supported Databricks environment is detected.
    """
    return is_in_databricks_notebook() or is_in_databricks_job() or is_in_databricks_model_serving_environment()


def patch_dbx_credentials() -> bool:
    """Expose Databricks runtime credentials through MLflow environment variables for workers.

    Each variable is named by uppercasing the configuration attribute and adding
    the ``DATABRICKS_`` prefix, making the values available to DDP processes.

    Note:
        This is required to ensure that Databricks credentials are available as environment variables for DDP processes.

    Returns:
        True when a notebook configuration is available, otherwise False.
    """
    if not is_in_databricks():
        return False

    from databricks.sdk.config import Config

    config = Config()
    if config.host:
        os.environ["DATABRICKS_HOST"] = config.host

    authorization = config.authenticate().get("Authorization", "")
    has_bearer_token = authorization.lower().startswith("bearer ")
    if has_bearer_token:
        os.environ["DATABRICKS_TOKEN"] = authorization.split(" ", 1)[1]

    if config.skip_verify is not None:
        os.environ["DATABRICKS_INSECURE"] = str(config.skip_verify).lower()

    return bool(config.host and has_bearer_token)


def get_experiment_id(default_name: str | None = None) -> str | None:
    """Get the notebook ID for use as an experiment ID, or return the default.

    Args:
        default_name: Value to return outside a notebook or when its ID is unavailable.

    Returns:
        The notebook ID when available, otherwise the supplied default.
    """
    if is_in_databricks_notebook():
        notebook_id = get_notebook_id()
        if notebook_id:
            return notebook_id
    return default_name


def get_databricks_tags() -> dict[str, str]:
    """Collect Databricks cluster tags, node type, runtime version, and worker count.

    Outside Databricks, no cluster metadata is collected.

    Returns:
        Collected metadata, with the worker count stored as an integer when available.

    Raises:
        TypeError: If the cluster tag list contains an entry, because the list is
            indexed with a string while reading its value.
    """
    tags = {}
    if is_in_databricks():
        all_tags_str = _get_property_from_spark_context("spark.databricks.clusterUsageTags.clusterAllTags")
        if all_tags_str:
            tags_list = json.loads(all_tags_str)
            for tag in tags_list:
                if "key" in tag and "value" in tag:
                    tags[tag["key"]] = tag["value"]
        node_type = _get_property_from_spark_context("spark.databricks.clusterUsageTags.clusterNodeType")
        if node_type:
            tags["clusterNodeType"] = str(node_type)
        runtime_version = os.environ.get("DATABRICKS_RUNTIME_VERSION")
        if runtime_version:
            tags["runtimeVersion"] = str(runtime_version)
        num_worker = _get_property_from_spark_context("spark.databricks.clusterUsageTags.clusterWorkers")
        if num_worker:
            tags["numWorkers"] = int(num_worker)
    return tags  # ty: ignore[invalid-return-type]
