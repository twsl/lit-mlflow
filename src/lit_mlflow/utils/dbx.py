from dataclasses import dataclass, field
import functools
import json
import logging
import os
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from databricks.sdk import WorkspaceClient

from databricks.sdk import WorkspaceClient
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

logger = logging.getLogger(__name__)

_DATABRICKS_CFG_PATH = Path.home() / ".databrickscfg"


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
        True when the SDK provides a host and authorization header, otherwise False.
    """
    if not is_in_databricks():
        return False

    context = resolve_databricks_credentials()

    return context is not None


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


@dataclass(frozen=True)
class DatabricksCredentials:
    """Resolved Databricks workspace credentials."""

    host: str
    token: str


@dataclass(frozen=True)
class DatabricksContext:
    """Resolved Databricks notebook context from dbutils.

    All fields are None when dbutils is unavailable. The context is resolved
    once and cached — every field comes from the same ``getContext()`` call.
    """

    host: str | None = None
    token: str | None = None
    cluster_id: str | None = None
    notebook_path: str | None = None
    notebook_id: str | None = None
    user_name: str | None = None
    user_id: str | None = None
    workspace_id: str | None = None
    browser_host_name: str | None = None
    tags: dict[str, str] = field(default_factory=dict)
    extra_context: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def resolve_databricks_credentials() -> DatabricksCredentials | None:
    """Resolve credentials from dbutils and propagate to env vars and config file.

    Returns None when dbutils is unavailable (e.g. on workers without a notebook
    context, or outside Databricks entirely). In that case, existing env vars or
    ``~/.databrickscfg`` are left untouched.

    When credentials are resolved, this function:
      1. Sets ``DATABRICKS_HOST`` and ``DATABRICKS_TOKEN`` in ``os.environ``
      2. Writes ``~/.databrickscfg`` with a ``[DEFAULT]`` profile
    """
    if _credentials_already_set():
        return DatabricksCredentials(
            host=os.environ["DATABRICKS_HOST"],
            token=os.environ["DATABRICKS_TOKEN"],
        )

    try:
        credentials = _get_credentials_from_dbutils()
    except Exception:
        logger.debug("Could not resolve Databricks credentials from dbutils.", exc_info=True)
        return None

    if credentials is None:
        return None

    _propagate_credentials(credentials)
    return credentials


def resolve_databricks_context() -> DatabricksContext:
    """Resolve the full notebook context from dbutils, including cluster and user info.

    Returns a ``DatabricksContext`` with every available field populated.
    When dbutils is unavailable, all fields are None and ``error`` is set.

    This is a superset of ``resolve_databricks_credentials`` — it also resolves
    cluster ID, notebook path, user name, tags, and extra context.
    """
    try:
        ctx = _get_dbutils_context()
    except Exception:
        logger.debug("Could not resolve Databricks context from dbutils.", exc_info=True)
        return DatabricksContext(error="dbutils unavailable")

    if ctx is None:
        return DatabricksContext(error="dbutils unavailable")

    return _extract_context(ctx)


def _credentials_already_set() -> bool:
    return "DATABRICKS_HOST" in os.environ and "DATABRICKS_TOKEN" in os.environ


@functools.lru_cache(maxsize=1)
def _get_credentials_from_dbutils() -> DatabricksCredentials | None:
    """Extract workspace URL and token from the notebook context via dbutils.

    Retries up to 5 times with a 1-second backoff because the dbutils JVM bridge
    can fail transiently in multi-process environments.
    """
    max_retries = 5
    last_exception: Exception | None = None

    for _ in range(max_retries):
        try:
            ctx = _get_dbutils_context()
            if ctx is None:
                return None
            extracted = _extract_context(ctx)
            if extracted.host and extracted.token:
                return DatabricksCredentials(host=extracted.host, token=extracted.token)
        except Exception as exc:
            last_exception = exc
            time.sleep(1)
        else:
            return None

    logger.warning(
        "Failed to get Databricks credentials after %d retries. Last error: %s",
        max_retries,
        last_exception,
    )
    return None


def _extract_context(ctx: Any) -> DatabricksContext:
    """Extract all available fields from a dbutils notebook context object."""

    def _safe_get(callable_attr: Any) -> str | None:
        try:
            return callable_attr().get()
        except Exception:
            return None

    def _safe_get_json(callable_attr: Any) -> dict[str, Any]:
        try:
            raw = callable_attr().get()
            return json.loads(raw) if isinstance(raw, str) else {}
        except Exception:
            return {}

    return DatabricksContext(
        host=_safe_get(ctx.apiUrl),
        token=_safe_get(ctx.apiToken),
        cluster_id=_safe_get(ctx.clusterId) if hasattr(ctx, "clusterId") else None,
        notebook_path=_safe_get(ctx.notebookPath) if hasattr(ctx, "notebookPath") else None,
        notebook_id=_safe_get(ctx.notebookId) if hasattr(ctx, "notebookId") else None,
        user_name=_safe_get(ctx.userName) if hasattr(ctx, "userName") else None,
        user_id=_safe_get(ctx.userId) if hasattr(ctx, "userId") else None,
        workspace_id=_safe_get(ctx.workspaceId) if hasattr(ctx, "workspaceId") else None,
        browser_host_name=_safe_get(ctx.browserHostName) if hasattr(ctx, "browserHostName") else None,
        tags=_safe_get_json(ctx.tags) if hasattr(ctx, "tags") else {},
        extra_context=_safe_get_json(ctx.extraContext) if hasattr(ctx, "extraContext") else {},
    )


@functools.lru_cache(maxsize=1)
def _get_dbutils_context() -> Any | None:
    """Return the dbutils notebook context, or None if unavailable."""
    try:
        from databricks.sdk import WorkspaceClient

        return WorkspaceClient().dbutils.entry_point.getDbutils().notebook().getContext()
    except Exception:
        return None


def _propagate_credentials(credentials: DatabricksCredentials) -> None:
    """Set env vars and write ``~/.databrickscfg`` if not already present."""
    os.environ["DATABRICKS_HOST"] = credentials.host
    os.environ["DATABRICKS_TOKEN"] = credentials.token

    if _config_already_has_default(credentials):
        return

    try:
        _DATABRICKS_CFG_PATH.write_text(f"[DEFAULT]\nhost={credentials.host}\ntoken={credentials.token}\n")
    except OSError:
        logger.debug("Could not write %s.", _DATABRICKS_CFG_PATH, exc_info=True)


def _config_already_has_default(credentials: DatabricksCredentials) -> bool:
    """Return True if ``~/.databrickscfg`` already has matching [DEFAULT] host and token."""
    if not _DATABRICKS_CFG_PATH.is_file():
        return False

    try:
        text = _DATABRICKS_CFG_PATH.read_text()
    except OSError:
        logger.debug("Could not read %s.", _DATABRICKS_CFG_PATH, exc_info=True)
        return False

    expected_host = f"host={credentials.host}"
    expected_token = f"token={credentials.token}"

    in_default = False
    found_host = False
    found_token = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "[DEFAULT]":
            in_default = True
            found_host = False
            found_token = False
        elif stripped.startswith("[") and stripped.endswith("]"):
            in_default = False
        elif in_default:
            if stripped == expected_host:
                found_host = True
            elif stripped == expected_token:
                found_token = True

    return found_host and found_token
