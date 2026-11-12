import pytest

from lit_mlflow import DbxMLFlowLogger, MlFlowAutoCallback
from lit_mlflow.__about__ import __version__


def test_version() -> None:
    assert __version__ is not None


@pytest.mark.parametrize(("verbose", "patch_device_monitor"), [(True, True), (False, False)])
def test_callback_instantiation(verbose: bool, patch_device_monitor: bool) -> None:
    callback = MlFlowAutoCallback(verbose=verbose, patch_device_monitor=patch_device_monitor)

    assert callback.verbose is verbose
    assert callback.patch_device_monitor is patch_device_monitor
    assert callback.client is None


def test_logger_instantiation() -> None:
    logger = DbxMLFlowLogger(
        experiment_name="unit-test",
        run_name="unit-run",
        tracking_uri="http://127.0.0.1:5000",
        tags={"purpose": "test"},
    )

    assert logger._experiment_name == "unit-test"
    assert logger._run_name == "unit-run"
    assert logger._tracking_uri == "http://127.0.0.1:5000"
    assert logger.tags == {"purpose": "test"}
    assert MlFlowAutoCallback()._get_logger([logger]) is logger
