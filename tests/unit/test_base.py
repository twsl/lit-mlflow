from pathlib import Path

import lightning.pytorch as pl
from lightning.pytorch.loggers import MLFlowLogger
import pytest
import torch

from lit_mlflow import DbxMLFlowLogger, MlFlowAutoCallback
from lit_mlflow.__about__ import __version__


def test_version() -> None:
    assert __version__ is not None


@pytest.mark.parametrize(("verbose", "patch_device_monitor"), [(True, True), (False, False)])
def test_callback_instantiation(verbose: bool, patch_device_monitor: bool) -> None:
    callback = MlFlowAutoCallback(verbose=verbose, patch_device_monitor=patch_device_monitor)

    assert callback.verbose is verbose
    assert callback.patch_device_monitor is patch_device_monitor
    assert callback.serialization_format == "pickle"
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


def test_pt2_captures_detached_input_once(tmp_path: Path) -> None:
    callback = MlFlowAutoCallback(serialization_format="pt2", input_example_fn=lambda batch: batch["features"])
    callback.logger = MLFlowLogger(tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}")
    trainer = pl.Trainer(logger=False, enable_checkpointing=False, enable_model_summary=False)
    model = pl.LightningModule()
    inputs = torch.tensor([[1.0], [2.0]], requires_grad=True)

    callback.on_train_batch_start(trainer, model, {"features": inputs}, 0)
    inputs.detach().zero_()
    callback.on_train_batch_start(trainer, model, {"features": torch.ones(2, 1)}, 1)

    assert callback._input_example is not None
    assert not callback._input_example[0].requires_grad
    torch.testing.assert_close(callback._input_example[0], torch.tensor([[1.0], [2.0]]))
    callback.on_fit_start(trainer, model)
    assert callback._input_example is None
    callback.on_train_batch_start(trainer, model, {"features": torch.ones(2, 1)}, 0)
    callback.logger = None
    callback.on_exception(trainer, model, RuntimeError("interrupted"))
    assert callback._input_example is None


def test_pt2_rejects_batch_without_tensor_inputs(tmp_path: Path) -> None:
    callback = MlFlowAutoCallback(serialization_format="pt2")
    callback.logger = MLFlowLogger(tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}")
    trainer = pl.Trainer(logger=False, enable_checkpointing=False, enable_model_summary=False)

    with pytest.raises(ValueError, match="input_example_fn"):
        callback.on_train_batch_start(trainer, pl.LightningModule(), {"features": torch.ones(2, 1)}, 0)


def test_pt2_without_supported_logger_does_not_inspect_batch() -> None:
    callback = MlFlowAutoCallback(serialization_format="pt2")
    trainer = pl.Trainer(logger=False, enable_checkpointing=False, enable_model_summary=False)

    callback.on_train_batch_start(trainer, pl.LightningModule(), {"features": torch.ones(2, 1)}, 0)

    assert callback._input_example is None
