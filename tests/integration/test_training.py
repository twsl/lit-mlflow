from pathlib import Path
from typing import Literal

import lightning.pytorch as pl
from lightning.pytorch.loggers import MLFlowLogger
import mlflow.pytorch
import pytest
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from lit_mlflow import MlFlowAutoCallback


class FakeModel(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(1, 1)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layer(self.dropout(inputs))

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        inputs, targets = batch
        self.layer.eval()
        loss = nn.functional.mse_loss(self(inputs), targets)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.1)


@pytest.mark.parametrize(
    ("serialization_format", "fail_pt2_export"), [("pickle", False), ("pt2", False), ("pt2", True)]
)
def test_training_logs_to_mlflow(
    tmp_path: Path,
    serialization_format: Literal["pickle", "pt2"],
    fail_pt2_export: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if fail_pt2_export:
        save_model = mlflow.pytorch.save_model

        def fail_pt2_then_save(*args: object, **kwargs: object) -> None:
            if kwargs.get("serialization_format") == "pt2":
                raise RuntimeError("simulated export failure")
            save_model(*args, **kwargs)

        monkeypatch.setattr(mlflow.pytorch, "save_model", fail_pt2_then_save)
    inputs = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    dataloader = DataLoader(TensorDataset(inputs, inputs * 2), batch_size=2)
    logger = MLFlowLogger(
        experiment_name="integration-test",
        tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        artifact_location=(tmp_path / "artifacts").as_uri(),
    )
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        logger=logger,
        callbacks=[
            MlFlowAutoCallback(verbose=False, patch_device_monitor=False, serialization_format=serialization_format)
        ],
        default_root_dir=tmp_path,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        log_every_n_steps=1,
    )

    model = FakeModel()
    if fail_pt2_export:
        with pytest.warns(UserWarning, match="PT2 model export failed; falling back to pickle"):
            trainer.fit(model, train_dataloaders=dataloader)
    else:
        trainer.fit(model, train_dataloaders=dataloader)

    assert model.training
    assert not model.layer.training
    run = logger.experiment.get_run(logger.run_id)
    assert run.info.status == "FINISHED"
    assert run.data.params["epochs"] == "1"
    assert "train_loss" in run.data.metrics
    artifacts = logger.experiment.list_artifacts(logger.run_id, "model/model")
    artifact_names = {Path(artifact.path).name for artifact in artifacts}
    assert "MLmodel" in artifact_names
    if serialization_format == "pt2":
        if fail_pt2_export:
            assert "input_example.json" not in artifact_names
        else:
            assert "input_example.json" in artifact_names
        data_artifacts = logger.experiment.list_artifacts(logger.run_id, "model/model/data")
        if fail_pt2_export:
            assert any(artifact.path.endswith(".pth") for artifact in data_artifacts)
            assert not any(artifact.path.endswith(".pt2") for artifact in data_artifacts)
        else:
            assert any(artifact.path.endswith(".pt2") for artifact in data_artifacts)
            assert not any(artifact.path.endswith(".pth") for artifact in data_artifacts)
    model_path = logger.experiment.download_artifacts(logger.run_id, "model/model")
    loaded_model = mlflow.pytorch.load_model(model_path)
    model.eval()
    if serialization_format == "pickle":
        loaded_model.eval()
    torch.testing.assert_close(loaded_model(inputs[:3]), model(inputs[:3]))
    if serialization_format == "pt2" and not fail_pt2_export:
        pyfunc_model = mlflow.pyfunc.load_model(model_path)
        torch.testing.assert_close(torch.as_tensor(pyfunc_model.predict(inputs[:3].numpy())), model(inputs[:3]))


class MultiInputModel(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(1, 1)

    def forward(self, inputs: Tensor, offsets: Tensor) -> Tensor:
        return self.layer(inputs) + offsets

    def training_step(self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int) -> Tensor:
        inputs, offsets, targets = batch
        return nn.functional.mse_loss(self(inputs, offsets), targets)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.1)


class DictBatchModel(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1))

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features)

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        return nn.functional.mse_loss(self(batch["features"]), batch["targets"])

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.1)


def collate_dict_batch(batch: list[tuple[Tensor, Tensor]]) -> dict[str, Tensor]:
    return {
        "features": torch.stack([item[0] for item in batch]),
        "targets": torch.stack([item[1] for item in batch]),
    }


def test_pt2_export_with_dict_batches_and_variable_batch_shapes(tmp_path: Path) -> None:
    features = torch.arange(14, dtype=torch.float32).reshape(7, 2) / 10
    targets = features.sum(dim=1, keepdim=True)
    dataloader = DataLoader(TensorDataset(features, targets), batch_size=4, collate_fn=collate_dict_batch)
    logger = MLFlowLogger(
        experiment_name="dict-batch-test",
        tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        artifact_location=(tmp_path / "artifacts").as_uri(),
    )
    callback = MlFlowAutoCallback(
        patch_device_monitor=False,
        serialization_format="pt2",
        input_example_fn=lambda batch: batch["features"],
    )
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        logger=logger,
        callbacks=[callback],
        default_root_dir=tmp_path,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )
    model = DictBatchModel()

    trainer.fit(model, train_dataloaders=dataloader)

    model_path = logger.experiment.download_artifacts(logger.run_id, "model/model")
    loaded_model = mlflow.pytorch.load_model(model_path)
    model.eval()
    for batch in (features[:5], features[:1]):
        torch.testing.assert_close(loaded_model(batch), model(batch))

    pyfunc_model = mlflow.pyfunc.load_model(model_path)
    for batch in (features[:5], features[:1]):
        torch.testing.assert_close(torch.as_tensor(pyfunc_model.predict(batch.numpy())), model(batch))
    assert callback._input_example is None


def test_pt2_export_with_multiple_inputs(tmp_path: Path) -> None:
    inputs = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    dataloader = DataLoader(TensorDataset(inputs, inputs + 1, inputs * 2), batch_size=2)
    logger = MLFlowLogger(
        experiment_name="multi-input-test",
        tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        artifact_location=(tmp_path / "artifacts").as_uri(),
    )
    callback = MlFlowAutoCallback(
        patch_device_monitor=False,
        serialization_format="pt2",
        input_example_fn=lambda batch: (batch[0], batch[1]),
    )
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        logger=logger,
        callbacks=[callback],
        default_root_dir=tmp_path,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )
    model = MultiInputModel()

    trainer.fit(model, train_dataloaders=dataloader)

    model_path = logger.experiment.download_artifacts(logger.run_id, "model/model")
    metadata = mlflow.models.Model.load(Path(model_path) / "MLmodel")
    assert "python_function" not in metadata.flavors
    assert not (Path(model_path) / "input_example.json").exists()
    loaded_model = mlflow.pytorch.load_model(model_path)
    torch.testing.assert_close(loaded_model(inputs[:3], inputs[:3] + 1), model(inputs[:3], inputs[:3] + 1))
    assert callback._input_example is None
