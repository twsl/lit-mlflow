from pathlib import Path

import lightning.pytorch as pl
from lightning.pytorch.loggers import MLFlowLogger
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from lit_mlflow import MlFlowAutoCallback


class FakeModel(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(1, 1)

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        inputs, targets = batch
        loss = nn.functional.mse_loss(self.layer(inputs), targets)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.1)


def test_training_logs_to_mlflow(tmp_path: Path) -> None:
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
        callbacks=[MlFlowAutoCallback(verbose=False, patch_device_monitor=False)],
        default_root_dir=tmp_path,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        log_every_n_steps=1,
    )

    trainer.fit(FakeModel(), train_dataloaders=dataloader)

    run = logger.experiment.get_run(logger.run_id)
    assert run.info.status == "FINISHED"
    assert run.data.params["epochs"] == "1"
    assert "train_loss" in run.data.metrics
    artifacts = logger.experiment.list_artifacts(logger.run_id, "model/model")
    assert "MLmodel" in {Path(artifact.path).name for artifact in artifacts}
