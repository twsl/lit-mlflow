import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--experiment-name", default="/Shared/lit-mlflow")
arguments, _ = parser.parse_known_args()
experiment_name = arguments.experiment_name or "lightning_logs"

# COMMAND ----------
import torch
from torch import Tensor
from torch.utils.data import Dataset


class RandomDataset(Dataset):
    def __init__(self, size: int, num_samples: int) -> None:
        self.len = num_samples
        self.data = torch.randn(num_samples, size)

    def __getitem__(self, index: int) -> Tensor:
        return self.data[index]

    def __len__(self) -> int:
        return self.len


# COMMAND ----------
import lightning as L  # noqa: N812
from torch.utils.data import DataLoader


class RandomDataModule(L.LightningDataModule):
    def __init__(self, size: int = 32, num_samples: int = 10000, batch_size: int = 32, num_workers: int = 5) -> None:
        """The Random data module.

        Args:
            size: The tensor size.
            num_samples: The number of samples.
            batch_size: The batch size.
            num_workers: Number of workers.
        """
        super().__init__()
        self.size = size
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str | None = None) -> None:
        """Setup the data module per stage.

        Args:
            stage: The training stage.
        """
        self.data_test = RandomDataset(self.size, self.num_samples)
        self.data_train = RandomDataset(self.size, self.num_samples)
        self.data_val = RandomDataset(self.size, self.num_samples)
        self.data_predict = RandomDataset(self.size, self.num_samples)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.data_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.data_test, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self.data_predict, batch_size=self.batch_size, num_workers=self.num_workers)


# COMMAND ----------
from typing import Any


class BoringModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(32, 2)

        self.training_step_outputs = []
        self.validation_step_outputs = []

    def forward(self, x):
        self.log("system/cpu_utilization_percentage", 0.0)
        return self.layer(x)

    def loss(self, batch, prediction):
        # An arbitrary loss to have a loss that updates the model weights during `Trainer.fit` calls
        return torch.nn.functional.mse_loss(prediction, torch.ones_like(prediction))

    def training_step(self, batch, batch_idx) -> dict[str, Any]:
        output = self(batch)
        loss = self.loss(batch, output)
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx) -> None:
        output = self(batch)
        loss = self.loss(batch, output)
        self.log("valid_loss", loss)

    def test_step(self, batch, batch_idx) -> None:
        output = self(batch)
        loss = self.loss(batch, output)
        self.log("test_loss", loss)

    # def on_train_epoch_end(self, outputs) -> Any:
    #     return torch.stack([x["loss"] for x in outputs]).mean()

    # def on_validation_epoch_end(self, outputs) -> Any:
    #     return torch.stack([x["x"] for x in outputs]).mean()

    # def on_test_epoch_end(self, outputs) -> Any:
    #     return torch.stack([x["y"] for x in outputs]).mean()

    def configure_optimizers(self) -> tuple[list[Any], list[Any]]:
        optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]


# COMMAND ----------
# COMMAND ----------
from lightning.pytorch.callbacks import DeviceStatsMonitor

# COMMAND ----------
from lightning.pytorch.callbacks.progress import ProgressBar, RichProgressBar, TQDMProgressBar

from lit_mlflow import DbxMLFlowLogger, MlFlowAutoCallback

# COMMAND ----------
dm = RandomDataModule()

model = BoringModel()

# Initialize a trainer
trainer = L.Trainer(
    limit_train_batches=1000,
    limit_val_batches=100,
    limit_test_batches=10,
    num_sanity_val_steps=0,
    max_epochs=50,
    log_every_n_steps=100,
    enable_model_summary=False,
    logger=DbxMLFlowLogger(experiment_name=arguments.experiment_name),
    callbacks=[ProgressBar(), MlFlowAutoCallback(), DeviceStatsMonitor()],
)

# Train the model ⚡
trainer.fit(model, datamodule=dm)

trainer.test(datamodule=dm, ckpt_path="best")

# COMMAND ----------
run = trainer.logger.experiment.get_run(trainer.logger.run_id)  # ty: ignore[missing-attribute]

print(f"Run ID: {run.info.run_id}")
for name, value in sorted(run.data.metrics.items()):
    print(f"{name}: {value}")
