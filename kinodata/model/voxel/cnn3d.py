import torch
from lightning.pytorch import LightningModule
from torch.nn import (
    BatchNorm3d,
    Conv3d,
    Identity,
    MaxPool3d,
    ReLU,
    Sequential,
    LazyLinear,
    Flatten,
    BatchNorm1d,
)
from torch.nn.functional import relu
from torch import Tensor, nn
from torch.optim import AdamW
from torchmetrics.regression import PearsonCorrCoef


class VoxelModel(LightningModule):

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
        lr: float = 1e-4,
        lr_decay: float = 1e-5,
    ):
        super().__init__()
        self.corr_metrics = {key: PearsonCorrCoef() for key in ["train", "val", "test"]}
        self.save_hyperparameters()
        self.define_model()

    def define_model(self):
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def configure_optimizers(self):
        return AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.lr_decay
        )

    def _step(self, batch, key, *args, **kwargs):
        x = batch[0]
        y = batch[1]
        pred = self(x)
        loss = (pred.flatten() - y.flatten()).pow(2).mean()
        mae = (pred.flatten() - y.flatten()).abs().mean()
        self.log(f"{key}/loss", loss)
        self.log(f"{key}/mae", mae, on_epoch=True)
        self.corr_metrics[key](pred.flatten().detach().cpu(), y.flatten().cpu())
        return x, y, pred, loss

    def _shared_on_epoch_end(self, key):
        self.log(f"{key}/corr", self.corr_metrics[key].compute())
        self.corr_metrics[key].reset()

    def on_train_epoch_end(self):
        return self._shared_on_epoch_end("train")

    def on_validation_epoch_end(self):
        return self._shared_on_epoch_end("val")

    def on_test_epoch_end(self):
        return self._shared_on_epoch_end("test")

    def training_step(self, batch, *args, **kwargs):
        x, y, pred, loss = self._step(batch, "train", *args, **kwargs)
        return loss

    def validation_step(self, batch, *args, **kwargs):
        x, y, pred, loss = self._step(batch, "val", *args, **kwargs)
        return loss

    def test_step(self, batch, *args, **kwargs):
        x, y, pred, loss = self._step(batch, "test", *args, **kwargs)
        return loss

    def predict_step(self, batch, *args):
        x = batch[0]
        y = batch[1]
        chembl_activity_id = batch[3]
        if isinstance(chembl_activity_id, torch.Tensor):
            chembl_activity_id = chembl_activity_id.cpu().flatten()
        pred = self.forward(x).flatten()
        return {
            "pred": pred,
            "target": y,
            "chembl_activity_id": chembl_activity_id,
        }


class PafnucyBlock(nn.Module):

    def __init__(
        self,
        kernel_size: int | None = None,
        hidden_channels: int | None = None,
        in_channels: int | None = None,
    ):
        super().__init__()
        if in_channels is None:
            in_channels = hidden_channels
        if kernel_size is None:
            kernel_size = 3
        if in_channels != hidden_channels:
            self.proj_lin = nn.Conv3d(
                in_channels, hidden_channels, 1, padding=0, stride=1, bias=False
            )
        else:
            self.proj_lin = Identity()
        self.cnn1 = nn.Conv3d(
            in_channels, hidden_channels, kernel_size, padding="same", stride=1
        )
        self.bn1 = nn.BatchNorm3d(hidden_channels)
        self.cnn2 = nn.Conv3d(hidden_channels, hidden_channels, 1, padding=0, stride=1)
        self.bn2 = nn.BatchNorm3d(hidden_channels)

    def forward(self, x: Tensor):
        z = self.cnn1(x)
        z = self.bn1(z)
        z = relu(z)
        z = self.cnn2(z)
        z = self.bn2(z)
        return relu(z + self.proj_lin(x))


class PafnucyPool(nn.Module):

    def __init__(
        self,
        hidden_channels: int,
        type: str = "max",
    ):
        super().__init__()
        match type:
            case "max":
                self.pool = nn.MaxPool3d(2)
            case "avg":
                self.pool = nn.AvgPool3d(2)
            case "learned":
                self.pool = nn.Conv3d(hidden_channels, hidden_channels, 2, stride=2)
            case _:
                raise ValueError(f"Unknown pooling type {type}")

    def forward(self, x: Tensor):
        return self.pool(x)


class PafnucyIshVoxelModel(VoxelModel):

    def __init__(
        self,
        in_channels,
        hidden_channels=32,
        dense_channels=32,
        kernel_sizes: list | None = None,
        lr=0.0001,
        lr_decay=0.00001,
        pooling_type: str = "max",
        pool_every: int = 1,
    ):
        if kernel_sizes is None:
            kernel_sizes = [5, 3, 3]
        super().__init__(in_channels, hidden_channels, lr, lr_decay)

    @property
    def hidden_channel_list(self) -> list[int]:
        h = self.hparams.hidden_channels
        return [h] + [
            (h := h * 2) if i % self.hparams.pool_every == 0 else h
            for i in range(len(self.hparams.kernel_sizes) - 1)
        ]

    def define_model(self):
        hidden_dims = self.hidden_channel_list
        self.blocks = nn.ModuleList()
        self.blocks.append(
            PafnucyBlock(
                in_channels=self.hparams.in_channels,
                hidden_channels=hidden_dims[0],
                kernel_size=self.hparams.kernel_sizes[0],
            )
        )

        for i, kernel_size in enumerate(self.hparams.kernel_sizes[1:]):
            in_dim = hidden_dims[i]
            out_dim = hidden_dims[i + 1]
            self.blocks.append(
                PafnucyBlock(
                    in_channels=in_dim, hidden_channels=out_dim, kernel_size=kernel_size
                )
            )
        self.pools = nn.ModuleList()
        for j in range(len(self.blocks)):
            if j % self.hparams.pool_every == 0:
                self.pools.append(
                    PafnucyPool(hidden_dims[j], type=self.hparams.pooling_type)
                )
                continue
            self.pools.append(Identity())

        self.dense = Sequential(
            Flatten(start_dim=1),
            LazyLinear(self.hparams.dense_channels),
            ReLU(),
            BatchNorm1d(self.hparams.dense_channels),
            LazyLinear(self.hparams.dense_channels // 2),
            ReLU(),
            BatchNorm1d(self.hparams.dense_channels // 2),
            LazyLinear(1),
        )

    def forward(self, x):
        for block, pool in zip(self.blocks, self.pools):
            x = block(x)
            x = pool(x)
        return self.dense(x)


if __name__ == "__main__":
    model = PafnucyIshVoxelModel(
        12,
        128,
        512,
        [5, 3, 3, 3, 3],
        pool_every=2,
    )
    print(model)

    sample_input = torch.randn(8, 12, 24, 24, 24)
    print(model(sample_input).shape)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters : {num_params // 1e6}M")
