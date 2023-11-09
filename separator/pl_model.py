from model.PM_Unet import Model_Unet

from train.loss import MultiResSpecLoss

from train import augment

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import ScaleInvariantSignalDistortionRatio
from torchmetrics.functional.audio import (
    scale_invariant_signal_distortion_ratio,
    signal_distortion_ratio,
)


def compute_uSDR(
    predT: torch.Tensor, tgtT: torch.Tensor, delta: float = 1e-7
) -> torch.Tensor:
    num = torch.sum(torch.square(tgtT), dim=(1, 2))
    den = torch.sum(torch.square(tgtT - predT), dim=(1, 2))
    num += delta
    den += delta
    usdr = 10 * torch.log10(num / den)
    return usdr.mean()


class My_model(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.model = Model_Unet(
            source=["drums", "bass", "other", "vocals"], depth=4, channel=28
        )

        # loss
        self.criterion_1 = nn.L1Loss()
        self.criterion_2 = MultiResSpecLoss(
            factor=1, f_complex=1, gamma=0.3, n_ffts=[4096]
        )
        self.criterion_3 = ScaleInvariantSignalDistortionRatio()

        # augment
        self.augment = [augment.Shift(shift=int(8192), same=True)]
        self.augment += [
            augment.PitchShift_f(proba=0.2),
            augment.TimeChange_f(proba=0.2),
            augment.FlipChannels(),
            augment.FlipSign(),
            augment.Remix(proba=1, group_size=4),
            augment.Scale(proba=1, min=0.25, max=1.25),
            augment.FadeMask(proba=0.1),
            augment.Double(proba=0.1),
            augment.Reverse(proba=0.2),
        ]
        self.augment = torch.nn.Sequential(*self.augment)

    def forward(self, x):
        x = self.model(x)
        return x

    def loss(self, y_true, y_pred):  # L = L_1 + L_{MRS} - L_{SISDR}
        loss = (
            self.criterion_1(y_pred, y_true)
            + self.criterion_2(y_pred, y_true)
            - self.criterion_3(y_pred, y_true)
        )
        return loss

    def training_step(self, batch, batch_idx):
        source = batch
        source = self.augment(source)
        mix = source.sum(dim=1)

        source_predict = self.model(mix)

        drums_loss = self.loss(source_predict[:, 0], source[:, 0]) / 3

        bass_loss = self.loss(source_predict[:, 1], source[:, 1]) / 3

        other_loss = self.loss(source_predict[:, 2], source[:, 2]) / 3

        vocals_loss = self.loss(source_predict[:, 3], source[:, 3]) / 3

        loss = 0.25 * (drums_loss + bass_loss + other_loss + vocals_loss)

        self.log_dict(
            {
                "train_loss": loss,
                "train_drums": drums_loss,
                "train_bass": bass_loss,
                "train_other": other_loss,
                "train_vocals": vocals_loss,
            },
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log_dict(
            {
                "train_drums_sdr": signal_distortion_ratio(
                    source_predict[:, 0], source[:, 0]
                ).mean(),
                "train_bass_sdr": signal_distortion_ratio(
                    source_predict[:, 1], source[:, 1]
                ).mean(),
                "train_other_sdr": signal_distortion_ratio(
                    source_predict[:, 2], source[:, 2]
                ).mean(),
                "train_vocals_sdr": signal_distortion_ratio(
                    source_predict[:, 3], source[:, 3]
                ).mean(),
            },
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

        self.log_dict(
            {
                "train_drums_sisdr": scale_invariant_signal_distortion_ratio(
                    source_predict[:, 0], source[:, 0]
                ).mean(),
                "train_bass_sisdr": scale_invariant_signal_distortion_ratio(
                    source_predict[:, 1], source[:, 1]
                ).mean(),
                "train_other_sisdr": scale_invariant_signal_distortion_ratio(
                    source_predict[:, 2], source[:, 2]
                ).mean(),
                "train_vocals_sisdr": scale_invariant_signal_distortion_ratio(
                    source_predict[:, 3], source[:, 3]
                ).mean(),
            },
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

        self.log_dict(
            {
                "train_drums_usdr": compute_uSDR(
                    source_predict[:, 0], source[:, 0]
                ).mean(),
                "train_bass_usdr": compute_uSDR(
                    source_predict[:, 1], source[:, 1]
                ).mean(),
                "train_other_usdr": compute_uSDR(
                    source_predict[:, 2], source[:, 2]
                ).mean(),
                "train_vocals_usdr": compute_uSDR(
                    source_predict[:, 3], source[:, 3]
                ).mean(),
            },
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        source = batch

        mix = source.sum(dim=1)

        source_predict = self.model(mix)

        drums_loss = self.loss(source_predict[:, 0], source[:, 0]) / 3

        bass_loss = self.loss(source_predict[:, 1], source[:, 1]) / 3

        other_loss = self.loss(source_predict[:, 2], source[:, 2]) / 3

        vocals_loss = self.loss(source_predict[:, 3], source[:, 3]) / 3

        loss = 0.25 * (drums_loss + bass_loss + other_loss + vocals_loss)

        self.log_dict(
            {
                "valid_loss": loss,
                "valid_drums": drums_loss,
                "valid_bass": bass_loss,
                "valid_other": other_loss,
                "valid_vocals": vocals_loss,
            },
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log_dict(
            {
                "valid_drums_sdr": signal_distortion_ratio(
                    source_predict[:, 0], source[:, 0]
                ).mean(),
                "valid_bass_sdr": signal_distortion_ratio(
                    source_predict[:, 1], source[:, 1]
                ).mean(),
                "valid_other_sdr": signal_distortion_ratio(
                    source_predict[:, 2], source[:, 2]
                ).mean(),
                "valid_vocals_sdr": signal_distortion_ratio(
                    source_predict[:, 3], source[:, 3]
                ).mean(),
            },
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

        self.log_dict(
            {
                "valid_drums_sisdr": scale_invariant_signal_distortion_ratio(
                    source_predict[:, 0], source[:, 0]
                ).mean(),
                "valid_bass_sisdr": scale_invariant_signal_distortion_ratio(
                    source_predict[:, 1], source[:, 1]
                ).mean(),
                "valid_other_sisdr": scale_invariant_signal_distortion_ratio(
                    source_predict[:, 2], source[:, 2]
                ).mean(),
                "valid_vocals_sisdr": scale_invariant_signal_distortion_ratio(
                    source_predict[:, 3], source[:, 3]
                ).mean(),
            },
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

        self.log_dict(
            {
                "valid_drums_usdr": compute_uSDR(
                    source_predict[:, 0], source[:, 0]
                ).mean(),
                "valid_bass_usdr": compute_uSDR(
                    source_predict[:, 1], source[:, 1]
                ).mean(),
                "valid_other_usdr": compute_uSDR(
                    source_predict[:, 2], source[:, 2]
                ).mean(),
                "valid_vocals_usdr": compute_uSDR(
                    source_predict[:, 3], source[:, 3]
                ).mean(),
            },
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.RAdam(self.parameters(), lr=0.5 * 3e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=40
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "valid_loss",
        }


if __name__ == "__main__":
    from data.dataset import get_musdb_wav_datasets
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

    train_set = get_musdb_wav_datasets(
        musdb="../musdb18hq", data_type="train", metadata="./metadata", segment=7
    )
    test_set = get_musdb_wav_datasets(
        musdb="../musdb18hq", data_type="test", metadata="./metadata1", segment=7
    )

    train_dl = torch.utils.data.DataLoader(
        train_set, batch_size=6, shuffle=True, drop_last=True, num_workers=2
    )
    valid_dl = torch.utils.data.DataLoader(
        test_set, batch_size=6, shuffle=False, num_workers=2
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_loss", mode="min", save_top_k=1
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    mp_model = My_model(
        train_set=train_set, valid_set=test_set, num_workers=49, batch_size=2
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices="auto",
        max_epochs=1000,
        callbacks=[checkpoint_callback, lr_monitor],
        precision="bf16-mixed",
    )

    trainer.fit(mp_model, train_dl, valid_dl)
