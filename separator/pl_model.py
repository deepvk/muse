from model.PM_Unet import Model_Unet

from train.loss import MultiResSpecLoss

from train import augment

from pathlib import Path
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import ScaleInvariantSignalDistortionRatio
from torchmetrics.functional.audio import (
    scale_invariant_signal_distortion_ratio,
    signal_distortion_ratio,
)


class PM_model(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.model = Model_Unet(
            depth=config.model_depth,
            source=config.model_source,
            channel=config.model_channel,
            is_mono=config.is_mono,
            mask_mode=config.mask_mode,
            skip_mode=config.skip_mode,
            nfft=config.nfft,
            bottlneck_lstm=config.bottlneck_lstm,
            layers=config.layers,
            stft_flag=config.stft_flag,
        )

        # loss
        # Loss = (L_1 + L_{MRS} - L_{SISDR})
        self.criterion_1 = nn.L1Loss()
        self.criterion_2 = MultiResSpecLoss(
            factor=config.factor,
            f_complex=config.c_factor,
            gamma=config.gamma,
            n_ffts=config.loss_nfft,
        )
        self.criterion_3 = ScaleInvariantSignalDistortionRatio()

        # augment
        self.augment = [
            augment.Shift(proba=config.proba_shift, shift=config.shift, same=True)
        ]
        self.augment += [
            augment.PitchShift(
                proba=config.pitchshift_proba,
                min_semitones=config.vocals_min_semitones,
                max_semitones=config.vocals_max_semitones,
                min_semitones_other=config.other_min_semitones,
                max_semitones_other=config.other_max_semitones,
                flag_other=config.pitchshift_flag_other,
            ),
            augment.TimeChange(
                factors_list=config.time_change_factors, proba=config.time_change_proba
            ),
            augment.FlipChannels(proba=config.proba_flip_channel),
            augment.FlipSign(proba=config.proba_flip_sign),
            augment.Remix(proba=config.remix_proba, group_size=config.remix_group_size),
            augment.Scale(
                proba=config.scale_proba, min=config.scale_min, max=config.scale_max
            ),
            augment.FadeMask(proba=config.fade_mask_proba),
            augment.Double(proba=config.double_proba),
            augment.Reverse(proba=config.reverse_proba),
            augment.RemixWave(
                proba=config.mushap_proba, group_size=config.mushap_depth
            ),
        ]
        self.augment = torch.nn.Sequential(*self.augment)

        self.model.apply(self.__init_weights)

    def __init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            torch.nn.init.xavier_uniform(m.weight)

    def __usdr(self, predT, tgtT, delta=1e-7):
        """
        latex: $usdr=10\log_{10} (\dfrac{\| tgtT\|^2 + \delta}{  \| predT - tgtT\| ^{2} + \delta})$
        """
        num = torch.sum(torch.square(tgtT), dim=(1, 2))
        den = torch.sum(torch.square(tgtT - predT), dim=(1, 2))
        num += delta
        den += delta
        usdr = 10 * torch.log10(num / den)
        return usdr.mean()

    def forward(self, x):
        x = self.model(x)
        return x

    def loss(self, y_true, y_pred):
        # losses are averaged
        loss = (
            self.criterion_1(y_pred, y_true)
            + self.criterion_2(y_pred, y_true)
            - self.criterion_3(y_pred, y_true)
        ) / 3
        return loss

    def training_step(self, batch, batch_idx):
        source = batch
        source = self.augment(source)
        mix = source.sum(dim=1)

        source_predict = self.model(mix)

        drums_pred, drums_target = source_predict[:, 0], source[:, 0]
        bass_pred, bass_target = source_predict[:, 1], source[:, 1]
        other_pred, other_target = source_predict[:, 2], source[:, 2]
        vocals_pred, vocals_target = source_predict[:, 3], source[:, 3]

        drums_loss = self.loss(drums_pred, drums_target)

        bass_loss = self.loss(bass_pred, bass_target)

        other_loss = self.loss(other_pred, other_target)

        vocals_loss = self.loss(vocals_pred, vocals_target)

        loss = 0.25 * (
            drums_loss + bass_loss + other_loss + vocals_loss
        )  # losses averaged across sources

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
                    drums_pred, drums_target
                ).mean(),
                "train_bass_sdr": signal_distortion_ratio(
                    bass_pred, bass_target
                ).mean(),
                "train_other_sdr": signal_distortion_ratio(
                    other_pred, other_target
                ).mean(),
                "train_vocals_sdr": signal_distortion_ratio(
                    vocals_pred, vocals_target
                ).mean(),
            },
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

        self.log_dict(
            {
                "train_drums_sisdr": scale_invariant_signal_distortion_ratio(
                    drums_pred, drums_target
                ).mean(),
                "train_bass_sisdr": scale_invariant_signal_distortion_ratio(
                    bass_pred, bass_target
                ).mean(),
                "train_other_sisdr": scale_invariant_signal_distortion_ratio(
                    other_pred, other_target
                ).mean(),
                "train_vocals_sisdr": scale_invariant_signal_distortion_ratio(
                    vocals_pred, vocals_target
                ).mean(),
            },
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

        self.log_dict(
            {
                "train_drums_usdr": self.__usdr(drums_pred, drums_target).mean(),
                "train_bass_usdr": self.__usdr(bass_pred, bass_target).mean(),
                "train_other_usdr": self.__usdr(other_pred, other_target).mean(),
                "train_vocals_usdr": self.__usdr(vocals_pred, vocals_target).mean(),
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
        drums_pred, drums_target = source_predict[:, 0], source[:, 0]
        bass_pred, bass_target = source_predict[:, 1], source[:, 1]
        other_pred, other_target = source_predict[:, 2], source[:, 2]
        vocals_pred, vocals_target = source_predict[:, 3], source[:, 3]

        drums_loss = self.loss(drums_pred, drums_target)

        bass_loss = self.loss(bass_pred, bass_target)

        other_loss = self.loss(other_pred, other_target)

        vocals_loss = self.loss(vocals_pred, vocals_target)

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
                    drums_pred, drums_target
                ).mean(),
                "valid_bass_sdr": signal_distortion_ratio(
                    bass_pred, bass_target
                ).mean(),
                "valid_other_sdr": signal_distortion_ratio(
                    other_pred, other_target
                ).mean(),
                "valid_vocals_sdr": signal_distortion_ratio(
                    vocals_pred, vocals_target
                ).mean(),
            },
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

        self.log_dict(
            {
                "valid_drums_sisdr": scale_invariant_signal_distortion_ratio(
                    drums_pred, drums_target
                ).mean(),
                "valid_bass_sisdr": scale_invariant_signal_distortion_ratio(
                    bass_pred, bass_target
                ).mean(),
                "valid_other_sisdr": scale_invariant_signal_distortion_ratio(
                    other_pred, other_target
                ).mean(),
                "valid_vocals_sisdr": scale_invariant_signal_distortion_ratio(
                    vocals_pred, vocals_target
                ).mean(),
            },
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

        self.log_dict(
            {
                "valid_drums_usdr": self.__usdr(drums_pred, drums_target).mean(),
                "valid_bass_usdr": self.__usdr(bass_pred, bass_target).mean(),
                "valid_other_usdr": self.__usdr(other_pred, other_target).mean(),
                "valid_vocals_usdr": self.__usdr(vocals_pred, vocals_target).mean(),
            },
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.RAdam(self.parameters(), lr=config.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=config.T_0
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "valid_loss",
        }


def main(config):
    from data.dataset import get_musdb_wav_datasets
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

    Path(config.musdb_path).mkdir(exist_ok=True, parents=True)
    Path(config.metadata_train_path).mkdir(exist_ok=True, parents=True)
    train_set = get_musdb_wav_datasets(
        musdb=config.musdb_path,
        data_type="train",
        metadata=config.metadata_train_path,
        segment=config.segment,
    )

    Path(config.metadata_test_path).mkdir(exist_ok=True, parents=True)
    test_set = get_musdb_wav_datasets(
        musdb=config.musdb_path,
        data_type="test",
        metadata=config.metadata_test_path,
        segment=config.segment,
    )

    train_dl = torch.utils.data.DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=config.shuffle_train,
        drop_last=config.drop_last,
        num_workers=config.num_workers,
    )
    valid_dl = torch.utils.data.DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=config.shuffle_valid,
        num_workers=config.num_workers,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_loss",
        mode=config.metric_monitor_mode,
        save_top_k=config.save_top_k_model_weights,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    mp_model = PM_model(config)

    trainer = pl.Trainer(
        accelerator="gpu" if config.device == "cuda" else "cpu",
        devices="auto",
        max_epochs=config.max_epochs,
        callbacks=[checkpoint_callback, lr_monitor],
        precision=config.precision,
        gradient_clip_val=config.grad_clip,
    )

    trainer.fit(mp_model, train_dl, valid_dl)


if __name__ == "__main__":
    from config.config import TrainConfig

    config = TrainConfig()
    main(config)
