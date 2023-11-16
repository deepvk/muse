import random
import torchaudio
import torch as th
from torch import nn
from torch_audiomentations import PitchShift


class Shift(nn.Module):
    """
    Randomly shift audio in time by up to `shift` samples.
    """

    def __init__(self, shift=8192, same=False):
        super().__init__()
        self.shift = shift
        self.same = same

    def forward(self, wav):
        batch, sources, channels, time = wav.size()
        length = time - self.shift
        if self.shift > 0:
            if not self.training:
                wav = wav[..., :length]
            else:
                srcs = 1 if self.same else sources
                offsets = th.randint(self.shift, [batch, srcs, 1, 1], device=wav.device)
                offsets = offsets.expand(-1, sources, channels, -1)
                indexes = th.arange(length, device=wav.device)
                wav = wav.gather(3, indexes + offsets)
        return wav


class FlipChannels(nn.Module):
    """
    Flip left-right channels.
    """

    def forward(self, wav):
        batch, sources, channels, time = wav.size()
        if self.training and wav.size(2) == 2:
            left = th.randint(2, (batch, sources, 1, 1), device=wav.device)
            left = left.expand(-1, -1, -1, time)
            right = 1 - left
            wav = th.cat([wav.gather(2, left), wav.gather(2, right)], dim=2)
        return wav


class FlipSign(nn.Module):
    """
    Random sign flip.
    """

    def forward(self, wav):
        batch, sources, channels, time = wav.size()
        if self.training:
            signs = th.randint(
                2, (batch, sources, 1, 1), device=wav.device, dtype=th.float32
            )
            wav = wav * (2 * signs - 1)
        return wav


class Remix(nn.Module):
    """
    Shuffle sources to make new mixes.
    """

    def __init__(self, proba=1, group_size=4):
        """
        Shuffle sources within one batch.
        Each batch is divided into groups of size `group_size` and shuffling is done within
        each group separatly.
        """
        super().__init__()
        self.proba = proba
        self.group_size = group_size

    def forward(self, wav):
        batch, streams, channels, time = wav.size()
        device = wav.device

        if self.training and random.random() < self.proba:
            group_size = self.group_size or batch
            if batch % group_size != 0:
                raise ValueError(
                    f"Batch size {batch} must be divisible by group size {group_size}"
                )
            groups = batch // group_size
            wav = wav.view(groups, group_size, streams, channels, time)
            permutations = th.argsort(
                th.rand(groups, group_size, streams, 1, 1, device=device), dim=1
            )
            wav = wav.gather(1, permutations.expand(-1, -1, -1, channels, time))
            wav = wav.view(batch, streams, channels, time)
        return wav


class Scale(nn.Module):
    def __init__(self, proba=1.0, min=0.25, max=1.25):
        """
        Args:
            time_mask_param - maximum possible length seconds of the mask
        """
        super().__init__()
        self.proba = proba
        self.min = min
        self.max = max

    def forward(self, wav):
        batch, streams, channels, time = wav.size()
        device = wav.device
        if self.training and random.random() < self.proba:
            scales = th.empty(batch, streams, 1, 1, device=device).uniform_(
                self.min, self.max
            )
            wav *= scales
        return wav


class FadeMask(nn.Module):
    """
    Apply masking to a spectrogram in the time domain.
    https://pytorch.org/audio/main/generated/torchaudio.transforms.TimeMasking.html
    """

    def __init__(self, proba=1, sample_rate=44100, time_mask_param=2):
        """
        Args:
            time_mask_param - maximum possible length seconds of the mask
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.time_mask = torchaudio.transforms.TimeMasking(
            time_mask_param=sample_rate * time_mask_param
        )
        self.proba = proba

    def forward(self, wav):
        if random.random() < self.proba:
            wav = wav.clone()
            wav[:, 0] = self.time_mask(wav[:, 0])
            wav[:, 1] = self.time_mask(wav[:, 1])
            wav[:, 2] = self.time_mask(wav[:, 2])
            wav[:, 3] = self.time_mask(wav[:, 3])

        return wav  # output -> tensor


class PitchShift_f(nn.Module):  # input -> tensor
    """
    Pitch shift the sound up or down without changing the tempo.
    https://github.com/asteroid-team/torch-audiomentations/blob/main/torch_audiomentations/augmentations/pitch_shift.py
    """

    def __init__(
        self,
        proba=1,
        min_semitones=-5,
        max_semitones=5,
        min_semitones_other=-2,
        max_semitones_other=2,
        sample_rate=44100,
        flag_other=False,
    ):
        """
        Args:
            min_semitones - vocal source
            max_semitones - vocals source
            min_semitones_other - drums, bass, other source
            max_semitones_other - drums, bass, other source
            flag_other - apply augmentation other sources
        """
        super().__init__()
        self.pitch_vocals = PitchShift(
            p=proba,
            min_transpose_semitones=min_semitones,
            max_transpose_semitones=max_semitones,
            sample_rate=sample_rate,
        )

        self.flag_other = flag_other
        if flag_other:
            self.pitch_other = PitchShift(
                p=proba,
                min_transpose_semitones=min_semitones_other,
                max_transpose_semitones=max_semitones_other,
                sample_rate=sample_rate,
            )

    def forward(self, wav):
        wav = wav.clone()
        if self.flag_other:
            wav[:, 0] = self.pitch_other(wav[:, 0])
            wav[:, 1] = self.pitch_other(wav[:, 1])
            wav[:, 2] = self.pitch_other(wav[:, 2])
        wav[:, 3] = self.pitch_vocals(wav[:, 3])

        return wav


class TimeChange_f(nn.Module):
    """
    Changes the speed or duration of the signal without changing the pitch.
    https://pytorch.org/audio/stable/generated/torchaudio.transforms.SpeedPerturbation.html
    """

    def __init__(self, factors_list, proba=1, sample_rate=44100):
        super().__init__()
        self.sample_rate = sample_rate
        self.proba = proba

        factors_list = [0.8, 0.85, 0.9, 0.95, 1.05, 1.1, 1.15, 1.2]
        self.time = torchaudio.transforms.SpeedPerturbation(
            orig_freq=sample_rate, factors=factors_list
        )

    def forward(self, wav):
        if random.random() < self.proba:
            wav, _ = self.time(wav)

        return wav


# new augment


class Double(nn.Module):
    """
    With equal probability makes both channels the same to left/right original channel.
    """

    def __init__(self, proba=1):
        super().__init__()
        self.proba = proba

    def forward(self, wav):
        num_samples = wav.shape[-1]

        if random.random() < self.proba:
            wav = wav.clone()

            if random.random() < 0.5:
                wav[:, 0][:, 1] = wav[:, 0][:, 0]
                wav[:, 1][:, 1] = wav[:, 1][:, 0]
                wav[:, 2][:, 1] = wav[:, 2][:, 0]
                wav[:, 3][:, 1] = wav[:, 3][:, 0]
            else:
                wav[:, 0][:, 0] = wav[:, 0][:, 1]
                wav[:, 1][:, 0] = wav[:, 1][:, 1]
                wav[:, 2][:, 0] = wav[:, 2][:, 1]
                wav[:, 3][:, 0] = wav[:, 3][:, 1]

        return wav


class Reverse(nn.Module):
    """
    Reverse (invert) the vocal source along the time axis
    """

    def __init__(self, proba=1, min_band_part=0.2, max_band_part=0.4):
        """
        Args:
            min_band_part - minimum track share inversion
            max_band_part - maximum track share inversion
        """
        super().__init__()
        self.proba = proba
        self.min_band_part = min_band_part
        self.max_band_part = max_band_part

    def forward(self, wav):
        num_samples = wav.shape[-1]

        if random.random() < self.proba:
            wav = wav.clone()

            end = random.randint(
                int(num_samples * self.min_band_part),
                int(num_samples * self.max_band_part),
            )
            start = random.randint(0, num_samples - end)
            wav[..., start : end + start][:, 3] = th.flip(
                wav[..., start : end + start][:, 3], [2]
            )

        return wav


class Remix_wave(nn.Module):
    """
    Mashup track in group
    """

    def __init__(self, proba=1, group_size=4, mix_depth=2):
        """
        Args:
            group_size - group size
            mix_depth - number mashup track
        """
        super().__init__()
        self.proba = proba
        self.remix = Remix(proba=1, group_size=group_size)
        self.mix_depth = mix_depth

    def forward(self, wav):
        if random.random() < self.proba:
            batch, streams, channels, time = wav.size()
            device = wav.device
            mix = wav.clone()
            for i in range(self.mix_depth):
                mix += self.remix(wav)
            return mix
        else:
            return wav


class Remix_channel(nn.Module):
    """
    Shuffle sources channels within one batch
    """

    def __init__(self, proba=1):
        super().__init__()

        self.proba = proba

    def forward(self, wav):
        batch, streams, channels, time = wav.size()
        device = wav.device
        if self.training and random.random() < self.proba:
            drums = wav[:, 0].reshape(-1, time)
            bass = wav[:, 1].reshape(-1, time)
            other = wav[:, 2].reshape(-1, time)
            vocals = wav[:, 3].reshape(-1, time)

            s0 = drums[th.randperm(drums.size()[0])].view(batch, 1, 2, time)
            s1 = bass[th.randperm(bass.size()[0])].view(batch, 1, 2, time)
            s2 = other[th.randperm(other.size()[0])].view(batch, 1, 2, time)
            s3 = vocals[th.randperm(vocals.size()[0])].view(batch, 1, 2, time)

            return th.concat((s0, s1, s2, s3), dim=1)
        else:
            return wav
