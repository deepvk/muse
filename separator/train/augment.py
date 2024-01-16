import random
import torchaudio
import torch as th
from torch import nn
from torch_audiomentations import PitchShift as ps


class Shift(nn.Module):
    """
    Shifts audio in time for data augmentation during training. Applies a random shift up to 'shift' samples.
    If 'same' is True, all sources in a batch are shifted by the same amount; otherwise, each is shifted differently.

    Args:
        proba (float): Probability of applying the shift.
        shift (int): Maximum number of samples for the shift. Defaults to 8192.
        same (bool): Apply the same shift to all sources in a batch. Defaults to False.
    """

    def __init__(self, proba=1, shift=8192, same=False):
        super().__init__()
        self.shift = shift
        self.same = same
        self.proba = proba

    def forward(self, wav):
        if self.shift < 1:
            return wav

        batch, sources, channels, time = wav.size()
        length = time - self.shift

        if random.random() < self.proba:
            srcs = 1 if self.same else sources
            offsets = th.randint(self.shift, [batch, srcs, 1, 1], device=wav.device)
            offsets = offsets.expand(-1, sources, channels, -1)
            indexes = th.arange(length, device=wav.device)
            wav = wav.gather(3, indexes + offsets)
        return wav


class FlipChannels(nn.Module):
    """
    Flip left-right channels.
    Args:
        proba (float): Probability of applying the flip left-right channels.
    """

    def __init__(self, proba=1):
        super().__init__()
        self.proba = proba

    def forward(self, wav):
        batch, sources, channels, time = wav.size()
        if wav.size(2) == 2:
            if random.random() < self.proba:
                left = th.randint(2, (batch, sources, 1, 1), device=wav.device)
                left = left.expand(-1, -1, -1, time)
                right = 1 - left
                wav = th.cat([wav.gather(2, left), wav.gather(2, right)], dim=2)
        return wav


class FlipSign(nn.Module):
    """
    Random sign flip.
    Args:
        proba (float): Probability of applying the sign flip.
    """

    def __init__(self, proba=1):
        super().__init__()

        self.proba = proba

    def forward(self, wav):
        batch, sources, channels, time = wav.size()
        if random.random() < self.proba:
            signs = th.randint(
                2, (batch, sources, 1, 1), device=wav.device, dtype=th.float32
            )
            wav = wav * (2 * signs - 1)
        return wav


class Remix(nn.Module):
    """
    Randomly shuffles sources within each batch during training to create new mixes. Shuffling is performed within groups.
    Args:
        proba (float): Probability of applying the shuffle.
        group_size (int): Size of groups within which shuffling occurs.
    """

    def __init__(self, proba=1, group_size=4):
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
    """
    Scales the amplitude of the audio waveform during training. The scaling factor is chosen randomly within a specified range.
    Args:
        proba (float): Probability of applying the scaling.
        min (float): Minimum scaling factor.
        max (float): Maximum scaling factor.
    """

    def __init__(self, proba=1.0, min=0.25, max=1.25):
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
    Applies time-domain masking to the spectrogram for data augmentation.
    Args:
        proba (float): Probability of applying the mask.
        sample_rate (int): Sample rate of the audio.
        time_mask_param (int): Maximum possible length in seconds of the mask.
    """

    def __init__(self, proba=1, sample_rate=44100, time_mask_param=2):
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


class PitchShift(nn.Module):  # input -> tensor
    """
    Applies pitch shifting to audio sources. The pitch is shifted up or down without changing the tempo.
    Args:
        proba (float): Probability of applying the pitch shift.
        min_semitones (int): Min shift for vocal source.
        max_semitones (int): Max shift for vocal source.
        min_semitones_other (int): Min shift for other sources.
        max_semitones_other (int): Max shift for other sources.
        sample_rate (int): Sample rate of audio.
        flag_other (bool): Apply augmentation to other sources.
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
        super().__init__()
        self.pitch_vocals = ps(
            p=proba,
            min_transpose_semitones=min_semitones,
            max_transpose_semitones=max_semitones,
            sample_rate=sample_rate,
        )

        self.flag_other = flag_other
        if flag_other:
            self.pitch_other = ps(
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


class TimeChange(nn.Module):
    """
    Changes the speed or duration of the signal without affecting the pitch.
    Args:
        factors_list (list): List of factors to adjust speed.
        proba (float): Probability of applying the time change.
        sample_rate (int): Sample rate of audio.
    """

    def __init__(self, factors_list, proba=1, sample_rate=44100):
        super().__init__()
        self.sample_rate = sample_rate
        self.proba = proba

        self.time = torchaudio.transforms.SpeedPerturbation(
            orig_freq=sample_rate, factors=factors_list
        )

    def forward(self, wav):
        if random.random() < self.proba:
            wav, _ = self.time(wav)

        return wav


class Double(nn.Module):
    """
    With equal probability, makes both channels the same as either the left or right original channel.
    Args:
        proba (float): Probability of applying the doubling.
    """

    def __init__(self, proba=1):
        super().__init__()
        self.proba = proba

    def forward(self, wav):
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
    Reverses a segment of the vocal source along the time axis.
    Args:
        proba (float): Probability of applying the reversal.
        min_band_part (float): Minimum fraction of the track to be inverted.
        max_band_part (float): Maximum fraction of the track to be inverted."""

    def __init__(self, proba=1, min_band_part=0.2, max_band_part=0.4):
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


class RemixWave(nn.Module):
    """
    Creates a mashup track within a batch.
    Args:
        proba (float): Probability of applying the mashup.
        group_size (int): Group size for mashup.
        mix_depth (int): Number of tracks to mix.
    """

    def __init__(self, proba=1, group_size=4, mix_depth=2):
        super().__init__()
        self.proba = proba
        self.remix = Remix(proba=1, group_size=group_size)
        self.mix_depth = mix_depth

    def forward(self, wav):
        if random.random() < self.proba:
            mix = wav.clone()
            for i in range(self.mix_depth):
                mix += self.remix(wav)
            return mix
        else:
            return wav


class RemixChannel(nn.Module):
    """
    Shuffles source channels within a batch.
    Args:
        proba (float): Probability of applying the channel shuffle.
    """

    def __init__(self, proba=1):
        super().__init__()

        self.proba = proba

    def forward(self, wav):
        batch, streams, channels, time = wav.size()
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
