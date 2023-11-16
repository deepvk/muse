import torch
import torch as th
from typing import Tuple, Optional, Union
import math
import torch.nn.functional as F


class STFT:
    def __init__(self, n_fft: int=4096, pad: int=0):
        self.n_fft = n_fft
        self.pad = pad
        self.hop_length = self.n_fft // 4

    def __pad1d(
        self,
        x: torch.Tensor,
        paddings: Tuple[int, int],
        mode: str = "constant",
        value: float = 0.0,
    ):
        """Tiny wrapper around F.pad, just to allow for reflect padding on small input.
        If this is the case, we insert extra 0 padding to the right before the reflection happen.
        """
        x0 = x
        length = x.shape[-1]
        padding_left, padding_right = paddings
        if mode == "reflect":
            max_pad = max(padding_left, padding_right)
            if length <= max_pad:
                extra_pad = max_pad - length + 1
                extra_pad_right = min(padding_right, extra_pad)
                extra_pad_left = extra_pad - extra_pad_right
                paddings = (
                    padding_left - extra_pad_left,
                    padding_right - extra_pad_right,
                )
                x = F.pad(x, (extra_pad_left, extra_pad_right))
        out = F.pad(x, paddings, mode, value)
        assert out.shape[-1] == length + padding_left + padding_right
        assert (out[..., padding_left : padding_left + length] == x0).all()
        return out

    def _spec(self, x: torch.Tensor):
        *other, length = x.shape
        x = x.reshape(-1, length)
        z = th.stft(
            x,
            self.n_fft * (1 + self.pad),
            self.hop_length or self.n_fft // 4,
            window=th.hann_window(self.n_fft).to(x),
            win_length=self.n_fft,
            normalized=True,
            center=True,
            return_complex=True,
            pad_mode="reflect",
        )
        _, freqs, frame = z.shape
        return z.view(*other, freqs, frame)

    def _ispec(self, z: torch.Tensor, length: int):
        *other, freqs, frames = z.shape
        n_fft = 2 * freqs - 2
        z = z.view(-1, freqs, frames)
        win_length = n_fft // (1 + self.pad)
        is_mps = z.device.type == "mps"
        if is_mps:
            z = z.cpu()
        z = th.istft(
            z,
            n_fft,
            self.hop_length,
            window=th.hann_window(win_length).to(z.real),
            win_length=win_length,
            normalized=True,
            length=length,
            center=True,
        )
        _, length = z.shape
        return z.view(*other, length)

    def stft(self, x: torch.Tensor):
        hl = self.hop_length
        x0 = x  # noqa
        le = int(math.ceil(x.shape[-1] / self.hop_length))
        pad = hl // 2 * 3
        x = self.__pad1d(
            x, (pad, pad + le * self.hop_length - x.shape[-1]), mode="reflect"
        )
        z = self._spec(x)[..., :-1, :]
        z = z[..., 2 : 2 + le]
        return z

    def istft(self, z: torch.Tensor, length: int=0, scale: Optional[int]=0):
        hl = self.hop_length // (4**scale)
        z = F.pad(z, (0, 0, 0, 1))
        z = F.pad(z, (2, 2))
        pad = hl // 2 * 3
        le = hl * int(math.ceil(length / hl)) + 2 * pad

        x = self._ispec(z, length=le)

        x = x[..., pad : pad + length]
        return x
