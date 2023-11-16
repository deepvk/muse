import torch
import torch as th
import torch.nn as nn
from model.STFT import STFT
from functools import partial
from model.modules import Encoder, Decoder, Bottleneck_v2, Bottleneck
from typing import List, Optional


class Model_Unet(nn.Module):
    def __init__(
        self,
        depth: int=4,
        source: List[str]=["drums", "bass", "other", "vocals"],
        channel: int=28,
        is_mono: Optional[bool]=False,
        mask_mode: Optional[bool]=False,
        skip_mode: str="concat",
        nfft: int=4096,
        bottlneck_lstm: Optional[bool]=True,
        layers: int=2,
    ):
        """
        depth - (int) number of layers encoder and decoder
        source - (list[str]) list of source names
        channel - (int) initial number of hidden channels
        is_mono - (bool) mono input/output audio channel
        mask_mode - (bool) mask inference
        skip_mode - (concat or add) types skip connection
                concat: concatenates output encoder and decoder
                add: add output encoder and decoder
        nfft - (int) number of fft bins
        bottlneck_lstm - (bool) lstm bottlneck
                True: bottlneck_lstm - bilstm bottlneck
                False: bottlneck_conv - convolution bottlneck
        layers - (int) number bottlneck_lstm layers
        """
        super().__init__()
        self.sources = source
        skip_channel = 2 if skip_mode == "concat" else 1
        self.skip_mode = True if skip_mode == "concat" else False
        stereo = 1 if is_mono else 2
        self.mask_mode = mask_mode

        norm = self.__norm("InstanceNorm2d")
        act = self.__get_act("gelu")

        self.stft = STFT(nfft)
        self.conv_magnitude = nn.Conv2d(
            in_channels=stereo,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            bias=False,
        )

        self.conv_magnitude_final = nn.Conv2d(
            in_channels=channel,
            out_channels=len(source) * stereo,
            kernel_size=1,
            stride=1,
            bias=False,
        )

        self.conv_phase = nn.Conv2d(
            in_channels=stereo,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            bias=False,
        )

        self.conv_phase_final = nn.Conv2d(
            in_channels=channel,
            out_channels=len(source) * stereo,
            kernel_size=1,
            stride=1,
            bias=False,
        )

        self.encoder_magnitude = nn.ModuleList()
        self.decoder_magnitude = nn.ModuleList()

        self.encoder_phase = nn.ModuleList()
        self.decoder_phase = nn.ModuleList()

        for idx in range(depth):
            self.encoder_magnitude.append(
                Encoder(
                    input_channel=channel,
                    out_channel=channel * 2,
                    scale=(4, 1),
                    stride=(4, 1),
                    padding=0,
                    normalization=norm,
                    activation=act,
                )
            )

            self.encoder_phase.append(
                Encoder(
                    input_channel=channel,
                    out_channel=channel * 2,
                    scale=(4, 1),
                    stride=(4, 1),
                    padding=0,
                    normalization=norm,
                    activation=act,
                )
            )
            channel *= 2

        if bottlneck_lstm:
            self.bottleneck_magnitude = Bottleneck_v2(
                input_channel=channel * (nfft // 2) // (2 ** (2 * depth)),
                out_channel=channel,
                normalization=nn.InstanceNorm1d,
                activation=act,
                layers=layers,
            )
        else:
            self.bottleneck_magnitude = Bottleneck(
                input_channel=channel,
                out_channels=channel,
                normalization=norm,
                activation=act,
            )

        self.bottleneck_phase = Bottleneck(channel, channel, norm, act)

        for idx in range(depth):
            self.decoder_magnitude.append(
                Decoder(
                    input_channel=channel * skip_channel,
                    out_channel=channel // 2,
                    scale=(4, 1),
                    stride=(4, 1),
                    padding=0,
                    normalization=norm,
                    activation=act,
                )
            )
            self.decoder_phase.append(
                Decoder(
                    input_channel=channel * skip_channel,
                    out_channel=channel // 2,
                    scale=(4, 1),
                    stride=(4, 1),
                    padding=0,
                    normalization=norm,
                    activation=act,
                )
            )
            channel //= 2

    def __wave2feature(self, z: torch.Tensor):
        # z = self.stft.stft(wave)
        phase = th.atan2(z.imag, z.real)
        magnitude = z.abs()
        return magnitude, phase

    def __get_act(self, act_type: str):
        if act_type == "gelu":
            return nn.GELU()
        elif act_type == "relu":
            return nn.ReLU()
        elif act_type[:3] == "elu":
            alpha = float(act_type.replace("elu", ""))
            return nn.ELU(alpha)
        else:
            raise Exception

    def __norm(self, norm_type: str):
        if norm_type == "BatchNorm":
            return nn.BatchNorm2d
        elif norm_type == "InstanceNorm2d":
            return nn.InstanceNorm2d
        elif norm_type == "InstanceNorm1d":
            return nn.InstanceNorm1d
        else:
            return nn.Identity()

    def __normal(self, x: torch.Tensor):  # normalization input signal
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (1e-5 + std)
        return mean, std, x

    def forward(self, x: torch.Tensor):
        # length_wave = x.shape[-1]

        x_m, x_p = self.__wave2feature(x)

        B, C, Fq, T = x_m.shape
        S = len(self.sources)

        # normalization magnitude input
        mean_m, std_m, x_m = self.__normal(x_m)
        x_mix = x_m
        # normalization magnitude phase input
        mean_p, std_p, x_p = self.__normal(x_p)

        skip_m = []  # skip connection magnitude branch
        skip_p = []  # skip connection phase branch

        x_m = self.conv_magnitude(x_m)  # start conv magnitude
        x_p = self.conv_phase(x_p)  # start conv phase

        for idx_enc in range(len(self.encoder_magnitude)):
            x_m = self.encoder_magnitude[idx_enc](x_m)  # encoder layer magnitude
            x_p = self.encoder_phase[idx_enc](x_p)  # encoder layer phase

            skip_m.append(x_m)  # skip magnitude
            skip_p.append(x_p)  # skip phase

        x_m = self.bottleneck_magnitude(x_m)
        x_p = self.bottleneck_phase(x_p)

        for idx in range(len(self.decoder_magnitude)):
            if self.skip_mode:
                x_m = self.decoder_magnitude[idx](
                    torch.concat((x_m, skip_m[-idx - 1]), dim=1)
                )  # decoder layer magnitude
                x_p = self.decoder_phase[idx](
                    torch.concat((x_p, skip_p[-idx - 1]), dim=1)
                )  # decoder layer phase
            else:
                x_m = self.decoder_magnitude[idx](x_m + skip_m[-idx - 1])
                x_p = self.decoder_phase[idx](x_p + skip_p[-idx - 1])

        x_m = self.conv_magnitude_final(x_m)  # final conv magnitude
        x_p = self.conv_phase_final(x_p)

        x_m = x_m.view(B, S, -1, Fq, T)
        x_p = x_p.view(B, S, -1, Fq, T)

        if self.mask_mode:
            mask = nn.functional.softmax(x_m.view(B, S, -1, Fq, T), dim=1)
            x_m = x_mix.view(B, 1, C, Fq, T) * mask

        x_m = x_m * std_m[:, None] + mean_m[:, None]
        x_p = x_p * std_p[:, None] + mean_p[:, None]

        imag = x_m * th.sin(x_p)
        real = x_m * th.cos(x_p)
        z = th.complex(real, imag)

        return z
        # return self.stft.istft(z, length_wave)
