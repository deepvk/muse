import torch
import torch as th
import torch.nn as nn
import math
from torch.nn import functional as F


class DownSample(nn.Module):
    def __init__(
        self,
        input_channel,
        out_channel,
        scale,
        stride,
        padding,
        activation,
        normalization,
    ):
        """
        DownSample - block include layer normalization, layer activation and Conv2d layer
        Args:
            scale - kernel size
            activation - activation layer
            normalization - normalization layer
        """
        super().__init__()

        self.conv_layer = nn.Sequential(
            normalization(input_channel),
            activation,
            nn.Conv2d(
                in_channels=input_channel,
                out_channels=out_channel,
                kernel_size=scale,
                stride=stride,
                padding=padding,
                bias=False,
            ),
        )

    def forward(self, x):
        return self.conv_layer(x)


class UpSample(nn.Module):
    def __init__(
        self,
        input_channel,
        out_channel,
        scale,
        stride,
        padding,
        activation,
        normalization,
    ):
        """
        UpSample - block include layer normalization, layer activation and Conv2d layer
        Args:
            scale - kernel size
            activation - activation layer
            normalization - normalization layer
        """
        super().__init__()

        self.convT_layer = nn.Sequential(
            normalization(input_channel),
            activation,
            nn.ConvTranspose2d(
                in_channels=input_channel,
                out_channels=out_channel,
                kernel_size=scale,
                stride=stride,
                padding=padding,
                bias=False,
            ),
        )

    def forward(self, x):
        return self.convT_layer(x)


class InceptionBlock(nn.Module):
    def __init__(self, input_channel, out_channels, activation, normalization):
        """InceptionBlock - The block includes 3 branches consisting of normalization layers, activation layers and 2d convolution with 1, 3 and 5 core sizes respectively.
        Args:
           activation - activation layer
           normalization - normalization layer
        """
        super().__init__()

        self.conv_layer_1 = nn.Sequential(
            normalization(input_channel),
            activation,
            nn.Conv2d(
                in_channels=input_channel,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        )

        self.conv_layer_2 = nn.Sequential(
            normalization(input_channel),
            activation,
            nn.Conv2d(
                in_channels=input_channel,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding="same",
                bias=False,
            ),
        )

        self.conv_layer_3 = nn.Sequential(
            normalization(input_channel),
            activation,
            nn.Conv2d(
                in_channels=input_channel,
                out_channels=out_channels,
                kernel_size=5,
                stride=1,
                padding="same",
                bias=False,
            ),
        )

    def forward(self, x):
        x1 = self.conv_layer_1(x)
        x2 = self.conv_layer_2(x)
        x3 = self.conv_layer_3(x)
        return torch.concat((x1, x2, x3), dim=1)


class Encoder(nn.Module):
    def __init__(
        self,
        input_channel,
        out_channel,
        scale,
        stride,
        padding,
        activation,
        normalization,
    ):
        """
        Encoder layer - Block included DownSample layer and InceptionBlock.
        Args:
            scale - scale (kernel size) DownSample layer
            stride - stride DownSample layer
            padding - padding DownSample layer
            activation - activation layer
            normalization - normalization layer
        """
        super().__init__()

        self.inception_layer = InceptionBlock(
            input_channel, out_channel, activation, normalization
        )
        self.down_layer = DownSample(
            out_channel * 3,
            out_channel,
            scale,
            stride,
            padding,
            activation,
            normalization,
        )

    def forward(self, x):
        x = self.inception_layer(x)
        x = self.down_layer(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        input_channel,
        out_channel,
        scale,
        stride,
        padding,
        activation,
        normalization,
    ):
        """
        Decoder layer - Block included UpSample layer and InceptionBlock.
        Args:
            scale - scale (kernel size) UpSample layer
            stride - stride UpSample layer
            padding - padding UpSample layer
            activation - activation layer
            normalization - normalization layer
        """
        super().__init__()

        self.inception_layer = InceptionBlock(
            input_channel, out_channel, activation, normalization
        )
        self.up_layer = UpSample(
            out_channel * 3,
            out_channel,
            scale,
            stride,
            padding,
            activation,
            normalization,
        )

    def forward(self, x):
        x = self.inception_layer(x)
        x = self.up_layer(x)
        return x


class BLSTM(nn.Module):
    """
    BiLSTM with same hidden units as input dim.
    If `max_steps` is not None, input will be splitting in overlapping
    chunks and the LSTM applied separately on each chunk.
    """

    def __init__(self, dim, layers=1, max_steps=None, skip=False):
        super().__init__()
        assert max_steps is None or max_steps % 4 == 0
        self.max_steps = max_steps
        self.lstm = nn.LSTM(
            bidirectional=True, num_layers=layers, hidden_size=dim, input_size=dim
        )
        self.linear = nn.Linear(2 * dim, dim)
        self.skip = skip

    def __unfold(self, a, kernel_size, stride):
        """Given input of size [*OT, T], output Tensor of size [*OT, F, K]
        with K the kernel size, by extracting frames with the given stride.

        This will pad the input so that `F = ceil(T / K)`.

        see https://github.com/pytorch/pytorch/issues/60466
        """
        *shape, length = a.shape
        n_frames = math.ceil(length / stride)
        tgt_length = (n_frames - 1) * stride + kernel_size
        a = F.pad(a, (0, tgt_length - length))
        strides = list(a.stride())
        assert strides[-1] == 1, "data should be contiguous"
        strides = strides[:-1] + [stride, 1]
        return a.as_strided([*shape, n_frames, kernel_size], strides)

    def forward(self, x):
        B, C, T = x.shape
        y = x
        framed = False
        if self.max_steps is not None and T > self.max_steps:
            width = self.max_steps
            stride = width // 2
            frames = self.__unfold(x, width, stride)
            nframes = frames.shape[2]
            framed = True
            x = frames.permute(0, 2, 1, 3).reshape(-1, C, width)

        x = x.permute(2, 0, 1)

        x = self.lstm(x)[0]
        x = self.linear(x)
        x = x.permute(1, 2, 0)
        if framed:
            out = []
            frames = x.reshape(B, -1, C, width)
            limit = stride // 2
            for k in range(nframes):
                if k == 0:
                    out.append(frames[:, k, :, :-limit])
                elif k == nframes - 1:
                    out.append(frames[:, k, :, limit:])
                else:
                    out.append(frames[:, k, :, limit:-limit])
            out = torch.cat(out, -1)
            out = out[..., :T]
            x = out
        if self.skip:
            x = x + y
        return x


class Bottleneck_v2(nn.Module):
    def __init__(
        self,
        input_channel,
        out_channel,
        activation,
        normalization,
        layers=2,
        max_steps=200,
        skip=True,
        stride=1,
        padding=1,
    ):
        """
        Bottleneck - bi-lstm bottleneck
        Args:
            activation - activation layer
            normalization - normalization layer
            layers - number of recurrent layers
            skip - include skip conncetion bi-lstm
            stride - stride Conv1d
            padding - stride Conv1d
        """
        super().__init__()

        self.conv_layer = nn.Sequential(
            normalization(input_channel, affine=True),
            activation,
            nn.Conv1d(
                input_channel, out_channel, kernel_size=3, stride=stride, padding="same"
            ),
        )

        self.biLSTM = BLSTM(out_channel, layers=layers, max_steps=max_steps, skip=skip)

        self.conv_layer_1 = nn.Sequential(
            normalization(out_channel, affine=True),
            activation,
            nn.Conv1d(out_channel, input_channel, kernel_size=1, stride=stride),
        )

    def forward(self, x):
        B, C, F, T = x.shape
        x = x.view(B, C * F, T)
        x = self.conv_layer(x)
        x = self.biLSTM(x)
        x = self.conv_layer_1(x)
        x = x.view(B, C, F, T)
        return x


class Bottleneck(nn.Module):
    def __init__(self, input_channel, out_channels, normalization, activation):
        """
        Bottleneck - convolution bottleneck
        """
        super().__init__()

        self.conv_layer_1 = nn.Sequential(
            normalization(input_channel),
            activation,
            nn.Conv2d(
                in_channels=input_channel,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding="same",
                bias=False,
            ),
        )

        self.conv_layer_2 = nn.Sequential(
            normalization(input_channel),
            activation,
            nn.Conv2d(
                in_channels=input_channel,
                out_channels=out_channels,
                kernel_size=2,
                stride=1,
                padding="same",
                bias=False,
            ),
        )

        self.conv_layer_3 = nn.Sequential(
            normalization(input_channel),
            activation,
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,  # padding='same',
                bias=False,
            ),
        )

    def forward(self, x):
        x1 = self.conv_layer_1(x)
        x2 = self.conv_layer_2(x) + x1
        x3 = self.conv_layer_3(x2) + x1

        return x3
