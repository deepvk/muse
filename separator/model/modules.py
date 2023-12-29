import torch
import torch.nn as nn
import math
from torch.nn import functional as F


class DownSample(nn.Module):
    """
    DownSample - dimensionality reduction block that includes layer normalization, activation layer, and Conv2d layer.
    Args:
        input_channel (int): Number of input channels.
        out_channel (int): Number of output channels.
        scale (int, tuple): Kernel size.
        stride (int, tuple): Stride of the convolution.
        padding (int, tuple or str): Padding added to all four sides of the input.
        activation (object): Activation layer.
        normalization (object): Normalization layer.
    """

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
    """
    UpSample - dimensionality boosting block that includes layer normalization, activation layer, and Conv2d layer.
    Args:
        input_channel (int): Number of input channels.
        out_channel (int): Number of output channels.
        scale (int, tuple): Kernel size.
        stride (int, tuple): Stride of the convolution.
        padding (int, tuple or str): Padding added to all four sides of the input.
        activation (object): Activation layer.
        normalization (object): Normalization layer.
    """

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
    """
    InceptionBlock: This block comprises three branches, each consisting of normalization layers, activation layers, and 2D convolution layers. The convolution layers in each branch have kernel sizes of 1, 3, and 5, respectively.
    Args:
        input_channel (int): Number of input channels.
        out_channel (int): Number of output channels.
        activation (object): Activation layer.
        normalization (object): Normalization layer.
    """

    def __init__(self, input_channel, out_channel, activation, normalization):
        super().__init__()

        self.conv_layer_1 = nn.Sequential(
            normalization(input_channel),
            activation,
            nn.Conv2d(
                in_channels=input_channel,
                out_channels=out_channel,
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
                out_channels=out_channel,
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
                out_channels=out_channel,
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
    """
    Encoder layer - Block included DownSample layer and InceptionBlock.
    Args:
        input_channel (int): Number of input channels.
        out_channel (int): Number of output channels.
        scale (int, tuple): The size of the kernel used in the DownSample layer.
        stride (int, tuple): The stride used in the DownSample layer.
        padding (int, tuple or str): Padding added to all four sides of the input.
        activation (object): Activation layer.
        normalization (object): Normalization layer.
    """

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
    """
    Decoder layer - Block included UpSample layer and InceptionBlock.
    Args:
        input_channel (int): Number of input channels.
        out_channel (int): Number of output channels.
        scale (int, tuple): The size of the kernel used in the UpSample layer.
        stride (int, tuple): The stride used in the UpSample layer.
        padding (int, tuple or str): Padding added to all four sides of the input.
        activation (object): Activation layer.
        normalization (object): Normalization layer.
    """

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
    A bidirectional LSTM (BiLSTM) module with the same number of hidden units as the input dimension.
    This module can process inputs in overlapping chunks if `max_steps` is specified.
    In this case, the input will be split into chunks, and the LSTM will be applied to each chunk separately.
    Args:
        dim (int): The number of dimensions in the input and the hidden state of the LSTM.
        max_steps (int, optional): The maximum number of steps (length of chunks) for processing the input. Defaults to None.
        skip (bool, optional): Flag to enable skip connections. Defaults to False.
        layers (int): Number of recurrent layers
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
    """
    Bottleneck - bi-lstm bottleneck
    Args:
        input_channel (int): Number of input channels.
        out_channel (int): Number of output channels.
        layers (int): number of recurrent layers
        skip (bool): include skip conncetion bi-lstm
        stride (int, tuple): The stride used in the Conv1d layer.
        padding (int, tuple or str): Padding added to all four sides of the input.
        activation (object): Activation layer.
        normalization (object): Normalization layer.
    """

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
        padding="same",
    ):
        super().__init__()

        self.conv_layer = nn.Sequential(
            normalization(input_channel, affine=True),
            activation,
            nn.Conv1d(
                input_channel,
                out_channel,
                kernel_size=3,
                stride=stride,
                padding=padding,
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
    """
    Bottleneck - convolution bottleneck
    Args:
        input_channel (int): Number of input channels.
        out_channel (int): Number of output channels.
        activation (object): Activation layer.
        normalization (object): Normalization layer.
    """

    def __init__(self, input_channel, out_channels, normalization, activation):
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
                stride=1,
                bias=False,
            ),
        )

    def forward(self, x):
        x1 = self.conv_layer_1(x)
        x2 = self.conv_layer_2(x) + x1
        x3 = self.conv_layer_3(x2) + x1

        return x3
