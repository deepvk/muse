import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from model.utils import STFT
import math
from torch.nn import functional as F

from functools import partial

def get_norm(norm_type):
    def norm(c, norm_type):   
        if norm_type=='BatchNorm':
            return nn.BatchNorm2d(c)
        elif norm_type=='InstanceNorm2d':
            return nn.InstanceNorm2d(c, affine=True)
        elif norm_type=='InstanceNorm1d':
            return nn.InstanceNorm1d(c, affine=True)
        elif 'GroupNorm' in norm_type:
            g = int(norm_type.replace('GroupNorm', ''))
            return nn.GroupNorm(num_groups=g, num_channels=c)
        else:
            return nn.Identity()
    return partial(norm, norm_type=norm_type)

def get_act(act_type):
    if act_type=='gelu':
        return nn.GELU()
    elif act_type=='relu':
        return nn.ReLU()
    elif act_type[:3]=='elu':
        alpha = float(act_type.replace('elu', ''))
        return nn.ELU(alpha)
    else:
        raise Exception
        
     
    
class DownSample(nn.Module):
    def __init__(self, input_channel, out_channel, scale, stride, padding, activation, normalization):
        super().__init__()
        
        self.conv_layer = nn.Sequential(
            normalization(input_channel),
            activation,
            nn.Conv2d(in_channels=input_channel,
                              out_channels=out_channel,
                              kernel_size=scale, # change to kernel_size = scale (scale,1)
                              stride=stride,  # scale
                              padding=padding,
                              bias=False)
        )

    def forward(self, x):
        return self.conv_layer(x)

class UpSample(nn.Module):
    def __init__(self, input_channel, out_channel, scale, stride, padding, activation, normalization):
        super().__init__()
        
        self.convT_layer = nn.Sequential(
            normalization(input_channel),
            activation,
            nn.ConvTranspose2d(in_channels=input_channel,
                              out_channels=out_channel,
                              kernel_size=scale, # change to kernel_size = scale (scale,1)
                              stride=stride,  # scale
                              padding=padding,
                              bias=False)
        )
        
    def forward(self, x):
        return self.convT_layer(x)

    
class ConvBlock(nn.Module):
    def __init__(self, input_channel, out_channels, activation, normalization):
        super().__init__()
        
        self.conv_layer_1 = nn.Sequential(
            normalization(input_channel),
            activation,
            nn.Conv2d(in_channels=input_channel, 
                        out_channels=out_channels, 
                        kernel_size=1, 
                        stride=1,
                        bias=False)
        )
        
        self.conv_layer_2 = nn.Sequential(
            normalization(input_channel),
            activation,
            nn.Conv2d(in_channels=input_channel, 
                        out_channels=out_channels, 
                        kernel_size=3, 
                        stride=1, padding='same',
                        bias=False)
        )
        
        self.conv_layer_3 = nn.Sequential(
            normalization(input_channel),
            activation,
            nn.Conv2d(in_channels=input_channel, 
                        out_channels=out_channels, 
                        kernel_size=5, 
                        stride=1, padding='same',
                        bias=False)
        )
        
    def forward(self, x):
        x1 = self.conv_layer_1(x)
        x2 = self.conv_layer_2(x)
        x3 = self.conv_layer_3(x)
        return torch.concat((x1, x2, x3), dim = 1)
    
    
class Encoder_layer(nn.Module):
    def __init__(self, input_channel, out_channel, scale, stride, padding, activation, normalization):
        super().__init__()
        
        self.conv_block = ConvBlock(input_channel, out_channel, activation, normalization)
        self.down_sample = DownSample(out_channel*3, out_channel, scale, stride, padding, activation, normalization)
        
    def forward(self, x):
        
        x = self.conv_block(x)
        x = self.down_sample(x)
             
        return x

class Decoder_layer(nn.Module):
    def __init__(self, input_channel, out_channel, scale, stride, padding, activation, normalization):
        super().__init__()
        
        self.conv_block = ConvBlock(input_channel, out_channel, activation, normalization)
        self.up_layer = UpSample(out_channel*3, out_channel, scale, stride, padding, activation, normalization)
        
    
    def forward(self, x, length=None):
        x = self.conv_block(x)
        x = self.up_layer(x)
        return x


def unfold(a, kernel_size, stride):
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
    assert strides[-1] == 1, 'data should be contiguous'
    strides = strides[:-1] + [stride, 1]
    return a.as_strided([*shape, n_frames, kernel_size], strides)
    
    
    
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
        self.lstm = nn.LSTM(bidirectional=True, num_layers=layers, hidden_size=dim, input_size=dim)
        self.linear = nn.Linear(2 * dim, dim)
        self.skip = skip

    def forward(self, x):
        B, C, T = x.shape
        y = x
        framed = False
        if self.max_steps is not None and T > self.max_steps:
            width = self.max_steps
            stride = width // 2
            frames = unfold(x, width, stride)
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
    def __init__(self, input_channel, out_channel, activation, normalization,
                 layers=5, max_steps=200, skip=True, stride=1, padding=1):
        super().__init__()
        
        self.conv_layer = nn.Sequential(
            normalization(input_channel),
            activation,
            nn.Conv1d(input_channel, out_channel, kernel_size=3, stride=stride, padding='same')
        )
        
        self.biLSTM_1 = BLSTM(out_channel, layers=layers, max_steps=max_steps, skip=skip)
        
        self.conv_layer_1 = nn.Sequential(
            normalization(out_channel),
            activation,
            nn.Conv1d(out_channel, input_channel, kernel_size=1, stride=stride, padding='same')
        )
        
        
    def forward(self, x):
        B, C, F, T = x.shape
        skip = x
        
        x = x.view(B, C*F, T)
        x = self.conv_layer(x)
        x = self.biLSTM_1(x)
        x = self.conv_layer_1(x)
        x = x.view(B, C, F, T)
        
        return x + skip
    
class Block_Unet(nn.Module):
    def __init__(self, norm, act,
                 depth=4, 
                 source=['drums', 'bass', 'other', 'vocals'], 
                 channel=32, 
                 stereo=2, 
                 skip_mode='concat',
                 layers_lstm = 2,
                 fq = 2048):
        super().__init__()
        self.sources = source
        
        skip_channel = 2 if skip_mode=='concat' else 1
        self.skip_mode = True if skip_mode=='concat' else False 
        
        self.conv_input = nn.Conv2d(in_channels=stereo, 
                        out_channels=channel, 
                        kernel_size=1, 
                        stride=1,
                        bias=False)
        
        self.conv_final = nn.Conv2d(in_channels=channel+stereo, 
                        out_channels=len(source)*stereo, 
                        kernel_size = 1, 
                        stride=1, 
                        bias=False)
        
        self.encoder_layer = nn.ModuleList()
        
        self.decoder_layer = nn.ModuleList()
        
        for idx in range(depth):
            self.encoder_layer.append( Encoder_layer(input_channel = channel, 
                                                     out_channel = channel*2,
                                                     scale = (4,1),
                                                     stride = (2,1),
                                                     padding = (1,0), 
                                                     activation = act,
                                                     normalization = norm
                                                    ) 
                                     )
            channel *= 2
        self.bottleneck = Bottleneck_v2(input_channel = channel*fq//(2**depth), 
                                        out_channel = 128, 
                                        activation = act,  
                                        normalization = get_norm('InstanceNorm1d'), layers=layers_lstm)
        
        for idx in range(depth):
            self.decoder_layer.append( Decoder_layer(input_channel=channel*skip_channel,
                                                     out_channel=channel//2,
                                                     scale = (4,1),
                                                     stride = (2,1),
                                                     padding = (1,0), 
                                                     activation = act,
                                                     normalization = norm
                                                    ) 
                                     )
            channel //= 2
            
    def normal(self, x):                         # normalization input signal
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean)/ (1e-5 + std)
        return mean, std, x
    
    def forward(self, x):
        B, C, Fq, T = x.shape
        S = len(self.sources)
        mean, std, x = self.normal(x)
        skip = []
        x_mix = x
        x  = self.conv_input(x)
        for encoder in self.encoder_layer:
            x = encoder(x)
            skip.append(x)
        
        x = self.bottleneck(x)
        
        for idx, decoder in enumerate(self.decoder_layer):
            if self.skip_mode:
                x = decoder( torch.concat( (x, skip[-idx-1]), dim=1 ))
            else:
                x = decoder(x + skip[-idx-1])
        
        out = self.conv_final(torch.concat([x, x_mix], dim=1))
        out = out.view(B, S, -1, Fq, T)
        out = out * std[:, None] + mean[:, None]
        
        return out
    
    
from .utils import STFT

class Model_Unet(nn.Module):
    def __init__(self, source=['drums', 'bass', 'other', 'vocals'], 
                 mask_mode = False, nfft = 4096):
        super().__init__()
        
        norm = get_norm('InstanceNorm2d')
        act = get_act('gelu')
        
        self.mask_mode = mask_mode
        self.sources = source
        
        self.stft = STFT(nfft)
        self.magnitude_unet = Block_Unet(norm, act, channel = 36, depth = 4, layers_lstm=8)
        self.phase_unet = Block_Unet(norm, act, channel = 8, depth = 2, layers_lstm=3)
        
    def wave2feature(self, wave):        # wave ->  magnitude & phase
        z = self.stft.stft(wave)
        phase = th.atan2(z.imag, z.real)
        magnitude = z.abs()
        return magnitude, phase
    
        
    def forward(self, x):
        length = x.shape[-1]
        S = len(self.sources)

        x_m, x_p = self.wave2feature(x)
        B, C, Fq, T = x_m.shape
        x_mix = x_m
        x_m = self.magnitude_unet(x_m)
        
        if self.mask_mode:
            mask = nn.functional.softmax(x_m.view(B, S, -1, Fq, T), dim=1)
            return self.stft.istft(x_mix.view(B, 1, C, Fq, T) * mask, length)
        
        x_p = self.phase_unet(x_p)
        imag = x_m * th.sin(x_p)
        real = x_m * th.cos(x_p)
        z = th.complex(real, imag)
        
        
        return self.stft.istft(z, length)