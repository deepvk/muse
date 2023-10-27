import torch
import torch as th
import torch.nn as nn
from .utils import STFT
from functools import partial
from .modules import Encoder, Decoder, Bottleneck_v2, Bottleneck

def get_norm(norm_type):
    def norm(c, norm_type):   
        if norm_type=='BatchNorm':
            return nn.BatchNorm2d(c)
        elif norm_type=='InstanceNorm2d':
            return nn.InstanceNorm2d(c)
        elif norm_type=='InstanceNorm1d':
            return nn.InstanceNorm1d(c)
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


class Model_Unet(nn.Module):
    def __init__(self, 
                 depth=4, 
                 source=['drums', 'bass', 'other', 'vocals'], 
                 channel=28, 
                 is_mono=False,
                 mask_mode = False,
                 skip_mode='concat',
                 nfft = 4096,
                 bottlneck_lstm = True,
                 layers=2,
                ):
        super().__init__()
        
        self.sources = source
        skip_channel = 2 if skip_mode=='concat' else 1
        self.skip_mode = True if skip_mode=='concat' else False
        stereo = 1 if is_mono else 2
        self.mask_mode = mask_mode
        
        norm = get_norm('InstanceNorm2d')
        act = get_act('gelu')
         
        self.stft = STFT(nfft)
        self.conv_magnitude = nn.Conv2d(in_channels=stereo, 
                        out_channels=channel, 
                        kernel_size=1, 
                        stride=1,
                        bias=False)
        
        self.conv_magnitude_final = nn.Conv2d(in_channels=channel, 
                        out_channels=len(source)*stereo, 
                        kernel_size= 1, 
                        stride=1, 
                        bias=False)
        
        self.conv_phase = nn.Conv2d(in_channels=stereo, 
                        out_channels=channel, 
                        kernel_size=1, 
                        stride=1,
                        bias=False)
        
        self.conv_phase_final = nn.Conv2d(in_channels=channel, 
                        out_channels=len(source)*stereo, 
                        kernel_size=1, 
                        stride=1, 
                        bias=False)
        
        self.encoder_magnitude = nn.ModuleList()
        self.decoder_magnitude = nn.ModuleList()
        
        self.encoder_phase = nn.ModuleList()
        self.decoder_phase = nn.ModuleList()
        
        for idx in range(depth):
            self.encoder_magnitude.append( Encoder(input_channel = channel, 
                                                   out_channel = channel*2, 
                                                   scale = (4,1),
                                                   stride = (4,1),
                                                   padding = 0,
                                                   normalization=norm, 
                                                   activation=act) )
            
            self.encoder_phase.append(Encoder(input_channel=channel, 
                                              out_channel=channel*2, 
                                              scale=(4,1),
                                              stride=(4,1),
                                              padding=0,
                                              normalization=norm, 
                                              activation=act))
            channel *= 2
            
        if bottlneck_lstm:
            self.bottleneck_magnitude = Bottleneck_v2(input_channel=channel*(nfft//2)//(2**(2*depth)), 
                                                      out_channel=channel, 
                                                      normalization=nn.InstanceNorm1d, 
                                                      activation=act,
                                                      layers=layers)
        else:
            self.bottleneck_magnitude = Bottleneck(input_channel=channel, 
                                                   out_channels=channel, 
                                                   normalization=norm, 
                                                   activation=act)
            
        self.bottleneck_phase = Bottleneck(channel, channel, norm, act)
        
        for idx in range(depth):
            self.decoder_magnitude.append( Decoder(input_channel=channel*skip_channel,
                                                   out_channel=channel//2,
                                                   scale=(4,1),
                                                   stride=(4,1),
                                                   padding=0,
                                                   normalization=norm, 
                                                   activation=act) )
            self.decoder_phase.append( Decoder(input_channel=channel*skip_channel,
                                               out_channel=channel//2,
                                               scale=(4,1),
                                               stride=(4,1),
                                               padding=0,
                                               normalization=norm, 
                                               activation=act) )
            channel //= 2
    
    def wave2feature(self, wave):        # wave ->  magnitude & phase
        z = self.stft.stft(wave)
        phase = th.atan2(z.imag, z.real)
        magnitude = z.abs()
        return magnitude, phase
    
    def normal(self, x):                         # normalization input signal
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean)/ (1e-5 + std)
        return mean, std, x
    
    def forward(self, x):
        length_wave = x.shape[-1]
        
        x_m, x_p = self.wave2feature(x)
        
        B, C, Fq, T = x_m.shape
        S = len(self.sources)
        
        # normalization magnitude input
        mean_m, std_m, x_m = self.normal(x_m)
        x_mix = x_m
        # normalization magnitude phase input
        mean_p, std_p, x_p = self.normal(x_p)
        
        skip_m = [] # skip connection magnitude branch
        skip_p = [] # skip connection phase branch
        
        x_m = self.conv_magnitude(x_m)     # start conv magnitude
        x_p = self.conv_phase(x_p)     # start conv phase
        
        for idx_enc in range(len(self.encoder_magnitude)):
            x_m = self.encoder_magnitude[idx_enc](x_m) # encoder layer magnitude
            x_p = self.encoder_phase[idx_enc](x_p) # encoder layer phase
            
            skip_m.append(x_m) # skip magnitude
            skip_p.append(x_p) # skip phase
        
        x_m = self.bottleneck_magnitude(x_m)
        x_p = self.bottleneck_phase(x_p)
        
        for idx in range(len(self.decoder_magnitude)):
            if self.skip_mode:
                x_m = self.decoder_magnitude[idx](torch.concat( (x_m, skip_m[-idx-1]), dim=1 )) # decoder layer magnitude
                x_p = self.decoder_phase[idx](torch.concat( (x_p, skip_p[-idx-1]), dim=1 ))# decoder layer phase
            else:
                x_m = self.decoder_magnitude[idx](x_m + skip_m[-idx-1])
                x_p = self.decoder_phase[idx](x_p + skip_p[-idx-1])
            
        x_m = self.conv_magnitude_final(x_m)     # final conv magnitude
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
        
        return self.stft.istft(z, length_wave)