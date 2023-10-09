# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Data augmentations.
"""

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
            signs = th.randint(2, (batch, sources, 1, 1), device=wav.device, dtype=th.float32)
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
        each group separatly. This allow to keep the same probability distribution no matter
        the number of GPUs. Without this grouping, using more GPUs would lead to a higher
        probability of keeping two sources from the same track together which can impact
        performance.
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
                raise ValueError(f"Batch size {batch} must be divisible by group size {group_size}")
            groups = batch // group_size
            wav = wav.view(groups, group_size, streams, channels, time)
            permutations = th.argsort(th.rand(groups, group_size, streams, 1, 1, device=device),
                                      dim=1)
            wav = wav.gather(1, permutations.expand(-1, -1, -1, channels, time))
            wav = wav.view(batch, streams, channels, time)
        return wav


class Scale(nn.Module):
    def __init__(self, proba=1., min=0.25, max=1.25):
        super().__init__()
        self.proba = proba
        self.min = min
        self.max = max

    def forward(self, wav):
        batch, streams, channels, time = wav.size()
        device = wav.device
        if self.training and random.random() < self.proba:
            scales = th.empty(batch, streams, 1, 1, device=device).uniform_(self.min, self.max)
            wav *= scales
        return wav

    
### 

class FadeMask(nn.Module):

    def __init__(self, proba=1, sample_rate = 44100,
                 time_mask_param = 2):
        super().__init__()
        self.sample_rate = sample_rate
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=sample_rate*time_mask_param)
        self.proba = proba
        '''
        time_mask_param - maximum possible length of the mask
        '''
    def forward(self, wav):
        
        if random.random() < self.proba:
            wav = wav.clone()
            wav[:,0] = self.time_mask(wav[:,0])
            wav[:,1] = self.time_mask(wav[:,1])
            wav[:,2] = self.time_mask(wav[:,2])
            wav[:,3] = self.time_mask(wav[:,3])
            
        return wav   # output -> tensor
    
    
class PitchShift_f(nn.Module):  # input -> tensor     fast (костыли)
    '''
    https://github.com/asteroid-team/torch-audiomentations/blob/main/torch_audiomentations/augmentations/pitch_shift.py
    Pitch shift the sound up or down without changing the tempo
    '''
    def __init__(self, proba=1, min_semitones=-3, max_semitones=3, 
                 min_semitones_other=-2, max_semitones_other=2, sample_rate=44100):

        super().__init__()
#         self.pitch = PitchShift(p=proba, 
#                                 min_transpose_semitones=min_semitones,
#                                 max_transpose_semitones=max_semitones,
#                                 sample_rate = sample_rate)
        
        self.pitch_vocals = PitchShift(p=proba, 
                                min_transpose_semitones=min_semitones,
                                max_transpose_semitones=max_semitones,
                                sample_rate=sample_rate)
        
        self.pitch_other = PitchShift(p=proba, 
                                min_transpose_semitones=min_semitones_other,
                                max_transpose_semitones=max_semitones_other,
                                sample_rate=sample_rate)

        '''
        Minimum pitch shift transposition in semitones
        Maximum pitch shift transposition in semitones
        '''
        
    def forward(self, wav):
        wav = wav.clone()
        
        wav[:,0] = self.pitch_other(wav[:,0])
        wav[:,1] = self.pitch_other(wav[:,1])
        wav[:,2] = self.pitch_other(wav[:,2])
        wav[:,3] = self.pitch_vocals(wav[:,3])
        
        return wav   # output -> tensor
    
    
class TimeChange_f(nn.Module):  # input -> np.array      fast

    def __init__(self, proba=1, min_rate=0.8, max_rate = 1.3, sample_rate = 44100):
        super().__init__()
        self.sample_rate = sample_rate
        self.proba = proba
   
        factors_list = [0.8, 0.85, 0.9, 0.95, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3]
        self.time = torchaudio.transforms.SpeedPerturbation(orig_freq = sample_rate,
                                                            factors = factors_list)
        
    def forward(self, wav):

        if random.random() < self.proba:
            wav, _ = self.time(wav) # output - tensor
            
        return wav   # output -> tensor

#new augment

class Double(nn.Module):
    def __init__(self, proba=1):
        super().__init__()
        self.proba = proba
    
    def forward(self, wav):
        num_samples = wav.shape[-1]

        if random.random() < self.proba:
            wav = wav.clone()
            
            if random.random() < 0.5:
                wav[:,0][:,1] =  wav[:,0][:,0]
                wav[:,1][:,1] =  wav[:,1][:,0]
                wav[:,2][:,1] =  wav[:,2][:,0]
                wav[:,3][:,1] =  wav[:,3][:,0]
            else:
                wav[:,0][:,0] =  wav[:,0][:,1]
                wav[:,1][:,0] =  wav[:,1][:,1]
                wav[:,2][:,0] =  wav[:,2][:,1]
                wav[:,3][:,0] =  wav[:,3][:,1]
            
            
        return wav
    
class Reverse(nn.Module):
    def __init__(self, proba=1, min_band_part=0.2, max_band_part=0.4):
        super().__init__()
        self.proba = proba
        self.min_band_part = min_band_part
        self.max_band_part = max_band_part
    
    def forward(self, wav):
        num_samples = wav.shape[-1]

        if random.random() < self.proba:
            wav = wav.clone()
        
            end = random.randint(int(num_samples * self.min_band_part), int(num_samples * self.max_band_part))
            start = random.randint(0, num_samples - end)
            wav[..., start : end + start][:,3] = th.flip(wav[..., start : end + start][:,3], [2])
        
        return wav
