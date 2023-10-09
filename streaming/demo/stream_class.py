import tensorflow as tf
from typing import Tuple, Union

import torch
from torch.nn import functional as F
import torchaudio
from torchaudio.io import StreamReader, StreamWriter
from tqdm import notebook

import math
import os


class TFLiteTorchStream:
    NUM_CHANNELS = 2

    def __init__(
        self,
        model_filename: str,
        segment: int = 0.5,
        sample_rate: int = 44100,
        # STFT
        nfft: int = 4096,
        ):
        self.__interpreter = tf.lite.Interpreter(
            model_path=f"model/{model_filename}.tflite"
        )
        self.__interpreter.allocate_tensors()
        self.__input_details = self.__interpreter.get_input_details()
        self.__output_details = self.__interpreter.get_output_details()

        self.nfft = nfft
        self.hop_length = self.nfft // 4
        self.sample_rate = sample_rate
        self.segment = segment

    def __call__(
        self,
        track_path: str,
        out_dir: str,
        duration: Union[int, None] = None,
        overlap: int = 0
        ):
        waveform, sample_rate = torchaudio.load(track_path)
        if sample_rate != self.sample_rate:
            raise ValueError(f"Non supported sample_rate of {track_path=}")

        stream_mix = StreamReader(src=track_path)
        frames_per_chunk = int(44100 * self.segment)
        stream_mix.add_basic_audio_stream(
            frames_per_chunk=frames_per_chunk,
            sample_rate=44100
        )

        try:
            os.mkdir(out_dir)
        except OSError as error:
            pass

        out_paths = (
            f"{out_dir}/drums.wav",
            f"{out_dir}/bass.wav",
            f"{out_dir}/other.wav",
            f"{out_dir}/vocals.wav",
        )

        stream_drums = StreamWriter(dst=out_paths[0])
        stream_bass = StreamWriter(dst=out_paths[1])
        stream_other = StreamWriter(dst=out_paths[2])
        stream_vocals = StreamWriter(dst=out_paths[3])

        stream_drums.add_audio_stream(
            sample_rate,
            TFLiteTorchStream.NUM_CHANNELS
        )
        stream_bass.add_audio_stream(
            sample_rate,
            TFLiteTorchStream.NUM_CHANNELS
        )
        stream_other.add_audio_stream(
            sample_rate,
            TFLiteTorchStream.NUM_CHANNELS
        )
        stream_vocals.add_audio_stream(
            sample_rate,
            TFLiteTorchStream.NUM_CHANNELS
        )

        chunk_count = int(sample_rate * duration // frames_per_chunk) if duration else 0
        with (
            stream_drums.open(),
            stream_bass.open(),
            stream_other.open(),
            stream_vocals.open()
            ):
            for i, chunk in notebook.tqdm(enumerate(stream_mix.stream())):
                if duration and i > chunk_count:
                    break
                processed_chunk = (chunk[0].T)[None]
                if processed_chunk.shape[-1] != int(44100 * self.segment):
                    continue
                with torch.no_grad():
                    out = self.model_call(processed_chunk).permute(0,1,3,2)

                stream_drums.write_audio_chunk(0, out[0][0])
                stream_bass.write_audio_chunk(0, out[0][1])
                stream_other.write_audio_chunk(0, out[0][2])
                stream_vocals.write_audio_chunk(0, out[0][3])

        return out_paths

    def model_call(self, mix):
        length = mix.shape[-1]
        in_spectr = self.spectr(mix)

        self.__interpreter.set_tensor(
            self.__input_details[0]['index'],
            in_spectr
        )
        self.__interpreter.invoke()

        out_spectr_tf = self.__interpreter.get_tensor(
            self.__output_details[0]['index']
        )
        out_spectr = torch.tensor(out_spectr_tf, dtype=torch.cfloat)

        ret = self.ispectr(out_spectr, length)
        return ret

    def spectr(self, x):
        le = int(math.ceil(x.shape[-1] / self.hop_length))
        pad = self.hop_length // 2 * 3
        x = self.pad1d(
            x,
            (pad, pad + le * self.hop_length - x.shape[-1]),
            mode="reflect"
        )

        *other, length = x.shape
        x = x.reshape(-1, length)
        z = torch.stft(
            x,
            self.nfft,
            self.hop_length,
            window=torch.hann_window(self.nfft).to(x),
            win_length=self.nfft,
            normalized=True,
            center=True,
            return_complex=True,
            pad_mode='reflect'
        )
        _, freqs, frame = z.shape
        z = z.view(*other, freqs, frame)
        z = z[..., :-1, :]
        z = z[..., 2: 2 + le]
        return z

    def ispectr(
        self,
        z,
        length=None,
        scale=0
    ):
        hl = self.hop_length // (4**scale)
        z = F.pad(z, (0, 0, 0, 1))
        z = F.pad(z, (2, 2))
        pad = hl // 2 * 3
        le = hl * int(math.ceil(length / hl)) + 2 * pad

        *other, freqs, frames = z.shape
        n_fft = 2 * freqs - 2
        z = z.view(-1, freqs, frames)
        win_length = n_fft
        x = torch.istft(
            z,
            n_fft,
            hl,
            window=torch.hann_window(win_length).to(z.real),
            win_length=win_length,
            normalized=True,
            length=le,
            center=True
        )
        _, length = x.shape
        x = x.view(*other, le)
        x = x[..., pad: pad + length]
        return x

    @staticmethod
    def pad1d(
        x: torch.Tensor,
        paddings: Tuple[int, int],
        mode: str = 'constant',
        value: float = 0.
    ):
        """
        Tiny wrapper around F.pad, just to allow for reflect padding on small input.
        If this is the case, we insert extra 0 padding to the right before the reflection happen.
        """
        x0 = x
        length = x.shape[-1]
        padding_left, padding_right = paddings
        if mode == 'reflect':
            max_pad = max(padding_left, padding_right)
            if length <= max_pad:
                extra_pad = max_pad - length + 1
                extra_pad_right = min(padding_right, extra_pad)
                extra_pad_left = extra_pad - extra_pad_right
                paddings = (padding_left - extra_pad_left, padding_right - extra_pad_right)
                x = F.pad(x, (extra_pad_left, extra_pad_right))
        out = F.pad(x, paddings, mode, value)
        assert out.shape[-1] == length + padding_left + padding_right
        assert (out[..., padding_left: padding_left + length] == x0).all()
        return out
