import tensorflow as tf
import torch
import torchaudio
from torchaudio.io import StreamReader, StreamWriter

import os

from tqdm import tqdm
from typing import Union

from model.STFT import STFT


class TFLiteTorchStream:
    NUM_CHANNELS = 2

    def __init__(
        self,
        model_filename: str,
        segment: int = 1,
        sample_rate: int = 44100,
        # STFT
        nfft: int = 4096,
        ):
        self.__interpreter = tf.lite.Interpreter(
            model_path=f"tfile_model/{model_filename}.tflite"
        )
        self.__interpreter.allocate_tensors()
        self.__input_details = self.__interpreter.get_input_details()
        self.__output_details = self.__interpreter.get_output_details()

        self.nfft = nfft
        self.hop_length = self.nfft // 4
        self.sample_rate = sample_rate
        self.segment = segment
        self.stft = STFT(nfft)

    def __call__(
        self,
        track_path: str,
        out_dir: str,
        duration: Union[int, None] = None
        ):
        _, sample_rate = torchaudio.load(track_path)
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
            for i, chunk in tqdm(enumerate(stream_mix.stream())):
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
        in_spectr = self.stft.stft(mix)

        self.__interpreter.set_tensor(
            self.__input_details[0]['index'],
            in_spectr
        )
        self.__interpreter.invoke()

        out_spectr_tf = self.__interpreter.get_tensor(
            self.__output_details[0]['index']
        )
        out_spectr = torch.tensor(out_spectr_tf, dtype=torch.cfloat)

        ret = self.stft.istft(out_spectr, length)
        return ret
