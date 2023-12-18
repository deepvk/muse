import argparse
import gdown
import os
from pathlib import Path

import torch
import torchaudio
from torchaudio.transforms import Fade

from model.PM_Unet import Model_Unet


class InferenceModel:
    def __init__(self, config, model_bottlneck_lstm=True):
        self.config = config
        self.model_bottlneck_lstm = model_bottlneck_lstm
        self.resolve_weigths()

        self.model = Model_Unet(
            source=["drums", "bass", "other", "vocals"],
            depth=4,
            channel=28,
            bottlneck_lstm=model_bottlneck_lstm,
        )

        self.model.load_state_dict(
            torch.load(str(self.weights_path), map_location=torch.device("cpu"))
        )

        self.segment = self.config.segment
        self.overlap = self.config.overlap

    def resolve_weigths(self):
        if self.model_bottlneck_lstm:
            self.weights_path = self.config.weights_dir / "weight_LSTM.pt"
            gdrive_url = self.config.gdrive_weights_LSTM
        else:
            self.weights_path = self.config.weights_dir / "weight_conv.pt"
            gdrive_url = self.config.gdrive_weights_conv

        try:
            self.config.weights_dir.mkdir(parents=True)
            download_weights = True
        except FileExistsError:
            try:
                Path(self.weights_path).touch()
            except FileExistsError:
                download_weights = False

        if download_weights:
            gdown.download(gdrive_url, str(self.weights_path))

    def track(self, sample_mixture_path):
        if sample_mixture_path == self.config.default_input_dir:
            sample_mixture_path = self.resolve_default_sample()

        offset = self.config.offset
        duration = self.config.duration
        waveform, sr = torchaudio.load(sample_mixture_path)

        start = sr * offset if offset else None
        end = sr * (offset + duration) if duration else None
        mixture = waveform[:, start:end]

        ref = mixture.mean(0)
        mixture = (mixture - ref.mean()) / ref.std()  # normalization

        sources = self.separate_sources(mixture[None], sample_rate=sr)

        sources = sources * ref.std() + ref.mean()
        sources_list = ["drums", "bass", "other", "vocals"]
        B, S, C, T = sources.shape
        sources = (
            sources.view(B, S * C, T)
            / sources.view(B, S * C, T).max(dim=2)[0].unsqueeze(-1)
        ).view(B, S, C, T)
        sources = list(sources)

        audios = dict(zip(sources_list, sources[0]))
        audios["original"] = waveform[:, start:end]
        return audios

    def separate_sources(self, mix, sample_rate):
        device = self.config.device
        device = torch.device(device) if device else mix.device

        batch, channels, length = mix.shape

        chunk_len = int(sample_rate * self.segment * (1 + self.overlap))
        start = 0
        end = chunk_len
        overlap_frames = self.overlap * sample_rate
        fade = Fade(
            fade_in_len=0, fade_out_len=int(overlap_frames), fade_shape="linear"
        )

        final = torch.zeros(
            batch,
            len(["drums", "bass", "other", "vocals"]),
            channels,
            length,
            device=device,
        )

        while start < length - overlap_frames:
            chunk = mix[:, :, start:end]
            with torch.no_grad():
                out = self.model.forward(chunk)
            out = fade(out)
            final[:, :, :, start:end] += out
            if start == 0:
                fade.fade_in_len = int(overlap_frames)
                start += int(chunk_len - overlap_frames)
            else:
                start += chunk_len
            end += chunk_len
            if end >= length:
                fade.fade_out_len = 0
        return final

    def resolve_default_sample(self):
        default_input_dir = self.config.default_input_dir
        Path(default_input_dir).mkdir(parents=True, exist_ok=True)

        default_sample_path = f"{default_input_dir}/sample.wav"
        try:
            Path(default_sample_path).touch()
            gdown.download(self.config.gdrive_mix, default_sample_path)
        except FileExistsError:
            pass

        return default_sample_path


def main(args, config):
    inf_model = InferenceModel(config)
    audios = inf_model.track(args.mix_path)

    out_dir = f"{args.out_dir}/{os.path.basename(args.mix_path)}/"
    out_paths = (
        f"{out_dir}drums.wav",
        f"{out_dir}bass.wav",
        f"{out_dir}other.wav",
        f"{out_dir}vocals.wav",
    )

    torchaudio.save(out_paths[0], audios["drums"], config.sample_rate)
    torchaudio.save(out_paths[1], audios["bass"], config.sample_rate)
    torchaudio.save(out_paths[2], audios["other"], config.sample_rate)
    torchaudio.save(out_paths[3], audios["vocals"], config.sample_rate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script")
    from config.config import InferenceConfig

    config = InferenceConfig()

    parser.add_argument(
        "-I",
        dest="mix_path",
        help="path to mixture",
        default=config.default_input_dir,
        type=str,
    )
    parser.add_argument(
        "-O",
        dest="out_dir",
        help="specified output dir",
        default=config.default_result_dir,
        type=str,
    )
    # TODO : argument for weigths

    args = parser.parse_args()
    main(args, config)