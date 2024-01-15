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
            self.weights_path = (
                self.config.weights_dir / self.config.weights_LSTM_filename
            )
            gdrive_url = self.config.gdrive_weights_LSTM
        else:
            self.weights_path = (
                self.config.weights_dir / self.config.weights_conv_filename
            )
            gdrive_url = self.config.gdrive_weights_conv

        download_weights = True
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

    def track(self, sample_mixture_path, output_dir):
        if sample_mixture_path == self.config.default_input_dir:
            sample_mixture_path = self.resolve_default_sample()
        output_dir = f"{output_dir}/{sample_mixture_path.split('/')[-1]}"

        offset = self.config.offset
        duration = self.config.duration
        waveform, sr = torchaudio.load(sample_mixture_path)

        start = sr * offset if offset else None
        end = sr * (offset + duration) if duration else None
        mixture = waveform[:, start:end]

        # Normalize
        ref = mixture.mean(0)
        mixture = (mixture - ref.mean()) / ref.std()

        # Do separation
        sources = self.separate_sources(mixture[None], sample_rate=sr)

        # Denormalize
        sources = sources * ref.std() + ref.mean()
        
        sources_list = ["drums", "bass", "other", "vocals"]
        sources_ouputs = {s: f"{output_dir}/{s}.wav" for s in sources_list}

        B, S, C, T = sources.shape
        sources = (
            sources.view(B, S * C, T)
            / sources.view(B, S * C, T).max(dim=2)[0].unsqueeze(-1)
        ).view(B, S, C, T)
        sources = list(sources)

        audios = dict(zip(sources_list, sources[0]))
        for k, v in audios.items():
            audios[k] = {"source": v, "path": sources_ouputs[k]}

        return audios

    def separate_sources(self, mix, sample_rate):
        """
        Separates the audio mix into its constituent sources.

        Args:
            mix (Tensor): The input mixed audio signal tensor of shape (batch, channels, length).
            sample_rate (int): The sample rate of the audio signal.

        Returns:
            Tensor: The separated audio sources as a tensor.
        """
        # Set the device based on the configuration or input mix
        device = torch.device(self.config.device) if self.config.device else mix.device

        # Get the shape of the input mix
        batch, channels, length = mix.shape

        # Calculate chunk length for processing and overlap frames
        chunk_len = int(sample_rate * self.segment * (1 + self.overlap))
        overlap_frames = int(self.overlap * sample_rate)
        fade = Fade(fade_in_len=0, fade_out_len=overlap_frames, fade_shape="linear")

        # Initialize the tensor to hold the final separated sources
        num_sources = 4  # ["drums", "bass", "other", "vocals"]
        final = torch.zeros(batch, num_sources, channels, length, device=device)

        start, end = 0, chunk_len
        while start < length - overlap_frames:
            # Process each chunk with model and apply fade
            chunk = mix[:, :, start:end]
            with torch.no_grad():
                separated_sources = self.model.forward(chunk)
            separated_sources = fade(separated_sources)
            final[:, :, :, start:end] += separated_sources

            # Adjust the start and end for the next chunk, and update fade parameters
            start, end = self.__update_chunk_indices(
                start, end, chunk_len, overlap_frames, length, fade
            )

        return final

    @staticmethod
    def __update_chunk_indices(start, end, chunk_len, overlap_frames, length, fade):
        """
        Update the chunk indices for the next iteration and adjust fade parameters.

        Args:
            start (int): Current start index of the chunk.
            end (int): Current end index of the chunk.
            chunk_len (int): Length of each chunk.
            overlap_frames (int): Number of overlapping frames.
            length (int): Total length of the audio signal.
            fade (Fade): The Fade object used for applying fade in/out.

        Returns:
            Tuple[int, int]: The updated start and end indices for the next chunk.
        """
        if start == 0:
            fade.fade_in_len = overlap_frames
            start += chunk_len - overlap_frames
        else:
            start += chunk_len

        end = min(end + chunk_len, length)
        fade.fade_out_len = 0 if end >= length else overlap_frames

        return start, end

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
    audios = inf_model.track(args.mix_path, args.out_dir)

    torchaudio.save(audios["drums"]["path"], audios["drums"]["source"], config.sample_rate)
    torchaudio.save(audios["bass"]["path"], audios["bass"]["source"], config.sample_rate)
    torchaudio.save(audios["other"]["path"], audios["other"]["source"], config.sample_rate)
    torchaudio.save(audios["vocals"]["path"], audios["vocals"]["source"], config.sample_rate)


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
