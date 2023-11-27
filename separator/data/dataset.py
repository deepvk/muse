from collections import OrderedDict
import hashlib
import math
import json
import os
from pathlib import Path
import tqdm
import logging

import musdb
import julius
import torch as th
from torch import distributed
import torchaudio as ta
from torch.nn import functional as F


class File:
    MIXTURE = "mixture"
    EXT = ".wav"


def get_musdb_wav_datasets(
    musdb="musdb18hq",
    musdb_samplerate=44100,
    use_musdb=True,
    segment=11,
    shift=1,
    train_valid=False,
    full_cv=True,
    samplerate=44100,
    channels=2,
    normalize=True,
    metadata="./metadata",
    sources=["drums", "bass", "other", "vocals"],
    backend=None,
    data_type="train",
):
    """
    Extract the musdb dataset from the XP arguments.
    """
    sig = hashlib.sha1(str(musdb).encode()).hexdigest()[:8]
    metadata_file = Path(metadata) / ("musdb_" + sig + ".json")
    root = Path(musdb) / data_type
    if not metadata_file.is_file():
        metadata_file.parent.mkdir(exist_ok=True, parents=True)
        metadata = MetaData.build_metadata(root, sources)
        json.dump(metadata, open(metadata_file, "w"))
    metadata = json.load(open(metadata_file))

    valid_tracks = _get_musdb_valid()
    if train_valid:
        metadata_train = metadata
    else:
        metadata_train = {
            name: meta for name, meta in metadata.items() if name not in valid_tracks
        }

    data_set = Wavset(
        root,
        metadata_train,
        sources,
        segment=segment,
        shift=shift,
        samplerate=samplerate,
        channels=channels,
        normalize=normalize,
    )

    return data_set


def _get_musdb_valid():
    # Return musdb valid set.
    import yaml

    setup_path = Path(musdb.__path__[0]) / "configs" / "mus.yaml"
    setup = yaml.safe_load(open(setup_path, "r"))
    return setup["validation_tracks"]


class Wavset:
    def __init__(
        self,
        root,
        metadata,
        sources,
        segment=None,
        shift=None,
        normalize=True,
        samplerate=44100,
        channels=2,
        ext=File.EXT,
    ):
        """
        Waveset (or mp3 set for that matter).
        Can be used to train with arbitrary sources.
        Each track should be one folder inside of `path`.
        The folder should contain files named `{source}.{ext}`.

        Args:
            root (Path or str): root folder for the dataset.
            metadata (dict): output from `build_metadata`.
            sources (list[str]): list of source names.
            segment (None or float): segment length in seconds.
                If `None`, returns entire tracks.
            shift (None or float): stride in seconds bewteen samples.
            normalize (bool): normalizes input audio,
                **based on the metadata content**,
                i.e. the entire track is normalized, not individual extracts.
            samplerate (int): target sample rate. if the file sample rate
                is different, it will be resampled on the fly.
            channels (int): target nb of channels. if different, will be
                changed onthe fly.
            ext (str): extension for audio files (default is .wav).

        samplerate and channels are converted on the fly.
        """
        self.root = Path(root)
        self.metadata = OrderedDict(metadata)
        self.segment = segment
        self.shift = shift or segment
        self.normalize = normalize
        self.sources = sources
        self.channels = channels
        self.samplerate = samplerate
        self.ext = ext
        self.num_examples = []
        for name, meta in self.metadata.items():
            track_duration = meta["length"] / meta["samplerate"]
            if segment is None or track_duration < segment:
                examples = 1
            else:
                examples = int(
                    math.ceil((track_duration - self.segment) / self.shift) + 1
                )
            self.num_examples.append(examples)

    def __len__(self):
        return sum(self.num_examples)

    def get_file(self, name, source):
        return self.root / name / f"{source}{self.ext}"

    def __getitem__(self, index):
        for name, examples in zip(self.metadata, self.num_examples):
            if index >= examples:
                index -= examples
                continue
            meta = self.metadata[name]
            num_frames = -1
            offset = 0
            if self.segment is not None:
                offset = int(meta["samplerate"] * self.shift * index)
                num_frames = int(math.ceil(meta["samplerate"] * self.segment))
            wavs = []
            for source in self.sources:
                file = self.get_file(name, source)
                wav, _ = ta.load(str(file), frame_offset=offset, num_frames=num_frames)
                wav = self.__convert_audio_channels(wav, self.channels)
                wavs.append(wav)

            example = th.stack(wavs)
            example = julius.resample_frac(example, meta["samplerate"], self.samplerate)
            if self.normalize:
                example = (example - meta["mean"]) / meta["std"]
            if self.segment:
                length = int(self.segment * self.samplerate)
                example = example[..., :length]
                example = F.pad(example, (0, length - example.shape[-1]))
            return example

    def __convert_audio_channels(self, wav, channels=2):
        """Convert audio to the given number of channels."""
        *shape, src_channels, length = wav.shape
        if src_channels == channels:
            pass
        elif channels == 1:
            # Case 1:
            # The caller asked 1-channel audio, but the stream have multiple
            # channels, downmix all channels.
            wav = wav.mean(dim=-2, keepdim=True)
        elif src_channels == 1:
            # Case 2:
            # The caller asked for multiple channels, but the input file have
            # one single channel, replicate the audio over all channels.
            wav = wav.expand(*shape, channels, length)
        elif src_channels >= channels:
            # Case 3:
            # The caller asked for multiple channels, and the input file have
            # more channels than requested.
            # In that case return the first channels.
            wav = wav[..., :channels, :]
        else:
            # Case 4: What is a reasonable choice here?
            raise ValueError(
                "The audio file has less channels than requested \
                    but is not mono."
            )
        return wav


class MetaData:
    def __track_metadata(track, sources, normalize=True, ext=File.EXT):
        track_length = None
        track_samplerate = None
        mean = 0
        std = 1
        for source in sources + [File.MIXTURE]:
            source_file = track / f"{source}{ext}"
            if source == File.MIXTURE and not source_file.exists():
                audio = 0
                for sub_source in sources:
                    sub_file = track / f"{sub_source}{ext}"
                    sub_audio, sr = ta.load(sub_file)
                    audio += sub_audio
                would_clip = audio.abs().max() >= 1
                if would_clip:
                    assert (
                        ta.get_audio_backend() == "soundfile"
                    ), "use dset.backend=soundfile"
                ta.save(source_file, audio, sr, encoding="PCM_F")

            try:
                info = ta.info(str(source_file))
            except RuntimeError:
                logging.error(f"{source_file} is invalid")
                raise
            length = info.num_frames
            if track_length is None:
                track_length = length
                track_samplerate = info.sample_rate
            elif track_length != length:
                raise ValueError(
                    f"Invalid length for file {source_file}: "
                    f"expecting {track_length} but got {length}."
                )
            elif info.sample_rate != track_samplerate:
                raise ValueError(
                    f"Invalid sample rate for file {source_file}: "
                    f"expecting {track_samplerate} but got \
                        {info.sample_rate}."
                )
            if source == File.MIXTURE and normalize:
                try:
                    wav, _ = ta.load(str(source_file))
                except RuntimeError:
                    logging.error(f"{source_file} is invalid")
                    raise
                wav = wav.mean(0)
                mean = wav.mean().item()
                std = wav.std().item()

        return {
            "length": length,
            "mean": mean,
            "std": std,
            "samplerate": track_samplerate,
        }

    def build_metadata(path, sources, normalize=True, ext=File.EXT):
        """
        Build the metadata for `Wavset`.

        Args:
            path (str or Path): path to dataset.
            sources (list[str]): list of sources to look for.
            normalize (bool): if True, loads full track and store normalization
                values based on the mixture file.
            ext (str): extension of audio files (default is .wav).
        """

        meta = {}
        path = Path(path)
        pendings = []
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(8) as pool:
            for root, folders, files in os.walk(path, followlinks=True):
                root = Path(root)
                if root.name.startswith(".") or folders or root == path:
                    continue
                name = str(root.relative_to(path))
                pendings.append(
                    (
                        name,
                        pool.submit(
                            MetaData.__track_metadata, root, sources, normalize, ext
                        ),
                    )
                )

            for name, pending in tqdm.tqdm(pendings, ncols=120):
                meta[name] = pending.result()
        return meta
