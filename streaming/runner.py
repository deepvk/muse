import argparse
import gdown
import logging
import os
import re
import subprocess
from pathlib import Path

from stream_class import TFLiteTorchStream


def resolve_default_sample(config):
    default_input_dir = config.StreamConfig.default_input_dir
    Path(default_input_dir).mkdir(parents=True, exist_ok=True)

    default_sample_path = f"{default_input_dir}/sample.wav"
    try:
        Path(default_sample_path).touch()
        gdown.download(id=config.StreamConfig.gdrive_mix_id, output=default_sample_path)
    except FileExistsError:
        pass

    return default_sample_path


def main(args, config):
    try:
        Path(config.ConverterConfig.tflite_model_dst).mkdir(exist_ok=False)
        start_converter = True
    except FileExistsError:
        if len(os.listdir(config.ConverterConfig.tflite_model_dst)) == 0:
            start_converter = True
        else:
            start_converter = False

    if start_converter:
        subprocess.Popen(
            ["python3", config.StreamConfig.converter_script],
            executable="/bin/bash", shell=True
        )

    converter_outputs = os.listdir(config.ConverterConfig.tflite_model_dst)
    converter_outputs = list(
        filter(lambda x: re.match(r".*_outer_stft_.*\.tflite$", x), converter_outputs)
    )
    converter_outputs.sort(key=lambda x: os.stat(x).st_mtime, reverse=True)

    tflite_model_path = str(
        config.ConverterConfig.tflite_model_dst + f"\{converter_outputs[0]}"
    )
    parsed_segment = re.findall(r"_outer_stft_(.*)\.tflite$", tflite_model_path)

    track_path = args.mix_path
    if args.mix_path == config.StreamConfig.default_input_dir:
        track_path = resolve_default_sample(config)

    stream_class = TFLiteTorchStream(tflite_model_path, segment=float(parsed_segment))
    out_paths = stream_class(track_path, args.out_dir, args.duration)
    logging.info("Streams stored in : " + " ".join(out_paths))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runner script")
    from config import config

    parser.add_argument(
        "-I",
        dest="mix_path",
        help="path to mixture",
        default=config.StreamConfig.default_input_dir,
        type=str,
    )
    parser.add_argument(
        "-O",
        dest="out_dir",
        help="specified output dir",
        default=config.StreamConfig.default_result_dir,
        type=str,
    )
    parser.add_argument(
        "-d",
        dest="duration",
        help="specified first seconds to process",
        default=config.StreamConfig.default_duration,
        type=int,
    )

    # TODO: specify .tflite model manually
    args = parser.parse_args()
    main(args, config)
