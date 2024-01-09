import argparse
import gdown
import logging
import os
import re
import subprocess as sb
import sys
from pathlib import Path

from tf_lite_stream import TFLiteTorchStream


LOGGER = logging.getLogger(__name__)


def resolve_default_sample(config):
    default_input_dir = config.StreamConfig.default_input_dir
    Path(default_input_dir).mkdir(parents=True, exist_ok=True)

    default_sample_path = f"{default_input_dir}/sample.wav"
    try:
        Path(default_sample_path).touch(exist_ok=False)
        gdown.download(id=config.StreamConfig.gdrive_mix_id, output=default_sample_path)
    except FileExistsError:
        pass

    return default_sample_path


def resolve_tflite_model(config):
    try:
        Path(config.ConverterConfig.tflite_model_dst).mkdir(exist_ok=False)
        start_converter = True
    except (OSError, FileExistsError):
        if len(os.listdir(config.ConverterConfig.tflite_model_dst)) == 0:
            start_converter = True
        else:
            start_converter = False

    if start_converter:
        with sb.Popen(
            ["python3", config.StreamConfig.converter_script],
            stdout=sb.PIPE,
            stderr=sb.STDOUT,
        ) as proc:
            LOGGER.info(proc.stdout.read().decode())
        res = proc.wait()
        LOGGER.info(
            f"{config.StreamConfig.converter_script} finished with code : {res}"
        )

    converter_outputs = os.listdir(config.ConverterConfig.tflite_model_dst)
    converter_outputs = list(
        filter(lambda x: re.match(r".*_outer_stft_.*\.tflite$", x), converter_outputs)
    )
    converter_outputs = [
        f"{config.ConverterConfig.tflite_model_dst}/{filename}"
        for filename in converter_outputs
    ]
    converter_outputs.sort(key=lambda x: os.stat(x).st_mtime, reverse=True)
    tflite_model_path = converter_outputs[0]
    parsed_segment = re.findall(r"_outer_stft_(.*)\.tflite$", tflite_model_path)[0]

    return tflite_model_path, parsed_segment


def main(args, config):
    is_tflite_model_path_default = (
        args.tflite_model_path == config.ConverterConfig.tflite_model_dst
    )
    if not is_tflite_model_path_default and not args.tflite_model_segment:
        raise ValueError(
            "Specify segment [-s (0.5, 1, ...)] of STFT to outer tflite model"
        )

    if is_tflite_model_path_default:
        tflite_model_path, parsed_segment = resolve_tflite_model(config)
    else:
        tflite_model_path, parsed_segment = (
            args.tflite_model_path,
            args.tflite_model_segment,
        )

    track_path = args.mix_path
    if args.mix_path == config.StreamConfig.default_input_dir:
        track_path = resolve_default_sample(config)

    stream_class = TFLiteTorchStream(
        config, tflite_model_path, segment=float(parsed_segment)
    )
    out_paths = stream_class(track_path, args.out_dir, args.duration)
    LOGGER.info("Streams stored in : " + " ".join(out_paths))


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
    parser.add_argument(
        "-m",
        dest="tflite_model_path",
        help="path to tflite model",
        default=config.ConverterConfig.tflite_model_dst,
        type=str,
    )
    parser.add_argument(
        "-s",
        dest="tflite_model_segment",
        help="tflite model STFT window width (sample_rate * segment)",
        required=False,
        type=float,
    )

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(
        logging.Formatter(
            "[%(levelname)s](%(filename)s).%(funcName)s(%(lineno)d) - %(message)s"
        )
    )
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[stdout_handler],
        format="%(levelname)s : %(message)s",
    )

    args = parser.parse_args()
    main(args, config)
