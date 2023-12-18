from dataclasses import dataclass
from pathlib import Path


@dataclass
class ConverterConfig:
    weights_dir: Path = Path("/app/streaming/weights")
    gdrive_weights_LSTM_id: str = "18jT2TYffdRD1fL7wecAiM5nJPM_OKpNB"
    gdrive_weights_conv_id: str = "1VO07OYbsnCuEJYRSuA8HhjlQnx6dbWX7"

    original_model_src: str = "/app/separator/model"
    original_model_dst: str = "/app/streaming/model"
    model_py_module: str = "model.PM_Unet"
    model_class_name: str = "Model_Unet"
    tflite_model_dst: str = "tflite_model"


@dataclass
class StreamConfig:
    sample_rate: int = 44100
    nfft: int = 4096
    stft_py_module: str = "model.STFT"
    default_input_path: str = "/app/streaming/input"
    default_result_dir: str = "/app/streaming/streams"
    gdrive_mix_id: str = "1zJpyW1fYxHKXDcDH9s5DiBCYiRpraDB3"
    default_duration: int = 15
