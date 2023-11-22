from dataclasses import dataclass
from pathlib import Path
from typing import Union


@dataclass
class TrainConfig:
    device: str = "cuda"

    # datasets
    musdb_path: str = "/app/separator/train/musdb18hq"  # absolute to separator/
    metadata_train_path: str = "/app/separator/train/metadata/train"
    metadata_test_path: str = "/app/separator/train/metadata/test"
    segment: int = 7

    # dataloaders
    batch_size: int = 6
    shuffle_train: bool = True
    shuffle_valid: bool = False
    drop_last: bool = True
    num_workers: int = 2

    # checkpoint_callback
    metric_monitor_mode: str = "min"
    save_top_k_model_weights: int = 1

    # PM_Unet model
    model_source: list = ["drums", "bass", "other", "vocals"]
    model_depth: int = 4
    model_channel: int = 28
    is_mono: bool = False
    mask_mode: bool = False
    skip_mode: str = "concat"
    nfft: int = 4096
    bottlneck_lstm: bool = True
    layers: int = 2
    # augments
    shift: int = 8192
    pitchshift_proba: float = 0.2
    vocals_min_semitones: int = -10
    vocals_max_semitones: int = 10
    other_min_semitones: int = -2
    other_max_semitones: int = 2
    pitchshift_flag_other: bool = False
    time_change_proba: float = 0.2
    time_change_factors: list = [0.8, 0.85, 0.9, 0.95, 1.05, 1.1, 1.15, 1.2]
    remix_proba: float = 1
    remix_group_size: int = batch_size
    scale_proba: float = 1
    scale_min: float = 0.25
    scale_max: float = 1.25
    fade_mask_proba: float = 0.1
    double_proba: float = 0.1
    reverse_proba: float = 0.2

    # lr
    lr: float = 0.5 * 3e-4
    T_0: int = 40

    # lightning
    max_epochs: int = 1000
    precision: str = "bf16-mixed"


@dataclass
class EvalConfig:
    GDRIVE_PREFIX = "https://drive.google.com/uc?id="

    device: str = "cpu"

    # weights
    weights_dir: Path = Path("/app/separator/eval/weights")
    gdrive_weights_LSTM: str = f"{GDRIVE_PREFIX}18jT2TYffdRD1fL7wecAiM5nJPM_OKpNB"
    gdrive_weights_conv: str = f"{GDRIVE_PREFIX}1VO07OYbsnCuEJYRSuA8HhjlQnx6dbWX7"

    # eval instance
    segment: int = 7
    overlap: float = 0.2
    offset: Union[int, None] = None
    duration: Union[int, None] = None

    # eval
    sample_rate: int = 44100
    num_channels: int = 2
    default_result_dir: str = "/app/separator/eval/output"
    default_input_dir: str = "/app/separator/eval/input"
    # adele
    gdrive_mix: str = f"{GDRIVE_PREFIX}1zJpyW1fYxHKXDcDH9s5DiBCYiRpraDB3"
