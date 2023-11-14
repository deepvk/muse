from dataclasses import dataclass
from pathlib import Path
from typing import Union


GDRIVE_PREFIX = "https://drive.google.com/uc?id="

@dataclass
class TrainConfig:
    device: str = "cuda"

    # datasets
    musdb_path: str = "train/musdb18hq" # relative to separator/
    metadata_train_path: str = "train/metadata/train"
    metadata_test_path: str = "train/metadata/test"
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

    # PM_Unet model instance
    model_num_workers: int = 49
    model_batch_size: int = 2

    # lightning
    max_epochs: int = 1000
    precision: str = "bf16-mixed"

@dataclass
class EvalConfig:
    device: str = "cpu"

    # weights
    weights_dir: Path = Path('eval/weights')
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
    default_result_dir: str = "eval/output"
    default_input_dir: str = "eval/input"
    # adele
    gdrive_mix: str = f"{GDRIVE_PREFIX}1zJpyW1fYxHKXDcDH9s5DiBCYiRpraDB3"
