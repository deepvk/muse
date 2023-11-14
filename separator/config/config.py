from dataclasses import dataclass


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

