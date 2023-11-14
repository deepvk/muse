from dataclasses import dataclass


@dataclass
class TrainConfig:
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
