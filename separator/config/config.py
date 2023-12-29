from dataclasses import dataclass
from pathlib import Path
from typing import Union


from dataclasses import dataclass
from pathlib import Path
from typing import Union


@dataclass
class TrainConfig:

    # DATA OPTIONS
    musdb_path          : str = "musdb18hq" # Directory path where the MUSDB18-HQ dataset is stored.
    metadata_train_path : str = "metadata"  # Directory path for saving training metadata, like track names and lengths.
    metadata_test_path  : str = "metadata1" # Directory path for saving testing metadata.
    segment             : int = 5           # Length (in seconds) of each audio segment used during training.

    # MODEL OPTIONS
    model_source   : tuple = ("drums", "bass", "other", "vocals") # Sources to target in source separation.
    model_depth    : int   = 4                                    # The depth of the U-Net architecture.
    model_channel  : int   = 28                                   # Number of initial channels in U-Net layers.
    is_mono        : bool  = False                                # Indicates whether the input audio should be treated as mono (True) or stereo (False).
    mask_mode      : bool  = False                                # Whether to utilize masking within the model.
    skip_mode      : str   = "concat"                             # Mode of skip connections in U-Net ('concat' for concatenation, 'add' for summation).
    nfft           : int   = 4096                                 # Number of bins used in STFT.
    bottlneck_lstm : bool  = True                                 # Determines whether to use LSTM layers as bottleneck in the U-Net architecture.
    layers         : int   = 2                                    # Number of LSTM layers if bottleneck.
    stft_flag      : bool  = True                                 # A flag to decide whether to apply the STFT is required for tflite. 

    # TRAIN OPTIONS
    device                   : str  = "cuda"       # The computing platform for training: 'cuda' for NVIDIA GPUs or 'cpu'.
    batch_size               : int  = 6            # Batch size for training.
    shuffle_train            : bool = True         # Whether to shuffle the training dataset.
    shuffle_valid            : bool = False        # Whether to shuffle the valid dataset.
    drop_last                : bool = True         # Whether to drop the last incomplete batch in train data.
    num_workers              : int  = 2            # Number of worker processes used for loading data.
    metric_monitor_mode      : str  = "min"        # Strategy for monitoring metrics to save model checkpoints.
    save_top_k_model_weights : int  = 1            # Number of best-performing model weights to save based on the monitored metric.
    
    factor                   : int = 1             # Factors for different components of the loss function.
    c_factor                 : int = 1

    loss_nfft                : tuple = (4096,)     # Number of FFT bins for calculating loss.
    gamma                    : float = 0.3         # Gamma parameter for adjusting the focus of the loss on certain aspects of the audio spectrum.
    lr                       : float = 0.5 * 3e-3  # Learning rate for the optimizer.
    T_0                      : int   = 40          # Period of the cosine annealing schedule in learning rate adjustment.
    max_epochs               : int   = 100         # Maximum number of training epochs.
    precision                : str   = 16          # Precision of training computations.
    grad_clip                : float = 0.5         # Gradient clipping value.

    # AUGMENTATION OPTIONS
    proba_shift           : float = 0.5        # Probability of applying the shift.
    shift                 : int   = 8192       # Maximum number of samples for the shift.
    proba_flip_channel    : float = 1          # Probability of applying the flip left-right channels.
    proba_flip_sign       : float = 1          # Probability of applying the sign flip.
    pitchshift_proba      : float = 0.2        # Probability of applying pitch shift.
    vocals_min_semitones  : int   = -5         # The lower limit of vocal semitones.
    vocals_max_semitones  : int   = 5          # The upper limit of vocal semitones.
    other_min_semitones   : int   = -2         # The lower limit of non-vocal semitones.
    other_max_semitones   : int   = 2          # The upper limit of non-vocal semitones.
    pitchshift_flag_other : bool  = False      # Flag to enable pitch shift augmentation on non-vocal sources.
    time_change_proba     : float = 0.2        # Probability of applying time stretching.
    time_change_factors   : tuple = (0.8, 0.85, 0.9, 0.95, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3) # Factors for time stretching/compression, defining the range and intensity of this augmentation.
    remix_proba           : float = 1          # Probability of remixing audio tracks.
    remix_group_size      : int   = batch_size # Size of groups within which shuffling occurs.
    scale_proba           : float = 1          # Probability of applying the scaling.
    scale_min             : float = 0.25       # Minimum scaling factor.
    scale_max             : float = 1.25       # Maximum scaling factor.
    fade_mask_proba       : float = 0.1        # Probability of applying a fade effect.
    double_proba          : float = 0.1        # Probability of doubling one channel's audio to both channels.
    reverse_proba         : float = 0.2        # Probability of reversing a segment of the audio track.
    mushap_proba          : float = 0.0        # Probability create mashups.
    mushap_depth          : int   = 2          # Number of tracks to mix.


@dataclass
class InferenceConfig:
    GDRIVE_PREFIX = "https://drive.google.com/uc?id=" # Google Drive URL

    # MODEL OPTIONS
    weights_dir           : Path = Path("/app/separator/inference/weights")            # file name where weights are saved
    weights_LSTM_filename : str  = "weight_LSTM.pt"                                    # file name model with LSTM
    weights_conv_filename : str  = "weight_conv.pt"                                    # file name model without LSTM
    gdrive_weights_LSTM   : str  = f"{GDRIVE_PREFIX}1uhAVMvW3x-KL2T2-VkjKjn9K7dTJnoyo" # Google Drive URL that directs weights LSTM
    gdrive_weights_conv   : str  = f"{GDRIVE_PREFIX}1VO07OYbsnCuEJYRSuA8HhjlQnx6dbWX7" # Google Drive URL that directs weights without_LSTM
    device                : str  = "cpu"                                               # The computing platform for inference

    # INFERENCE OPTIONS
    segment            : int              = 7                                 # Length (in seconds) of each audio segment used during inference.
    overlap            : float            = 0.2                               # overlapping segments at the beginning of the track and at the end
    offset             : Union[int, None] = None                              # start of segment to split
    duration           : Union[int, None] = None                              # end of segment to split
    sample_rate        : int              = 44100                             # sample rate track
    num_channels       : int              = 2                                 # Number of channels in the audio track 
    default_result_dir : str              = "/app/separator/inference/output" # path file output tracks
    default_input_dir  : str              = "/app/separator/inference/input"  # path file input track
    
    # TEST TRACK
    gdrive_mix : str = f"{GDRIVE_PREFIX}1zJpyW1fYxHKXDcDH9s5DiBCYiRpraDB3" # Google Drive URL that directs test track
