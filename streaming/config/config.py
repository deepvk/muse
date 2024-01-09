from dataclasses import dataclass
from pathlib import Path


@dataclass
class ConverterConfig:
    # WEIGHTS LOAD
    weights_dir            : Path = Path("/app/streaming/weights")      # Path to the directory where the model weight files are stored. 
    weights_LSTM_filename  : str  = "weight_LSTM.pt"                    # This is the filename for the LSTM weights file.
    weights_conv_filename  : str  = "weight_conv.pt"                    # This is the filename for the without CNN weights file.
    gdrive_weights_LSTM_id : str  = "18jT2TYffdRD1fL7wecAiM5nJPM_OKpNB" # This is the Google Drive ID for the LSTM weights file.
    gdrive_weights_conv_id : str  = "1VO07OYbsnCuEJYRSuA8HhjlQnx6dbWX7" # This is the Google Drive ID for the CNN weights file.
    
    # MODEL OPTIONS
    original_model_src : str   = "/app/separator/model"        # This parameter represents the source directory of the original model.
    original_model_dst : str   = "/app/streaming/model"        # This parameter represents the destination directory of the original model.
    model_py_module    : str   = "model.PM_Unet"               # This is the python module where the model is defined
    model_class_name   : str   = "Model_Unet"                  # The name of the model class.
    tflite_model_dst   : str   = "/app/streaming/tflite_model" # This is the destination directory for the TFLite model.
    sample_rate        : int   = 44100                         # Sample rate track
    segment_duration   : float = 1.0                           # This parameter represents the duration of the audio segments that the model will process.


@dataclass
class StreamConfig:
    # STREAM OPTIONS
    converter_script   : str = "/app/streaming/converter.py"       # Path to the script used to convert the pytorch model to tflite.
    sample_rate        : int = 44100                               # Sample rate track.
    nfft               : int = 4096                                # Number of bins used in STFT.
    stft_py_module     : str = "model.STFT"                        # Path to the script STFT.
    default_input_dir  : str = "/app/streaming/input"              # Path to the directory where the input files are stored.
    default_result_dir : str = "/app/streaming/streams"            # Path directory in which processing results are saved.
    gdrive_mix_id      : str = "1zJpyW1fYxHKXDcDH9s5DiBCYiRpraDB3" # The Google Drive ID for the mix file.
    default_duration   : int = 15                                  # Length of an audio stream, in seconds.
