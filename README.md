# PM-Unet: phase and magnitude aware model for music source separation
 [![githubio](https://img.shields.io/badge/GitHub.io-Audio_Samples-blue?logo=Github&style=flat-square)](https://d-a-yakovlev.github.io/test/)
 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OXlCZgd5KidMDZDUItOIT9ZA4IUJHXsZ?usp=sharing)

## Navigation
1. [Structure](#structure)
2. [Docker](#docker)
3. [Training](#training)
4. [Inference](#inference)

## Structure
- [`separator`](./separator) ‒ main source code with model and dataset implementations and code to train model.
- [`streaming`](./streaming/demo) ‒ source code inference tf-lite version model.

## Docker
#### To set up environment with Docker

If you don't have Docker installed, please follow the links to find installation instructions for [Ubuntu](https://docs.docker.com/desktop/install/linux-install/), [Mac](https://docs.docker.com/desktop/install/mac-install/) or [Windows](https://docs.docker.com/desktop/install/windows-install/).

Build docker image:

    docker build -t pmunet .

Run docker image:

    bash run_docker.sh

## Data
Used dataset [MUSDB18-HQ](https://sigsep.github.io/datasets/musdb.html#musdb18-hq-uncompressed-wav). 

[![Download dataset](https://img.shields.io/badge/Download%20dataset-65c73b)](https://zenodo.org/record/3338373/files/musdb18hq.zip?download=1)

The dataset consists of
150 full-length stereo tracks sampled at 44.1 kHz. providing a
complete audio mix and four main elements: ”vocal”, ”bass”,
”drums” and ”other” for each sample, which can be considered as a target in the context of source separation. The kit
structure offers 100 training compositions and 50 validation
compositions

## Training
1. Configure arguments in `separator/config/config.py`.
2. `cd separator`.
3. Run `python3 separator/pl_model.py`.

## Inference

### Auto local
1. Configure arguments in `separator/config/config.py`.
2. `cd separator`.
3. `python3 inference.py [-IO]`
    - `-I` specify path to mixture, 
    - `-O` output dir, both of them optional. 
    
By default script loads `.pt` file with weights and `sample.wav` from google drive.

#### For example
``` 
python3 inference.py -I path/to/mix -O out_dir
```
With successful script run four audio files (`vocals.wav` and `drums.wav`, `bass.wav`, `other.wav`) will be in `out_dir`. By default in `separator/inference/output`.

**You can download weights manually**

Download one the .pt file below:
 * [LSTM-bottleneck version](https://drive.google.com/file/d/18jT2TYffdRD1fL7wecAiM5nJPM_OKpNB/view?usp=drive_link)
 * [WIthout LSTM-bottleneck version](https://drive.google.com/file/d/1VO07OYbsnCuEJYRSuA8HhjlQnx6dbWX7/view?usp=drive_link)

 ### Streaming
 In streaming section located scripts for: convert model to `tflite` format and run `tflite` model in `"stream mode"`.

1. Configure arguments in `streaming/config/config.py`.
2. `cd streaming`.
3. `python3 runner.py`
