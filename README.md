# PM-Unet: phase and magnitude aware model for music source separation
 [![githubio](https://img.shields.io/badge/GitHub.io-Audio_Samples-blue?logo=Github&style=flat-square)](https://d-a-yakovlev.github.io/test/)

Work in progress.

## Structure
- [`separator`](./separator) ‒ main source code with model and dataset implementations and code to train model.
- [`streaming`](./streaming/demo) ‒ source code inference tf-lite version model.

## Requirements
Create virtual environment with `venv` or `conda` and install requirements:
```bash
pip install -r requirements.txt
```
## Docker
Docker Will here
## Data
We use dataset [MUSDB18-HQ](https://sigsep.github.io/datasets/musdb.html#musdb18-hq-uncompressed-wav). 

[![Download dataset](https://img.shields.io/badge/Download%20dataset-65c73b)](https://zenodo.org/record/3338373/files/musdb18hq.zip?download=1)

The dataset consists of
150 full-length stereo tracks sampled at 44.1 kHz. providing a
complete audio mix and four main elements: ”vocal”, ”bass”,
”drums” and ”other” for each sample, which can be considered as a target in the context of source separation. The kit
structure offers 100 training compositions and 50 validation
compositions

## Training
0. Configure arguments in `config/config.py`.
1. Run `python -m separator.pl_model.py`.🙂

## Inference

### Auto local
1. Configure arguments in `separator/config/config.py`.
2. `cd separator`
3. `python3 eval.py [-IO]`
    - `-I` specify path to mixture, 
    - `-O` output dir, both of them optional. 
    
By default load `.pt` file with weigths and `sample.wav` using `gdown`

#### For example
``` 
python3 eval.py -I path/to/mix -O out_dir
```
You should get four separated audio files (vocals.wav and drums.wav, bass.wav, other.wav) in folder. 
All data stores in `separator/eval/output` by default (you can all specificate in config).

**You can download weights manually**

Download one the .pt file below:
 * [LSTM-bottleneck version](https://drive.google.com/file/d/18jT2TYffdRD1fL7wecAiM5nJPM_OKpNB/view?usp=drive_link)
 * [WIthout LSTM-bottleneck version](https://drive.google.com/file/d/1VO07OYbsnCuEJYRSuA8HhjlQnx6dbWX7/view?usp=drive_link)

### On collab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]()