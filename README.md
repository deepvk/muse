# PM-Unet: phase and magnitude aware model for music source separation
 [![githubio](https://img.shields.io/badge/GitHub.io-Audio_Samples-blue?logo=Github&style=flat-square)](https://d-a-yakovlev.github.io/test/)

Work in progress.

## Structure
- [`src`](./separator) ‒ main source code with model and dataset implementations and code to train model.
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
**Manually local**
1. Download one the .pt file below:
 * [LSTM-bottleneck version](https://drive.google.com/file/d/18jT2TYffdRD1fL7wecAiM5nJPM_OKpNB/view?usp=drive_link)
 * [WIthout LSTM-bottleneck version](https://drive.google.com/file/d/1VO07OYbsnCuEJYRSuA8HhjlQnx6dbWX7/view?usp=drive_link)

2. Put .pt file and your mixture of WAV format in
`separator/`
3. Run
`jupyter notebook test.py`

**Auto local**
0. Configure arguments in `config/config.py`.
1. `cd separator`
2. `python3 eval.py [-IO]` (`-I` specify path to mixture, `-O` output dir, both of them optional. By default load `.pt` file with weigths and `sample.wav` using `gdown`). All data stores in `separator/eval/`.

**On collab**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]()