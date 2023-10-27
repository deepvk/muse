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

For proper contributions, also use dev requirements:
```bash
pip install -r requirements-dev.txt
```
## Docker
Docker Will here
## Data
We use dataset with [MUSDB18-HQ](https://sigsep.github.io/datasets/musdb.html#musdb18-hq-uncompressed-wav)
The dataset consists of
150 full-length stereo tracks sampled at 44.1 kHz. providing a
complete audio mix and four main elements: ”vocal”, ”bass”,
”drums” and ”other” for each sample, which can be considered as a target in the context of source separation. The kit
structure offers 100 training compositions and 50 validation
compositions

## Training
1. Run `python -m separator.pl_model`.🙂