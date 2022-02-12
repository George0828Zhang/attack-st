# Attack Speech Translation

## Setup

1. Install fairseq
```bash
git clone https://github.com/George0828Zhang/fairseq.git
cd fairseq
git checkout 8b526ad
python setup.py build_ext --inplace
pip install .
```
2. Install dependencies
```bash
pip install -r requirements.txt
```

## Data and checkpoints
Download [here](https://ntucc365-my.sharepoint.com/:f:/g/personal/r09922057_ntu_edu_tw/EkdJ5w30xwNGqOSCNabH7JEBn1ULv0Fz1Nuq0CQXg-92lw?e=lkVrUX).
- Data, dictionary and spm model is in `covost2-atk/en/`.
- Checkpoints for each model is in respective folders.

## Usage
See [attack.py](https://github.com/George0828Zhang/attack-st/blob/main/eval/attack.py).
You may want to:
- Update `config_attack.yaml` for the paths to vocab and spm files. The use `--config-yaml <newyaml>` when invoking `attack.py`.