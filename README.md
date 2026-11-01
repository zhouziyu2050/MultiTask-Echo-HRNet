# HRNet-W18-lite Echocardiographic Measurement

Code accompanying the article, including preprocessing, training, and evaluation
of HRNet-W18-lite for six echocardiographic measurements: IVS, LVPW, LVID, LA,
aortic root, and ascending aorta.

## Installation

Run commands from this directory. A CUDA GPU is required; training uses bfloat16.

```sh
conda create -n measurement3 python=3.13 -y
conda activate measurement3
pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Files

```text
1-preprocess.py           Extract frames and generate training annotation columns
2-train.py                Train, validate, and export reports
3-evaluate.py             Evaluate a measurement checkpoint
model.py                  HRNet-W18-lite architecture
utils.py                  Visualization
annotations/measurements.csv  Manual and teacher-generated annotations
weights/trained/          Final measurement checkpoint
weights/pretrained/       Pretrained backbone
data/                     Videos and processed frames
outputs/                  Generated checkpoints and reports
```

## Data and Weights

Download the videos following [data/README.md](data/README.md), preserving their
original filenames:

```text
data/raw/EchoNet-LVH/
  ... original AVI videos, including subdirectories ...
```

The annotation table is included in `annotations/measurements.csv`, and the final
checkpoint is `weights/trained/hrnet_w18_lite.ckpt`. For training, download the
backbone weight following [weights/pretrained/README.md](weights/pretrained/README.md).

## Run

Preprocess the videos and annotations:

```sh
python 1-preprocess.py
```

This creates `data/processed/frames.npy` (640x480 BGR) and
`data/processed/measurements.csv` for training and evaluation.

Evaluate the supplied final checkpoint:

```sh
python 3-evaluate.py
```

Train a model:

```sh
python 2-train.py
```

To evaluate a newly trained checkpoint, set its path before running evaluation:

```powershell
$env:HRNET_CHECKPOINT = 'D:\runs\runs_lvh_fusion_ACTUAL_TIMESTAMP\fusion_weights_best.ckpt'
python 3-evaluate.py
```

Model and training settings follow the article. Paths can be configured through
environment variables:

| Environment variable | Default | Purpose |
|---|---|---|
| `ECHONET_DATA_ROOT` | `data/` | Contains `raw/EchoNet-LVH/` and generated `processed/` |
| `HRNET_OUTPUT_ROOT` | `outputs/` | Training and evaluation destination |
| `HRNET_CHECKPOINT` | Supplied final checkpoint | Checkpoint used by evaluation |

## Results

Training saves `fusion_weights_best.ckpt` and reports under
`outputs/runs_lvh_fusion_<timestamp>/`. Standalone evaluation writes reports to
`outputs/evaluation/`.

Reports include `summary.json`, `summary.csv`, per-task measurement CSVs, and
endpoint visualizations. Lengths are in cm, MAPE is a percentage, and CP is a
proportion in [0,1].