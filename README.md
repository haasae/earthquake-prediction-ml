# Earthquake Prediction with ML

This repository implements an end-to-end ML workflow the SOCR Earthquake Dataset. Here's how:
- download/load dataset
- clean + feature engineer
- visualize global earthquake locations 
- train an Artificial Neural Network to predict earthquake magnitude with regression
- evaluate and save the model + metrics

The purpose of the project was learning machine learning.

## Quickstart

### 1) Create env + install
```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

### 2) Run the full pipeline
```bash
python scripts/run_pipeline.py
```

Artifacts:
- `data/raw/earthquakes.csv` 
- `data/processed/train.csv`, `test.csv`
- `reports/figures/world_scatter.png`
- `models/ann_magnitude.keras`
- `reports/metrics.json`

## Dataset

By default, the pipeline tries to download a public SOCR earthquake CSV. If the URL changes or download fails:
1. Put your dataset file at `data/raw/earthquakes.csv`
2. Ensure it contains (or can be mapped to) columns:
   - `date`, `time`, `latitude`, `longitude`, `depth`, `magnitude`

You can also set `dataset.url` in `config.yaml`.

## Configuration

Edit `config.yaml` to control:
- dataset URL + output paths
- train/test split
- model hyperparameters

## Reproducibility

- Fixed random seeds for NumPy and TensorFlow where possible.
- Saved model + scaler + metrics.

