# Quickstart

**Goal:** Train a model and generate `reports/metrics.json`.

## Steps

1) Install dependencies

```bash
pip install -r requirements.txt
```

2) Add your dataset

Place your CSV at:

```text
data/raw/earthquakes.csv
```

3) Run

```bash
python scripts/run_pipeline.py
```

## Outputs

- `reports/metrics.json`
- `models/model.joblib`
- `models/preprocessor.joblib`
