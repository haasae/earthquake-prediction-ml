# Earthquake Magnitude Prediction (Reproducible ML Pipeline)

Train and evaluate regression models that predict **earthquake magnitude** from event metadata (time, location, depth, etc.).

This project is designed to look and feel like an industry ML repo:
- reproducible entrypoints
- robust data loading (comma **or** whitespace-delimited)
- saved metrics and artifacts
- documentation split by purpose (quickstart / how-to / explanations / reference)

## Quickstart (2 minutes)

1) Create environment + install dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2) Put your dataset at:

```text
data/raw/earthquakes.csv
```

3) Run the full pipeline

```bash
python scripts/run_pipeline.py
```

Outputs:
- `reports/metrics.json`
- `models/model.joblib` + `models/preprocessor.joblib`
- `reports/figures/world_scatter.png`

## Documentation

- Quickstart: `docs/quickstart.md`
- Tutorial: `docs/tutorial.md`
- How-to guides: `docs/how-to/`
- Explanations: `docs/explanations/`
- Reference: `docs/reference/`

## What this project is (and isn’t)

This is a portfolio-grade ML engineering project demonstrating data ingestion, feature engineering, evaluation and reproducibility.

It is **not** a scientific earthquake prediction system.
