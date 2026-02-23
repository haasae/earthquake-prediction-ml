# Earthquake Magnitude Prediction (Reproducible ML Pipeline)

Train and evaluate regression models that predict **earthquake magnitude** from event metadata (time, location, depth, etc.).
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

