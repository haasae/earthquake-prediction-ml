# Earthquake Magnitude Prediction (Reproducible ML Pipeline)

Train and evaluate a regression model that predicts **earthquake magnitude** from event metadata such as location, depth, and timestamp-derived features.

## What this project does

This repository provides an end-to-end, scriptable ML workflow:

- Robustly loads earthquake CSV files (comma-separated or whitespace-delimited).
- Builds features from latitude/longitude/depth and parsed date/time fields.
- Trains a Random Forest regressor (optionally with randomized hyperparameter search).
- Evaluates model quality with standard regression metrics.
- Saves reusable artifacts (trained model + scaler) and reports.

## Project structure

```text
.
├── config.yaml                         # Main project configuration
├── scripts/
│   ├── run_pipeline.py                 # Full training + evaluation pipeline
│   └── predict.py                      # Load saved artifacts and run sample predictions
├── src/earthquake_prediction/
│   ├── io.py                           # Data loading and schema checks
│   ├── features.py                     # Time + numeric feature engineering
│   ├── preprocess.py                   # Split + scaling pipeline
│   ├── train.py                        # Model training + optional tuning
│   ├── evaluate.py                     # Regression metrics
│   └── visualize.py                    # Diagnostic plotting
├── docs/
│   ├── quickstart.md
│   ├── tutorial.md
│   └── how-to/
└── tests/
```

## Quickstart (about 2 minutes)

### 1) Create a virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Add your dataset

Place your CSV at:

```text
data/raw/earthquakes.csv
```

The pipeline expects at least:

- `Latitude`
- `Longitude`
- `Depth`
- A target magnitude column (default: `Mag`)
- Date and time columns (common names are auto-detected, e.g. `Date(YYYY/MM/DD)` + `Time`)

### 3) (Optional) adjust configuration

Edit `config.yaml` if needed:

- `training.target` for your magnitude column name.
- `tuning.enabled` to enable/disable hyperparameter tuning.
- `tuning.n_iter` / `tuning.n_jobs` to control speed.

### 4) Run the full pipeline

```bash
python scripts/run_pipeline.py
```

## Outputs

After a successful run, you should see:

- `reports/metrics.json` (dataset info + model metadata + evaluation metrics)
- `models/model.joblib` (trained Random Forest model)
- `models/preprocessor.joblib` (fitted `StandardScaler`)
- `reports/figures/world_scatter.png` (basic world scatter visualization)

## Inference script

You can load saved artifacts and print sample predictions:

```bash
python scripts/predict.py --input data/raw/earthquakes.csv --model models/model.joblib --scaler models/preprocessor.joblib --target Mag
```

> Note: `scripts/predict.py` is a lightweight example utility for artifact loading and quick prediction output.

## Configuration reference

`config.yaml` controls dataset paths, training settings, model defaults, tuning behavior, and artifact locations.

Key fields:

- `dataset.raw_path`: input CSV path.
- `training.target`: target magnitude column.
- `training.test_size`, `training.random_state`: train/test split behavior.
- `model.random_forest.*`: baseline Random Forest parameters.
- `tuning.*`: randomized search controls.
- `artifacts.*`: output locations for model, scaler, metrics, and figures.

## Run tests

```bash
pytest -q
```

## Troubleshooting

- **Dataset not found**: ensure file exists at `data/raw/earthquakes.csv` (or update `dataset.raw_path` in `config.yaml`).
- **Target column not found**: set the correct magnitude column in `training.target`.
- **File loaded as one column**: your dataset may be whitespace-delimited; this loader attempts to auto-handle that format.
- **Training is slow**: reduce `tuning.n_iter`, set `tuning.n_jobs: 1`, or disable tuning with `tuning.enabled: false`.

## Documentation

- Quickstart: `docs/quickstart.md`
- Tutorial: `docs/tutorial.md`
- How-to guides:
  - `docs/how-to/use-your-own-dataset.md`
  - `docs/how-to/speed-up-training.md`
  - `docs/how-to/troubleshoot.md`
