# Tutorial: from raw CSV to evaluated model

By the end of this tutorial, you will:

1. Load a raw earthquake CSV (comma or whitespace-delimited)
2. Create a small feature set (location, depth, time features)
3. Train a baseline model (Random Forest)
4. Evaluate with RMSE/MAE/R² and save artifacts

## 1) Add your dataset

Put your file at `data/raw/earthquakes.csv`.

## 2) Configure the target

In `config.yaml`, set:

```yaml
training:
  target: Mag
```

If your dataset uses a different name (e.g. `magnitude`), update it here.

## 3) Run the pipeline

```bash
python scripts/run_pipeline.py
```

## 4) Inspect results

- Metrics: `reports/metrics.json`
- Model: `models/model.joblib`

Next:
- Use your own schema: `docs/how-to/use-your-own-dataset.md`
- Speed up tuning: `docs/how-to/speed-up-training.md`
