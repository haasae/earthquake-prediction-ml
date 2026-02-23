from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import yaml

# Allow running without installing as a package
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from earthquake_prediction.io import read_earthquake_csv
from earthquake_prediction.preprocess import prepare_data
from earthquake_prediction.train import train_random_forest
from earthquake_prediction.evaluate import regression_metrics
from earthquake_prediction.visualize import plot_world_scatter


def main() -> None:
    cfg = yaml.safe_load(open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8"))

    raw_path = PROJECT_ROOT / cfg["dataset"]["raw_path"]
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {raw_path}. Put your file there (see README)."
        )

    df = read_earthquake_csv(raw_path)

    target = str(cfg["training"]["target"])
    prep = prepare_data(
        df,
        target=target,
        test_size=float(cfg["training"]["test_size"]),
        random_state=int(cfg["training"]["random_state"]),
    )

    # Visualize
    figures_dir = PROJECT_ROOT / cfg["artifacts"]["figures_dir"]
    plot_world_scatter(df, figures_dir / "world_scatter.png")

    # Train
    base_params = cfg.get("model", {}).get("random_forest", {})
    tuning = cfg.get("tuning", {})

    train_res = train_random_forest(
        prep.X_train,
        prep.y_train,
        base_params=dict(base_params),
        tuning_enabled=bool(tuning.get("enabled", True)),
        cv_folds=int(tuning.get("cv_folds", 5)),
        n_iter=int(tuning.get("n_iter", 20)),
        n_jobs=int(tuning.get("n_jobs", 2)),
        random_state=int(cfg["training"]["random_state"]),
    )

    # Evaluate
    y_pred = train_res.model.predict(prep.X_test)
    metrics = regression_metrics(prep.y_test, y_pred)

    payload = {
        "dataset_rows": int(df.shape[0]),
        "target": target,
        "features": prep.feature_names,
        "model": {
            "type": "random_forest",
            "best_params": train_res.best_params,
            "cv_rmse_mean": train_res.cv_rmse_mean,
            "cv_rmse_std": train_res.cv_rmse_std,
        },
        "metrics": metrics,
    }

    # Save artifacts
    model_path = PROJECT_ROOT / cfg["artifacts"]["model_path"]
    preproc_path = PROJECT_ROOT / cfg["artifacts"]["preprocessor_path"]
    metrics_path = PROJECT_ROOT / cfg["artifacts"]["metrics_path"]

    model_path.parent.mkdir(parents=True, exist_ok=True)
    preproc_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(train_res.model, model_path)
    joblib.dump(prep.scaler, preproc_path)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("Done ✅")
    print(f"- Metrics: {metrics_path}")
    print(f"- Model:   {model_path}")
    print(f"- Scaler:  {preproc_path}")


if __name__ == "__main__":
    main()
