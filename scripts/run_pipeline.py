from __future__ import annotations

import json
from pathlib import Path
import yaml
import numpy as np

from earthquake_prediction.data import download_file
from earthquake_prediction.preprocess import (
    load_and_clean_csv,
    make_splits_and_scale,
    save_processed_splits,
    save_scaler,
)
from earthquake_prediction.visualize import plot_world_scatter
from earthquake_prediction.train import ModelConfig, set_seeds, train_ann_regression
from earthquake_prediction.evaluate import regression_metrics


def main() -> None:
    cfg = yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))

    raw_path = Path(cfg["dataset"]["raw_path"])
    url = cfg["dataset"]["url"]
    processed_dir = Path(cfg["dataset"]["processed_dir"])

    figures_dir = Path(cfg["artifacts"]["figures_dir"])
    model_path = Path(cfg["artifacts"]["model_path"])
    scaler_path = Path(cfg["artifacts"]["scaler_path"])
    metrics_path = Path(cfg["artifacts"]["metrics_path"])

    # 1) get data
    try:
        download_file(url, raw_path, overwrite=False)
    except Exception as e:
        if not raw_path.exists():
            raise RuntimeError(
                f"Failed to download dataset from {url}. "
                f"Place a CSV at {raw_path} and rerun. Original error: {e}"
            )
        print(f"[WARN] Download failed but {raw_path} exists; continuing. Error: {e}")

    # 2) preprocess
    df = load_and_clean_csv(raw_path)
    save_processed_splits(
        df,
        processed_dir=processed_dir,
        test_size=float(cfg["training"]["test_size"]),
        random_state=int(cfg["training"]["random_state"]),
    )

    prep = make_splits_and_scale(
        df,
        target=str(cfg["training"]["target"]),
        test_size=float(cfg["training"]["test_size"]),
        random_state=int(cfg["training"]["random_state"]),
    )
    save_scaler(prep.scaler, scaler_path)

    # 3) visualize
    plot_world_scatter(df, figures_dir / "world_scatter.png")

    # 4) train
    set_seeds(int(cfg["training"]["random_state"]))
    mcfg = ModelConfig(
        epochs=int(cfg["model"]["epochs"]),
        batch_size=int(cfg["model"]["batch_size"]),
        learning_rate=float(cfg["model"]["learning_rate"]),
        hidden_layers=list(cfg["model"]["hidden_layers"]),
        dropout=float(cfg["model"]["dropout"]),
        early_stopping_patience=int(cfg["model"]["early_stopping_patience"]),
    )

    model, train_info = train_ann_regression(
        prep.X_train, prep.y_train, prep.X_test, prep.y_test, mcfg
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)

    # 5) evaluate
    y_pred = model.predict(prep.X_test, verbose=0).reshape(-1)
    metrics = regression_metrics(prep.y_test, y_pred)

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metrics": metrics,
        "n_rows": int(df.shape[0]),
        "features": prep.feature_names,
        "training": {
            "test_size": float(cfg["training"]["test_size"]),
            "random_state": int(cfg["training"]["random_state"]),
        },
        "model": {
            "type": str(cfg["model"]["type"]),
            "hidden_layers": train_info.get("hidden_layers"),
            "epochs_ran": len(train_info["history"]["loss"]),
        },
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("Done ✅")
    print(f"- Figure:   {figures_dir / 'world_scatter.png'}")
    print(f"- Model:    {model_path}")
    print(f"- Scaler:   {scaler_path}")
    print(f"- Metrics:  {metrics_path}")


if __name__ == "__main__":
    main()
