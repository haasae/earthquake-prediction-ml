from pathlib import Path
import pandas as pd
from earthquake_prediction.preprocess import load_and_clean_csv

def test_load_and_clean_csv(tmp_path: Path):
    p = tmp_path / "eq.csv"
    pd.DataFrame({
        "date": ["2020-01-01"],
        "time": ["12:34:56"],
        "latitude": [10.0],
        "longitude": [20.0],
        "depth": [5.0],
        "magnitude": [4.2],
    }).to_csv(p, index=False)

    df = load_and_clean_csv(p)
    assert "month_sin" in df.columns
    assert df.shape[0] == 1
