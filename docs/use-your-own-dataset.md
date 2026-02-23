# How to use your own dataset

**Goal:** Run the pipeline on your own earthquake CSV.

## Requirements

Your file must include:
- Latitude, Longitude, Depth columns
- A magnitude target column (e.g. `Mag`)
- Date + Time columns (the project auto-detects common names)

## Steps

1) Place your file at:

```text
data/raw/earthquakes.csv
```

2) Update the target column in `config.yaml`:

```yaml
training:
  target: Mag
```

3) Run:

```bash
python scripts/run_pipeline.py
```

## Common issue: whitespace-delimited files

If your file looks like columns are aligned with spaces (not commas), the loader will parse whitespace automatically.
