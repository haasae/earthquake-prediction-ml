# Troubleshooting

## Error: Target column not found

Set the correct target column in `config.yaml`:

```yaml
training:
  target: Mag
```

## Error: dataset not found

Place your CSV at `data/raw/earthquakes.csv`.

## File loads as one column

Your file is likely whitespace-delimited. The loader auto-detects this, but if it fails, ensure the file has a header row.
