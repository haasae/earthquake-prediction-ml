# How to speed up training

If hyperparameter tuning feels slow on your laptop, reduce the tuning workload:

1) Lower the number of search iterations:

```yaml
tuning:
  n_iter: 10
```

2) Reduce parallelism:

```yaml
tuning:
  n_jobs: 1
```

3) Disable tuning (baseline model only):

```yaml
tuning:
  enabled: false
```
