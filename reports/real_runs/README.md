# Real Runs

Use this directory for experiment reports generated from real model outputs.

Each real run should include:

- Model versions and exact revisions for baseline and candidate models.
- Dataset names, versions, splits, filtering, and local file hashes where relevant.
- Hardware details: GPU model/count, CPU, RAM, operating system, CUDA version, driver version.
- Exact commands for SFT, DPO, evaluation, comparison, and report generation.
- Config files or immutable copies of config values.
- Metrics with category-level aggregation and regression thresholds.
- Qualitative examples for failures and notable behavior changes.
- Limitations and known sources of uncertainty.
- Next experiments or model/data changes suggested by the results.

Do not add fabricated metrics, guessed benchmark numbers, or claims of improvement without real runs.
