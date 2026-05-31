# Reports

Reports are separated by purpose:

- `sample/`: generated output from the deterministic sample pipeline check. These files are not benchmark results.
- `templates/`: reusable Markdown templates for future experiments.
- `real_runs/`: real experiment reports with model versions, dataset versions, hardware, commands, metrics, and examples.

Generated reports must stay honest:

- Keep TODO sections until real outputs exist.
- Do not report mock sample outputs as benchmark numbers.
- Include model versions, dataset versions, configs, seeds, hardware, and commands for real runs.

Run the sample path with:

```bash
make evaluate-sample
make report-sample
```

The sample path validates plumbing only.
