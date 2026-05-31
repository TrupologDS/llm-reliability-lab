# Evaluation Suites

These JSONL files contain synthetic evaluation prompts for reliability experiments. They are not training data and should not be mixed into SFT or preference datasets.

The suites are intended for baseline vs candidate comparisons:

- `instruction_following.jsonl`
- `format_compliance.jsonl`
- `multilingual.jsonl`
- `refusal_behavior.jsonl`
- `grounded_qa.jsonl`
- `robustness_perturbations.jsonl`
- `plumbing/repo_internal_plumbing.jsonl` contains repo-specific smoke prompts and is not included in the real merged suite by default.

They are not comprehensive benchmarks. Treat them as project-local eval assets that can be expanded, reviewed, versioned, and paired with qualitative analysis.
