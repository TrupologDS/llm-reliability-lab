# Error Analysis

## Failure Taxonomy Counts

| failure_code | baseline_count | candidate_count | delta | description |
| --- | --- | --- | --- | --- |
| F01_FORMAT_NONCOMPLIANCE | 20 | 13 | -7 | Required format not followed. |
| F02_CONSTRAINT_IGNORED | 30 | 27 | -3 | Explicit user constraint ignored. |
| F03_UNSUPPORTED_CLAIM | 7 | 6 | -1 | Unsupported or over-specific claim risk. |
| F05_UNDER_REFUSAL | 4 | 3 | -1 | Unsafe/refusal prompt complied with. |
| F06_LENGTH_BIAS | 11 | 6 | -5 | Response longer than requested or baseline. |
| F07_PROMPT_BRITTLENESS | 4 | 4 | 0 | Prompt perturbation changed behavior. |
| F08_MULTILINGUAL_FAILURE | 5 | 4 | -1 | Non-English or language-control issue. |
| F09_CONTEXT_CONTRADICTION | 3 | 3 | 0 | Contradicts prompt/context. |
| F10_LOW_USEFULNESS | 10 | 5 | -5 | Generic or shallow answer. |

## Observed Failure Patterns

- `F02_CONSTRAINT_IGNORED` remains the most common failure for both baseline and candidate.
- `F01_FORMAT_NONCOMPLIANCE` decreased from 20 to 13 assignments, but exact-output failures remain visible.
- `F06_LENGTH_BIAS` decreased from 11 to 6 assignments, consistent with shorter candidate responses.
- `F10_LOW_USEFULNESS` decreased from 10 to 5 assignments, but manual examples show some concision came with format regressions.
- `F08_MULTILINGUAL_FAILURE`, `F07_PROMPT_BRITTLENESS`, and `F09_CONTEXT_CONTRADICTION` remain present and need targeted eval expansion.

## Debugging Hypotheses

- UltraChat-style SFT likely encouraged concise helpful answers but did not teach strict extraction, exact-count, or regex-style compliance.
- Completion-only masking appears to work, so the observed regressions are more likely data/task-mix issues than prompt-token training leakage.
- The adapter may have shifted style toward natural prose, which can hurt prompts that require returning only a label, version, or fixed number of items.

## Next Experiments

- Add a small supervised mix of exact-output examples: labels, semantic versions, JSON-only, and fixed-count bullets.
- Add validation splits specifically for exact-output constraints before running a larger SFT pass.
- Compare a smaller learning rate or fewer SFT examples to reduce instruction-following drift.
- Add human review labels for examples where rule metrics and semantic usefulness disagree.
