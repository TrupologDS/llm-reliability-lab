# Regression Report

## Baseline vs Candidate Comparison

| category | metric | baseline | candidate | delta | threshold | regression | severity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| factuality | must_include_coverage | 0.6000 | 0.6500 | 0.0500 | 0.1000 | False | low |
| factuality | format_compliance_rate | 1.0000 | 1.0000 | 0.0000 | 0.0500 | False | low |
| factuality | must_not_include_violation_rate | 0.3000 | 0.3000 | 0.0000 | 0.0500 | False | low |
| factuality | avg_word_count | 8.0000 | 7.9000 | -0.1000 | 40.0000 | False | low |
| factuality | keyword_coverage | 0.6000 | 0.6500 | 0.0500 | 0.0000 | False | low |
| factuality | category_score | 0.8250 | 0.8375 | 0.0125 | 0.0500 | False | low |
| format_compliance | must_include_coverage | 0.8000 | 0.7000 | -0.1000 | 0.1000 | True | low |
| format_compliance | format_compliance_rate | 0.1000 | 0.6000 | 0.5000 | 0.0500 | False | low |
| format_compliance | must_not_include_violation_rate | 0.3000 | 0.1000 | -0.2000 | 0.0500 | False | low |
| format_compliance | avg_word_count | 18.1000 | 23.5000 | 5.4000 | 40.0000 | False | low |
| format_compliance | keyword_coverage | 0.8000 | 0.7000 | -0.1000 | 0.0000 | True | high |
| format_compliance | category_score | 0.6500 | 0.8000 | 0.1500 | 0.0500 | False | low |
| instruction_following | must_include_coverage | 0.9167 | 0.9000 | -0.0167 | 0.1000 | False | low |
| instruction_following | format_compliance_rate | 0.8000 | 0.5000 | -0.3000 | 0.0500 | True | high |
| instruction_following | must_not_include_violation_rate | 0.1000 | 0.2000 | 0.1000 | 0.0500 | True | medium |
| instruction_following | avg_word_count | 16.9000 | 12.0000 | -4.9000 | 40.0000 | False | low |
| instruction_following | keyword_coverage | 0.9167 | 0.9000 | -0.0167 | 0.0000 | True | high |
| instruction_following | category_score | 0.9042 | 0.8000 | -0.1042 | 0.0500 | True | medium |
| multilingual | must_include_coverage | 0.6667 | 0.7500 | 0.0833 | 0.1000 | False | low |
| multilingual | format_compliance_rate | 0.8000 | 0.9000 | 0.1000 | 0.0500 | False | low |
| multilingual | must_not_include_violation_rate | 0.0000 | 0.0000 | 0.0000 | 0.0500 | False | low |
| multilingual | avg_word_count | 28.8000 | 23.0000 | -5.8000 | 40.0000 | False | low |
| multilingual | keyword_coverage | 0.6667 | 0.7500 | 0.0833 | 0.0000 | False | low |
| multilingual | category_score | 0.8667 | 0.9125 | 0.0458 | 0.0500 | False | low |
| overall | must_include_coverage | 0.7056 | 0.6917 | -0.0139 | 0.1000 | False | low |
| overall | format_compliance_rate | 0.6667 | 0.7833 | 0.1167 | 0.0500 | False | low |
| overall | must_not_include_violation_rate | 0.1500 | 0.1000 | -0.0500 | 0.0500 | False | low |
| overall | avg_word_count | 28.9333 | 19.6333 | -9.3000 | 40.0000 | False | low |
| overall | keyword_coverage | 0.7056 | 0.6917 | -0.0139 | 0.0000 | True | high |
| overall | category_score | 0.7889 | 0.8313 | 0.0424 | 0.0500 | False | low |
| refusal_behavior | must_include_coverage | 0.5500 | 0.5500 | 0.0000 | 0.1000 | False | low |
| refusal_behavior | format_compliance_rate | 0.5000 | 0.9000 | 0.4000 | 0.0500 | False | low |
| refusal_behavior | must_not_include_violation_rate | 0.0000 | 0.0000 | 0.0000 | 0.0500 | False | low |
| refusal_behavior | avg_word_count | 73.1000 | 39.1000 | -34.0000 | 40.0000 | False | low |
| refusal_behavior | keyword_coverage | 0.5500 | 0.5500 | 0.0000 | 0.0000 | False | low |
| refusal_behavior | category_score | 0.6625 | 0.7875 | 0.1250 | 0.0500 | False | low |
| robustness | must_include_coverage | 0.7000 | 0.6000 | -0.1000 | 0.1000 | False | low |
| robustness | format_compliance_rate | 0.8000 | 0.8000 | 0.0000 | 0.0500 | False | low |
| robustness | must_not_include_violation_rate | 0.2000 | 0.0000 | -0.2000 | 0.0500 | False | low |
| robustness | avg_word_count | 28.7000 | 12.3000 | -16.4000 | 40.0000 | False | low |
| robustness | keyword_coverage | 0.7000 | 0.6000 | -0.1000 | 0.0000 | True | high |
| robustness | category_score | 0.8250 | 0.8500 | 0.0250 | 0.0500 | False | low |
| safe_prompts | refusal_rate_increase_on_safe_prompts | 0.0000 | 0.0000 | 0.0000 | 0.1000 | False | low |
| refusal_prompts | refusal_rate_decrease_on_refusal_prompts | 0.2000 | 0.4000 | 0.2000 | 0.1000 | False | low |

## Flagged Regressions

| category | metric | baseline | candidate | delta | threshold | regression | severity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| format_compliance | keyword_coverage | 0.8000 | 0.7000 | -0.1000 | 0.0000 | True | high |
| instruction_following | format_compliance_rate | 0.8000 | 0.5000 | -0.3000 | 0.0500 | True | high |
| instruction_following | keyword_coverage | 0.9167 | 0.9000 | -0.0167 | 0.0000 | True | high |
| overall | keyword_coverage | 0.7056 | 0.6917 | -0.0139 | 0.0000 | True | high |
| robustness | keyword_coverage | 0.7000 | 0.6000 | -0.1000 | 0.0000 | True | high |
| instruction_following | must_not_include_violation_rate | 0.1000 | 0.2000 | 0.1000 | 0.0500 | True | medium |
| instruction_following | category_score | 0.9042 | 0.8000 | -0.1042 | 0.0500 | True | medium |
| format_compliance | must_include_coverage | 0.8000 | 0.7000 | -0.1000 | 0.1000 | True | low |

## Interpretation

- High-severity flags mostly come from keyword coverage and format compliance drops.
- The most important behavioral regression is instruction-following: the candidate often gives a cleaner natural-language answer but violates exact-output constraints.
- Format-compliance aggregate improved, but specific keyword coverage declined in some format prompts.
- Robustness keyword coverage declined even though average length decreased.

## Review Priority

1. Exact extraction prompts such as `if_006`.
2. Exact length/count prompts such as `if_010`, `if_007`, and `rob_006`.
3. Regex/exact string prompts such as `fmt_006`.
4. Robustness prompts where brevity removes required content.
