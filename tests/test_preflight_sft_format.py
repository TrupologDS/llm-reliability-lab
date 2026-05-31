from __future__ import annotations

import pytest
from scripts.preflight_sft_format import (
    IGNORE_INDEX,
    PreflightError,
    validate_collated_labels,
    validate_response_template_presence,
)


def test_preflight_fails_when_all_labels_are_masked() -> None:
    labels = [[IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]]
    input_ids = [[10, 11, 12]]

    with pytest.raises(PreflightError, match="All labels are masked"):
        validate_collated_labels(labels, input_ids)


def test_preflight_fails_when_response_template_absent() -> None:
    tokenized = [{"input_ids": [1, 2, 3, 4]}]
    response_template_ids = [9, 9]

    with pytest.raises(PreflightError, match="response_template was not found"):
        validate_response_template_presence(tokenized, response_template_ids, "<assistant>")


def test_preflight_reports_trainable_label_tokens_when_valid() -> None:
    labels = [[IGNORE_INDEX, IGNORE_INDEX, 42, 43]]
    input_ids = [[10, 11, 12, 13]]

    stats = validate_collated_labels(labels, input_ids)

    assert stats[0].input_tokens == 4
    assert stats[0].masked_label_tokens == 2
    assert stats[0].trainable_label_tokens == 2


def test_preflight_fails_when_zero_labels_are_masked() -> None:
    labels = [[10, 11, 12]]
    input_ids = [[10, 11, 12]]

    with pytest.raises(PreflightError, match="Zero labels are masked"):
        validate_collated_labels(labels, input_ids)
