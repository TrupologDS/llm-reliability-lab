from __future__ import annotations

from llm_reliability_lab.sft import (
    append_eos_once,
    format_chat_messages_record,
    format_prompt_completion_record,
    format_sft_records,
)


def _example() -> dict:
    return {
        "id": "sft",
        "system": "You are concise.",
        "instruction": "Summarize the context.",
        "input": "The run stores configs and metrics.",
        "response": "The run stores reproducibility artifacts.",
        "source": "test",
    }


class FakeTokenizer:
    eos_token = "<|endoftext|>"


def test_prompt_completion_format_separates_target() -> None:
    formatted = format_prompt_completion_record(_example())
    assert formatted["prompt"].endswith("Assistant:")
    assert formatted["completion"] == "The run stores reproducibility artifacts."
    assert formatted["completion"] in formatted["text"]


def test_chat_messages_format_preserves_roles() -> None:
    formatted = format_chat_messages_record(_example())
    assert [message["role"] for message in formatted["messages"]] == ["system", "user", "assistant"]
    assert "Assistant:" in formatted["text"]


def test_format_sft_records_supports_legacy_text() -> None:
    formatted = format_sft_records([_example()], dataset_format="legacy_text")
    assert list(formatted[0]) == ["id", "text"]


def test_prompt_completion_appends_eos_once() -> None:
    formatted = format_prompt_completion_record(
        _example(),
        tokenizer=FakeTokenizer(),
        append_eos_to_completion=True,
    )
    assert formatted["completion"].endswith(FakeTokenizer.eos_token)
    assert formatted["completion"].count(FakeTokenizer.eos_token) == 1
    assert formatted["text"].count(FakeTokenizer.eos_token) == 1


def test_prompt_completion_does_not_duplicate_existing_eos() -> None:
    example = _example()
    example["response"] = f"Already done.{FakeTokenizer.eos_token}"
    formatted = format_prompt_completion_record(
        example,
        tokenizer=FakeTokenizer(),
        append_eos_to_completion=True,
    )
    assert formatted["completion"] == f"Already done.{FakeTokenizer.eos_token}"
    assert formatted["completion"].count(FakeTokenizer.eos_token) == 1


def test_format_sft_records_appends_eos_only_for_prompt_completion() -> None:
    formatted = format_sft_records(
        [_example()],
        dataset_format="prompt_completion",
        tokenizer=FakeTokenizer(),
        append_eos_to_completion=True,
    )
    assert formatted[0]["completion"].count(FakeTokenizer.eos_token) == 1


def test_append_eos_requires_tokenizer_eos_token() -> None:
    try:
        append_eos_once("completion", tokenizer=None)
    except ValueError as exc:
        assert "requires a tokenizer with eos_token" in str(exc)
    else:
        raise AssertionError("Expected append_eos_once without eos_token to fail.")
