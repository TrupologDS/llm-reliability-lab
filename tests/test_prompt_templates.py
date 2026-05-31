from __future__ import annotations

from llm_reliability_lab.prompts import (
    DefaultChatTemplate,
    build_eval_prompt,
    build_sft_messages,
    format_messages,
    format_sft_text,
)


def test_instruction_only_example() -> None:
    example = {"instruction": "Say hello.", "input": "", "response": "Hello."}
    messages = build_sft_messages(example)
    assert [message.role for message in messages] == ["user", "assistant"]
    assert messages[0].content == "Say hello."


def test_instruction_with_input_example() -> None:
    example = {"instruction": "Summarize.", "input": "Long context", "response": "Short."}
    text = format_sft_text(example, DefaultChatTemplate())
    assert "Input:" in text
    assert "Long context" in text


def test_system_prompt_example() -> None:
    example = {
        "system": "You are concise.",
        "instruction": "Answer.",
        "input": "",
        "response": "Done.",
    }
    text = format_sft_text(example)
    assert text.startswith("System: You are concise.")
    assert "Assistant: Done." in text


def test_empty_optional_fields_do_not_render_blank_sections() -> None:
    example = {"system": "", "instruction": "Answer.", "input": "", "response": "Done."}
    text = format_sft_text(example)
    assert "System:" not in text
    assert "Input:" not in text


def test_eval_prompt_includes_required_output_format_constraints() -> None:
    record = {
        "prompt": "Return status.",
        "constraints": {
            "required_format": "json",
            "must_include": ["status"],
            "must_not_include": ["markdown"],
            "max_words": 10,
        },
    }
    prompt = build_eval_prompt(record)
    assert "Return valid JSON only." in prompt
    assert "Must include: status." in prompt
    assert "Must not include: markdown." in prompt
    assert "Assistant:" in prompt


class FakeTokenizer:
    chat_template = "fake"

    def apply_chat_template(self, conversation, *, tokenize: bool, add_generation_prompt: bool) -> str:
        assert not tokenize
        rendered = "|".join(f"{message['role']}={message['content']}" for message in conversation)
        if add_generation_prompt:
            rendered += "|assistant="
        return rendered


def test_tokenizer_chat_template_backend() -> None:
    record = {"prompt": "Return JSON.", "constraints": {"required_format": "json"}}
    prompt = build_eval_prompt(
        record,
        template_backend="tokenizer_chat_template",
        tokenizer=FakeTokenizer(),
    )
    assert prompt.startswith("user=Return JSON.")
    assert prompt.endswith("|assistant=")


def test_tokenizer_chat_template_requires_tokenizer() -> None:
    try:
        format_messages([], template_backend="tokenizer_chat_template")
    except ValueError as exc:
        assert "requires a tokenizer" in str(exc)
    else:
        raise AssertionError("Expected tokenizer_chat_template without tokenizer to fail.")
