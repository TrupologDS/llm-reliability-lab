"""Explicit, testable prompt templates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Literal, Protocol

Role = Literal["system", "user", "assistant"]
TemplateBackend = Literal["tokenizer_chat_template", "default_chat_template"]


class ChatTemplateTokenizer(Protocol):
    """Tokenizer protocol for chat-template rendering."""

    chat_template: str | None

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str: ...


@dataclass(frozen=True)
class ChatMessage:
    """A minimal chat message representation independent of any tokenizer."""

    role: Role
    content: str


class DefaultChatTemplate:
    """Simple human-readable chat template used when tokenizer chat templates are unavailable."""

    role_labels: ClassVar[dict[str, str]] = {
        "system": "System",
        "user": "User",
        "assistant": "Assistant",
    }

    def format_messages(
        self,
        messages: list[ChatMessage],
        *,
        add_generation_prompt: bool = False,
    ) -> str:
        """Format chat messages as plain text."""

        rendered = [
            f"{self.role_labels[message.role]}: {message.content.strip()}"
            for message in messages
            if message.content.strip()
        ]
        if add_generation_prompt:
            rendered.append("Assistant:")
        return "\n\n".join(rendered).strip()


def format_messages(
    messages: list[ChatMessage],
    *,
    template_backend: TemplateBackend = "default_chat_template",
    tokenizer: ChatTemplateTokenizer | None = None,
    add_generation_prompt: bool = False,
) -> str:
    """Format messages with either the tokenizer chat template or the local fallback."""

    if template_backend == "tokenizer_chat_template":
        if tokenizer is None or not getattr(tokenizer, "chat_template", None):
            msg = (
                "template_backend='tokenizer_chat_template' requires a tokenizer with "
                "a chat_template. Set template_backend='default_chat_template' for the "
                "explicit fallback."
            )
            raise ValueError(msg)
        return tokenizer.apply_chat_template(
            messages_to_dicts(messages),
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        ).strip()

    return DefaultChatTemplate().format_messages(
        messages,
        add_generation_prompt=add_generation_prompt,
    )


def messages_to_dicts(messages: list[ChatMessage]) -> list[dict[str, str]]:
    """Convert message dataclasses to tokenizer-compatible dictionaries."""

    return [
        {"role": message.role, "content": message.content.strip()} for message in messages if message.content.strip()
    ]


def build_sft_messages(
    example: dict[str, Any],
    *,
    include_response: bool = True,
) -> list[ChatMessage]:
    """Build chat messages for one SFT record."""

    messages: list[ChatMessage] = []
    system = str(example.get("system", "")).strip()
    if system:
        messages.append(ChatMessage("system", system))

    user_content = build_instruction_user_content(example)
    messages.append(ChatMessage("user", user_content))

    response = str(example.get("response", "")).strip()
    if include_response and response:
        messages.append(ChatMessage("assistant", response))
    return messages


def build_instruction_user_content(example: dict[str, Any]) -> str:
    """Render instruction and optional input/context for a user message."""

    instruction = str(example.get("instruction", "")).strip()
    input_text = str(example.get("input", "")).strip()
    if input_text:
        return f"{instruction}\n\nInput:\n{input_text}"
    return instruction


def format_sft_text(
    example: dict[str, Any],
    template: DefaultChatTemplate | None = None,
    *,
    template_backend: TemplateBackend = "default_chat_template",
    tokenizer: ChatTemplateTokenizer | None = None,
) -> str:
    """Render one normalized SFT record into trainer text."""

    if template is not None:
        return template.format_messages(build_sft_messages(example))
    return format_messages(
        build_sft_messages(example),
        template_backend=template_backend,
        tokenizer=tokenizer,
    )


def build_preference_prompt(
    example: dict[str, Any],
    template: DefaultChatTemplate | None = None,
    *,
    template_backend: TemplateBackend = "default_chat_template",
    tokenizer: ChatTemplateTokenizer | None = None,
) -> str:
    """Render the prompt part of one preference record."""

    messages = [ChatMessage("user", str(example.get("prompt", "")).strip())]
    if template is not None:
        return template.format_messages(messages, add_generation_prompt=True)
    return format_messages(
        messages,
        template_backend=template_backend,
        tokenizer=tokenizer,
        add_generation_prompt=True,
    )


def build_eval_messages(eval_record: dict[str, Any]) -> list[ChatMessage]:
    """Build evaluation messages without leaking expected_behavior."""

    prompt = str(eval_record.get("prompt", "")).strip()
    constraints = format_constraints(eval_record.get("constraints", {}))
    if constraints:
        prompt = f"{prompt}\n\nOutput requirements:\n{constraints}"
    return [ChatMessage("user", prompt)]


def build_eval_prompt(
    eval_record: dict[str, Any],
    template: DefaultChatTemplate | None = None,
    *,
    template_backend: TemplateBackend = "default_chat_template",
    tokenizer: ChatTemplateTokenizer | None = None,
) -> str:
    """Render one evaluation record into a prompt string."""

    messages = build_eval_messages(eval_record)
    if template is not None:
        return template.format_messages(messages, add_generation_prompt=True)
    return format_messages(
        messages,
        template_backend=template_backend,
        tokenizer=tokenizer,
        add_generation_prompt=True,
    )


def format_constraints(constraints: dict[str, Any] | None) -> str:
    """Render explicit evaluation constraints."""

    if not constraints:
        return ""

    lines: list[str] = []
    required_format = constraints.get("required_format", "none")
    if required_format == "json":
        lines.append("- Return valid JSON only.")
    elif required_format == "bullets":
        lines.append("- Use a bullet list.")
    elif required_format == "short_answer":
        lines.append("- Give a short answer.")
    elif required_format == "free_text":
        lines.append("- Use plain free text.")

    max_words = constraints.get("max_words")
    if max_words is not None:
        lines.append(f"- Use at most {max_words} words.")

    must_include = constraints.get("must_include") or []
    if must_include:
        lines.append(f"- Must include: {', '.join(must_include)}.")

    must_not_include = constraints.get("must_not_include") or []
    if must_not_include:
        lines.append(f"- Must not include: {', '.join(must_not_include)}.")

    regex = constraints.get("regex")
    if regex:
        lines.append(f"- Must match regex: {regex}.")

    return "\n".join(lines)
