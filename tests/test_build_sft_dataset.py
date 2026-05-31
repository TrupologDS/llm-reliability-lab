from __future__ import annotations

from scripts.build_sft_dataset import (
    build_sft_examples,
    deduplicate_by_instruction,
    deterministic_sample,
    dolly_record_to_sft,
    extract_first_user_assistant_pair,
)


def test_extracts_first_user_assistant_pair_from_ultrachat_messages() -> None:
    messages = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Summarize the release note."},
        {"role": "assistant", "content": "The release improves the API."},
        {"role": "user", "content": "Add details."},
        {"role": "assistant", "content": "It also updates docs."},
    ]

    pair = extract_first_user_assistant_pair(messages)

    assert pair == {
        "system": "Be concise.",
        "instruction": "Summarize the release note.",
        "response": "The release improves the API.",
    }


def test_skips_malformed_ultrachat_conversations() -> None:
    assert extract_first_user_assistant_pair([{"role": "user", "content": "Only a user."}]) is None
    assert extract_first_user_assistant_pair([{"role": "assistant", "content": "No user first."}]) is None
    assert extract_first_user_assistant_pair("not messages") is None


def test_deduplicates_by_normalized_instruction() -> None:
    examples = [
        {"instruction": "Explain CUDA memory.", "response": "A" * 30},
        {"instruction": " explain   cuda MEMORY. ", "response": "B" * 30},
        {"instruction": "Explain tokenizer padding.", "response": "C" * 30},
    ]

    deduped, duplicates = deduplicate_by_instruction(examples)

    assert duplicates == 1
    assert [example["instruction"] for example in deduped] == ["Explain CUDA memory.", "Explain tokenizer padding."]


def test_sampling_is_deterministic_for_seed() -> None:
    examples = [{"instruction": f"prompt {index}", "response": "A" * 30} for index in range(10)]

    first = deterministic_sample(examples, num_examples=4, seed=17)
    second = deterministic_sample(examples, num_examples=4, seed=17)
    different = deterministic_sample(examples, num_examples=4, seed=18)

    assert first == second
    assert first != different


def test_maps_dolly_record_to_sft_schema() -> None:
    record = {
        "instruction": "Write a release summary.",
        "context": "The release fixed three bugs.",
        "response": "The release fixed three bugs.",
        "category": "summarization",
    }

    mapped = dolly_record_to_sft(record, source_value="databricks/databricks-dolly-15k:train")

    assert mapped == {
        "id": "",
        "system": "",
        "instruction": "Write a release summary.",
        "input": "The release fixed three bugs.",
        "response": "The release fixed three bugs.",
        "source": "databricks/databricks-dolly-15k:train:summarization",
    }


def test_build_sft_examples_filters_dedupes_samples_and_assigns_ids() -> None:
    raw_records = [
        {
            "messages": [
                {"role": "user", "content": "Explain LoRA."},
                {"role": "assistant", "content": "LoRA trains low-rank adapter matrices."},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": " explain   lora. "},
                {"role": "assistant", "content": "Duplicate prompt should be removed."},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Short answer?"},
                {"role": "assistant", "content": "Too short"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Explain eval leakage."},
                {"role": "assistant", "content": "Eval leakage happens when evaluation prompts enter training data."},
            ]
        },
    ]

    examples, stats = build_sft_examples(
        raw_records,
        source="HuggingFaceH4/ultrachat_200k",
        split="train_sft",
        source_name=None,
        num_examples=2,
        seed=42,
        max_prompt_chars=8000,
        max_response_chars=8000,
        min_response_chars=20,
    )

    assert stats.loaded == 4
    assert stats.extracted == 4
    assert stats.filtered == 3
    assert stats.duplicates_removed == 1
    assert stats.selected == 2
    assert {example["id"] for example in examples} == {"ultrachat_000001", "ultrachat_000002"}
    assert all(example["source"] == "HuggingFaceH4/ultrachat_200k:train_sft" for example in examples)
