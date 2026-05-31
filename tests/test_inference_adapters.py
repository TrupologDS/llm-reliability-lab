from __future__ import annotations

import json

from llm_reliability_lab.inference import is_peft_adapter_path, read_peft_adapter_config


def test_detects_peft_adapter_path(tmp_path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct"}),
        encoding="utf-8",
    )

    assert is_peft_adapter_path(adapter_dir)
    assert read_peft_adapter_config(adapter_dir)["base_model_name_or_path"] == "Qwen/Qwen2.5-0.5B-Instruct"


def test_non_adapter_path(tmp_path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    assert not is_peft_adapter_path(model_dir)
