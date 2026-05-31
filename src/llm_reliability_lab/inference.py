"""Inference backends for evaluation."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Literal

from llm_reliability_lab.data import write_jsonl
from llm_reliability_lab.metrics import expected_refusal
from llm_reliability_lab.prompts import TemplateBackend, build_eval_prompt

InferenceBackend = Literal["mock", "hf", "vllm"]


def generate_for_eval_records(
    eval_records: list[dict[str, Any]],
    *,
    model_name_or_path: str,
    backend: InferenceBackend = "mock",
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    template_backend: TemplateBackend = "default_chat_template",
    adapter_base_model_name_or_path: str | None = None,
) -> list[dict[str, Any]]:
    """Generate responses for evaluation records."""

    if backend == "mock":
        return [_generate_mock(record, model_name_or_path=model_name_or_path) for record in eval_records]
    if backend == "hf":
        return _generate_hf(
            eval_records,
            model_name_or_path=model_name_or_path,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            template_backend=template_backend,
            adapter_base_model_name_or_path=adapter_base_model_name_or_path,
        )
    if backend == "vllm":
        return _generate_vllm(
            eval_records,
            model_name_or_path=model_name_or_path,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
    msg = f"Unsupported inference backend: {backend}"
    raise ValueError(msg)


def save_generations_jsonl(generations: list[dict[str, Any]], path: str | Path) -> Path:
    """Save generation records as JSONL."""

    return write_jsonl(generations, path)


def _generate_mock(eval_record: dict[str, Any], *, model_name_or_path: str) -> dict[str, Any]:
    start = time.perf_counter()
    response = _mock_response(eval_record)
    latency = time.perf_counter() - start
    return {
        "id": eval_record["id"],
        "model": f"{model_name_or_path}::mock-deterministic-backend-not-a-model",
        "prompt": build_eval_prompt(eval_record),
        "response": response,
        "metadata": {
            "latency_seconds": latency,
            "num_input_tokens": None,
            "num_output_tokens": None,
            "backend": "mock",
        },
    }


def _mock_response(eval_record: dict[str, Any]) -> str:
    """Deterministic synthetic response for CPU-only pipeline checks."""

    constraints = eval_record.get("constraints") or {}
    must_include = constraints.get("must_include") or []
    required_format = constraints.get("required_format", "none")
    prompt = str(eval_record.get("prompt", "")).lower()
    if expected_refusal(eval_record):
        if "password" in prompt:
            return "I cannot help reveal passwords or private credentials."
        return "I cannot provide secrets. Revoke exposed credentials and use a safe secret manager."

    if "rotate" in prompt and "api key" in prompt:
        return "Revoke the exposed key, create a new key, update services, and audit recent usage."

    if required_format == "json":
        return '{"model":"sample","status":"ok"}'

    if required_format == "bullets":
        if {"config", "seed"} <= set(must_include):
            return "- Save the config.\n- Fix the seed."
        return "- Metrics report\n- Regression report\n- Config report"

    if "not a benchmark" in prompt:
        return "No, it is not a benchmark."
    if "final metrics" in prompt:
        return "No. The context says final metrics remain TODO until real experiments run."
    if "private customer data" in prompt:
        return "No, it uses synthetic examples."
    if "espanol" in prompt:
        return "La evaluacion reproducible usa los mismos datos, configs y metricas."
    if "francais" in prompt:
        return "Un risque est de renforcer une mauvaise preference."
    if "not spanish" in prompt:
        return "A data card documents a dataset, preprocessing, limits, and licensing."
    if "18 of 20" in prompt:
        return "Yes, compliance dropped by 15 points."
    if "no baseline" in prompt:
        return "No, a baseline is required."
    if "exactly five words" in prompt:
        return "Consistent checks reveal model regressions."
    if "ci config" in prompt:
        return "ci.yml"
    if "lowercase text" in prompt:
        return "configs control reproducibility"
    if "project name" in prompt:
        return "LLM Reliability Lab"
    if "lora" in prompt:
        return "Parameter-efficient fine-tuning."

    if must_include:
        return "This response covers " + ", ".join(must_include) + "."
    return "The answer follows the requested instruction concisely."


def _generate_hf(
    eval_records: list[dict[str, Any]],
    *,
    model_name_or_path: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    template_backend: TemplateBackend,
    adapter_base_model_name_or_path: str | None = None,
) -> list[dict[str, Any]]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        msg = "Install the train extra to use the Hugging Face inference backend."
        raise RuntimeError(msg) from exc

    is_adapter = is_peft_adapter_path(model_name_or_path)
    adapter_config = read_peft_adapter_config(model_name_or_path) if is_adapter else {}
    base_model_name_or_path = (
        adapter_base_model_name_or_path or adapter_config.get("base_model_name_or_path") or model_name_or_path
    )
    tokenizer_name_or_path = base_model_name_or_path if is_adapter else model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        device_map="auto",
        torch_dtype="auto",
    )
    if is_adapter:
        try:
            from peft import PeftModel
        except ImportError as exc:
            msg = "Install the train extra to evaluate a PEFT/LoRA adapter path."
            raise RuntimeError(msg) from exc
        model = PeftModel.from_pretrained(model, model_name_or_path)
    model.eval()

    outputs: list[dict[str, Any]] = []
    for eval_record in eval_records:
        prompt = build_eval_prompt(
            eval_record,
            template_backend=template_backend,
            tokenizer=tokenizer,
        )
        start = time.perf_counter()
        encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = generated[0][encoded["input_ids"].shape[-1] :]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        latency = time.perf_counter() - start
        outputs.append(
            {
                "id": eval_record["id"],
                "model": model_name_or_path,
                "prompt": prompt,
                "response": response,
                "metadata": {
                    "latency_seconds": latency,
                    "num_input_tokens": int(encoded["input_ids"].shape[-1]),
                    "num_output_tokens": int(new_tokens.shape[-1]),
                    "backend": "hf",
                    "adapter_base_model": base_model_name_or_path if is_adapter else None,
                    "peft_adapter": model_name_or_path if is_adapter else None,
                    "template_backend": template_backend,
                },
            }
        )
    return outputs


def is_peft_adapter_path(model_name_or_path: str | Path) -> bool:
    """Return true if a local path looks like a PEFT adapter directory."""

    adapter_path = Path(model_name_or_path)
    return adapter_path.exists() and (adapter_path / "adapter_config.json").exists()


def read_peft_adapter_config(model_name_or_path: str | Path) -> dict[str, Any]:
    """Read a PEFT adapter_config.json file without importing PEFT."""

    adapter_config_path = Path(model_name_or_path) / "adapter_config.json"
    with adapter_config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    if not isinstance(config, dict):
        msg = f"Expected JSON object in {adapter_config_path}"
        raise ValueError(msg)
    return config


def _generate_vllm(
    eval_records: list[dict[str, Any]],
    *,
    model_name_or_path: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[dict[str, Any]]:
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        msg = "Install the serving extra to use the vLLM inference backend."
        raise RuntimeError(msg) from exc

    prompts = [build_eval_prompt(record) for record in eval_records]
    sampling = SamplingParams(max_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
    llm = LLM(model=model_name_or_path)
    start = time.perf_counter()
    raw_outputs = llm.generate(prompts, sampling)
    total_latency = time.perf_counter() - start

    generations: list[dict[str, Any]] = []
    for eval_record, prompt, output in zip(eval_records, prompts, raw_outputs, strict=True):
        response = output.outputs[0].text.strip()
        generations.append(
            {
                "id": eval_record["id"],
                "model": model_name_or_path,
                "prompt": prompt,
                "response": response,
                "metadata": {
                    "latency_seconds": total_latency / max(len(eval_records), 1),
                    "num_input_tokens": None,
                    "num_output_tokens": len(output.outputs[0].token_ids) if output.outputs[0].token_ids else None,
                    "backend": "vllm",
                },
            }
        )
    return generations
