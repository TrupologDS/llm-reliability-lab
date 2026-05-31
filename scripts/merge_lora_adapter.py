"""Merge a PEFT/LoRA adapter into its base model for standalone evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from llm_reliability_lab.inference import read_peft_adapter_config
from llm_reliability_lab.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default=None, help="Base model name/path. Defaults to adapter_config.json.")
    parser.add_argument("--adapter", required=True, help="PEFT adapter directory containing adapter_config.json.")
    parser.add_argument("--output-dir", required=True, help="Directory for merged full model.")
    parser.add_argument("--torch-dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter_path = Path(args.adapter)
    adapter_config = read_peft_adapter_config(adapter_path)
    base_model = args.base_model or adapter_config.get("base_model_name_or_path")
    if not base_model:
        msg = "Base model was not provided and adapter_config.json has no base_model_name_or_path."
        raise ValueError(msg)

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        msg = "Install training dependencies with: pip install -e '.[train]'"
        raise RuntimeError(msg) from exc

    dtype = dtype_from_name(args.torch_dtype, torch)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    model = PeftModel.from_pretrained(model, str(adapter_path))
    merged = model.merge_and_unload()

    output_dir = ensure_dir(args.output_dir)
    merged.save_pretrained(output_dir, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=args.trust_remote_code)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved merged model to {output_dir}")


def dtype_from_name(name: str, torch_module):
    """Resolve a dtype name for transformers loading."""

    if name == "auto":
        return "auto"
    return {
        "float16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "float32": torch_module.float32,
    }[name]


if __name__ == "__main__":
    main()
