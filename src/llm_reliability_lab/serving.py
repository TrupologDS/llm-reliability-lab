"""Optional vLLM serving helpers."""

from __future__ import annotations

from llm_reliability_lab.config import ServingConfig


def build_vllm_command(config: ServingConfig) -> list[str]:
    """Build a vLLM OpenAI-compatible server command."""

    return [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        config.model_path,
        "--host",
        config.host,
        "--port",
        str(config.port),
        "--tensor-parallel-size",
        str(config.tensor_parallel_size),
        "--max-model-len",
        str(config.max_model_len),
        "--gpu-memory-utilization",
        str(config.gpu_memory_utilization),
    ]


def render_vllm_instructions(config: ServingConfig) -> str:
    """Render serving instructions without starting a server."""

    command = " ".join(build_vllm_command(config))
    return f"""# vLLM Serving

vLLM serving is optional. It requires compatible GPU hardware and the serving extra:

```bash
pip install -e ".[serving]"
```

Start an OpenAI-compatible server:

```bash
{command}
```

Example client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://{config.host}:{config.port}/v1", api_key="EMPTY")
response = client.chat.completions.create(
    model="{config.model_path}",
    messages=[{{"role": "user", "content": "Summarize this evaluation result."}}],
    temperature=0,
)
print(response.choices[0].message.content)
```
"""
