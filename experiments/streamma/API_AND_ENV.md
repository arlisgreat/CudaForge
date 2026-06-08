# API and Environment Notes

Do not commit API keys. Put secrets in the active shell environment only.

## Python/CUDA Environment

Known usable Python:

```bash
/data/workspace/airulan/conda_envs/kernelbench_py310_cu124/bin/python
```

Known env setup files:

```bash
source /data1/workspace/airulan/env124.sh
# or
source /data1/workspace/airulan/env130.sh
```

Prefer `env124.sh` first because the Python path name indicates a CUDA 12.4
KernelBench environment. Use `env130.sh` only if a CUDA 13.0 dependency is
explicitly required.

## Cloud API

Use environment variables, not checked-in config:

```bash
export OPENAI_API_KEY="<set in shell>"
export OPENAI_BASE_URL="https://aigc.x-see.cn/v1"
# alternate:
export OPENAI_BASE_URL="https://open.xiaojingai.com"
```

The current CudaForge `agents/query_server.py` reads `OPENAI_API_KEY` but does
not yet read `OPENAI_BASE_URL` for `server_type=openai`. A communication-layer
implementation should add configurable base URL support without hardcoding any
secret.

## Local Model Candidates

Likely local candidates under `/data1/hf_models`:

- `/data1/hf_models/DeepSeek-Coder-V2-Lite-Instruct`
- `/data1/hf_models/Qwen2.5-Coder-7B-Instruct`
- `/data1/hf_models/Qwen3-Coder-30B-A3B-Instruct`
- `/data1/hf_models/Qwen3-Coder-Next`

Use the same model, temperature, max tokens, and sampling settings across P0-P3
for a controlled comparison.
