# AGENTS.md

## Cursor Cloud specific instructions

### Project Overview

LightLLM is a Python-based LLM inference and serving framework. It provides an OpenAI-compatible HTTP API server for serving large language models with features like tensor parallelism, dynamic prompt caching, and multi-modal support.

### Environment Constraints

- **NVIDIA GPU required for full operation**: The inference server (`python3 -m lightllm.server.api_server`) and GPU-dependent unit tests require CUDA-capable GPUs. Without a GPU, only CPU-based unit tests and lint checks can run.
- **CPU-only mode**: When no GPU is present, importing model modules will fail at the Triton autotuner level (`get_current_device_name()` returns `None`). Core server logic (request management, radix cache, sampling params, shared memory) still works.
- **Python 3.12 compatibility**: The pinned `requirements.txt` has some packages incompatible with Python 3.12 (numpy==1.25.1, uvloop==0.17.0). Use `numpy>=1.26` and `uvloop>=0.19` instead.

### Running Lint

```bash
# Black (formatter check)
black --check --line-length=120 lightllm/

# Flake8 (linter)
flake8 --max-line-length=120 --ignore='TYP001,E722,C901,E203,E266,E402,E302,E241,E902,E731,F403,E701,F405,F401,W292,W293,W503,W606,E231,F541' lightllm/
```

### Running Tests

```bash
# CPU-compatible unit tests (no GPU needed)
python3 -m pytest unit_tests/server/ -v

# All unit tests (requires CUDA GPU)
python3 -m pytest unit_tests/ -v
```

### Running the Server (requires GPU + model weights)

```bash
python3 -m lightllm.server.api_server --model_dir /path/to/model --tp 1 --port 8000
```

### Key Gotchas

1. The `sgl-kernel`, `flashinfer-python`, `cuda_bindings`, `cupy-cuda12x`, and `nixl` packages from `requirements.txt` are GPU-only and will not install without CUDA. These are optional for development/testing of CPU-based logic.
2. Warnings about missing `sgl_kernel`, `lightllm_kernel`, `vllm`, and `deep_ep` during import are harmless - these are optional acceleration libraries.
3. The pre-commit config uses `black==21.12b0` (older) while `requirements.txt` pins `black==23.12.0`. Use the newer version for development.
4. Shared memory warnings (`resource_tracker: leaked shared_memory objects`) during tests are benign - caused by test processes not explicitly cleaning up `/dev/shm` segments.
