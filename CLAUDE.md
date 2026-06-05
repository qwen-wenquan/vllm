# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@AGENTS.md

## Build & Development Commands

**Never use system `python3` or bare `pip`.** All Python commands go through `uv` and `.venv/bin/python`.

```bash
# Environment setup
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements/lint.txt
pre-commit install

# Install (Python-only changes, uses precompiled binaries)
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto

# Install (with C/C++ changes, compiles from source)
uv pip install -e . --torch-backend=auto

# Test dependencies
uv pip install -r requirements/test/cuda.in   # resolves for current platform

# Run a single test
.venv/bin/python -m pytest tests/path/to/test_file.py -v

# Lint (staged files)
pre-commit run

# Lint (all files)
pre-commit run --all-files

# Specific linter
pre-commit run ruff-check --all-files

# Type checking (as CI runs it)
pre-commit run mypy-3.10 --all-files --hook-stage manual
```

Line length limit is **88 characters**.

## Architecture Overview

vLLM is an LLM inference and serving engine. The codebase has three major layers: **entrypoints** (user-facing APIs), **engine** (orchestration), and **model execution** (GPU computation).

### Entrypoints (`vllm/entrypoints/`)

- **`llm.py`** — Offline batch inference API (`LLM` class). The main programmatic entry point.
- **`openai/`** — OpenAI-compatible HTTP API server (production use). The primary serving entrypoint.
- **`anthropic/`** — Anthropic Messages API compatibility layer.
- **`grpc_server.py`** — gRPC serving interface.
- **`cli/`** — `vllm` CLI tool (maps to `vllm serve`, `vllm bench`, etc.).
- **`api_server.py`** — Demo-only simple API server. Do not modify; change `openai/api_server.py` instead.

### V1 Engine (`vllm/v1/`) — Active Architecture

The V1 engine is the current active architecture. The legacy engine in `vllm/engine/` exists for backward compatibility.

**Core engine loop** (`vllm/v1/engine/`):
- **`async_llm.py`** — `AsyncLLM`: The async engine client used by the API server. Manages request lifecycle and output streaming.
- **`core.py`** — `EngineCore`: Runs in a separate process. Houses the scheduler, KV cache management, and drives the model execution loop. Communicates via ZMQ (msgspec serialization).
- **`core_client.py`** — `EngineCoreClient`: Client-side proxy that communicates with EngineCore over ZMQ.
- **`coordinator.py`** — `DPCoordinator`: Intermediary for data-parallel deployments (DP>1), handling load-balancing stats between engine ranks and API server frontends.
- **`input_processor.py` / `output_processor.py`** — Tokenization and detokenization, running in the API server process to offload work from the engine core.

**Scheduling** (`vllm/v1/core/sched/`):
- `scheduler.py` — Request scheduling with continuous batching and chunked prefill.
- KV cache management in `vllm/v1/core/` (block pool, KV cache manager, prefix caching via block hashing).

**Workers** (`vllm/v1/worker/`):
- `gpu_worker.py` / `gpu_model_runner.py` — GPU-side execution. The model runner prepares input tensors, runs the model, and handles CUDA graph capture/replay.
- `cpu_worker.py` — CPU execution backend.
- Platform-specific runners: `xpu_model_runner.py`, `tpu_input_batch.py`.

**Executors** (`vllm/v1/executor/`):
- `uniproc_executor.py` — Single-process execution.
- `multiproc_executor.py` — Multi-process (tensor parallelism within a node).
- `ray_executor.py` — Ray-based distributed execution (multi-node).

### Model Execution Layer

**Model implementations** (`vllm/model_executor/models/`):
- ~290 model files, each implementing a HuggingFace architecture for optimized vLLM inference.
- `registry.py` — Maps HuggingFace architecture strings (e.g., `"LlamaForCausalLM"`) to vLLM model classes. When adding a new model, also update `tests/models/registry.py`.
- Models implement marker interfaces from `interfaces.py`: `SupportsLoRA`, `SupportsMultiModal`, `SupportsPP`, etc.

**Reusable layers** (`vllm/model_executor/layers/`):
- `linear.py` — Parallelized linear layers (column/row parallel, merged QKV).
- `attention/` — Attention layer abstraction over multiple backends.
- `rotary_embedding/` — RoPE and its variants.
- `layernorm.py` — RMSNorm, LayerNorm with fused kernels.
- `fused_moe/` — Mixture-of-Experts kernels and routing (major subsystem with its own modular kernel framework).
- `quantization/` — Quantization methods (FP8, GPTQ, AWQ, INT8, bitsandbytes, etc.).
- `vocab_parallel_embedding.py` — Vocabulary-parallel embeddings.
- `logits_processor.py` — Post-model logits processing.

**Model loading** (`vllm/model_executor/model_loader/`):
- `default_loader.py` — Standard HuggingFace weight loading.
- Specialized loaders: `gguf_loader.py`, `bitsandbytes_loader.py`, `sharded_state_loader.py`, `tensorizer_loader.py`.

### Attention System (`vllm/v1/attention/`)

- `backends/` — Backend implementations: FlashAttention, FlashInfer, Triton, ROCm, CPU, and others.
- `selector.py` — Auto-selects the best attention backend for the current platform.
- Uses PagedAttention for KV cache memory management.

### Configuration (`vllm/config/`)

Configs are split into focused modules: `model.py`, `cache.py`, `parallel.py`, `scheduler.py`, `speculative.py`, etc. `VllmConfig` (in `vllm.py`) is the top-level aggregate config passed throughout the engine.

### Distributed (`vllm/distributed/`)

- `parallel_state.py` — Global process group state for tensor/pipeline/data/expert parallelism.
- `kv_transfer/` — Disaggregated prefill/decode with KV cache transfer between nodes.
- `elastic_ep/` — Elastic expert parallelism.

### Other Subsystems

- **Speculative decoding** (`vllm/v1/spec_decode/`) — n-gram, EAGLE, DFlash, Medusa, suffix decoding.
- **Structured output** (`vllm/v1/structured_output/`) — Constrained generation via xgrammar, outlines, guidance.
- **LoRA** (`vllm/lora/`) — Multi-LoRA adapter serving.
- **Multimodal** (`vllm/multimodal/`) — Image, audio, video input processing and encoder budget management.
- **Tool/reasoning parsers** (`vllm/tool_parsers/`, `vllm/reasoning/`) — Streaming tool call and chain-of-thought parsing.
- **Platforms** (`vllm/platforms/`) — Hardware abstraction: CUDA, ROCm, CPU, TPU, XPU.
- **Compilation** (`vllm/compilation/`) — torch.compile integration, CUDA graph management, piecewise compilation.
- **Kernels** — C++/CUDA in `csrc/`, Python/Triton in `vllm/kernels/`.
- **Rust components** (`rust/`) — Chat templating, tool parsing, reasoning parsing, tokenization, metrics, and a Rust engine-core client.

### Key Design Patterns

- **ZMQ + msgspec**: The engine core runs in a separate process and communicates with the API server via ZMQ sockets, using msgspec for fast serialization.
- **Platform abstraction**: `vllm/platforms/` provides `current_platform` for hardware-specific behavior without scattered if/else checks.
- **Environment variables**: All env vars are declared in `vllm/envs.py` with type annotations. Access via `vllm.envs.VLLM_*`.
- **Model registry**: Models are registered by HuggingFace architecture string in `vllm/model_executor/models/registry.py` and lazily imported.
