# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This directory implements speculative decoding proposers for the vLLM V1 engine. Each proposer generates draft tokens that the target model then verifies in a single forward pass, trading extra draft computation for higher throughput.

## Proposer Taxonomy

There are two families of proposers with very different integration requirements:

### Model-based proposers (subclass `SpecDecodeBaseProposer`)

These run a neural network to produce drafts. The base class in `llm_base_proposer.py` provides the core multi-step autoregressive loop, CUDA graph dispatch, attention metadata construction, weight sharing, and buffer management.

| File | Method string | Key trait |
|------|--------------|-----------|
| `eagle.py` | `"eagle"` | Minimal subclass — the base class handles everything. Passes `hidden_states` from target model. |
| `draft_model.py` | `"draft_model"` | Standalone smaller model (no hidden state sharing). Overrides `_create_draft_vllm_config` and `_get_model`. |
| `dflash.py` | `"dflash"` | Cross-attention: context K/V from target hidden states, Q from query embeddings. Parallel drafting only. Overrides `set_inputs_first_pass`, `build_model_inputs_first_pass`, `dummy_run`. |
| `gemma4.py` | `"gemma4_mtp"` | Draft layers share KV cache with target model via cross-model KV sharing. Multiple KV cache groups. `constant_draft_positions = True`. |
| `step3p5.py` | `"step3p5_mtp"` | Extends `EagleProposer`. Per-layer draft-step selection via `spec_step_idx`. Multiple KV cache groups with per-group block tables. |

### Non-model proposers (standalone classes)

These do not subclass `SpecDecodeBaseProposer` and have simpler, self-contained interfaces.

| File | Method string | Key trait |
|------|--------------|-----------|
| `ngram_proposer.py` | `"ngram"` | CPU-based n-gram matching with numba JIT. No model to load. |
| `ngram_proposer_gpu.py` | `"ngram_gpu"` | GPU-accelerated n-gram matching using torch.compile. |
| `suffix_decoding.py` | `"suffix"` | Uses arctic_inference library for suffix-tree-based speculation. No model. |
| `medusa.py` | `"medusa"` | Multiple prediction heads on target hidden states. All draft tokens in one pass (parallel). |
| `extract_hidden_states.py` | `"extract_hidden_states"` | Not a true speculator — caches hidden states in KV cache for KV transfer. Always "accepts". |

### Custom class proposer

`custom_class_proposer.py` loads a user-provided proposer via `speculative_config.model` (a dotted Python path). The class must accept `VllmConfig` and expose a `propose()` method.

## How to Add a New Proposer

### 1. Decide which family

- If your proposer runs a draft model and needs attention/KV cache: subclass `SpecDecodeBaseProposer`.
- If your proposer is stateless or doesn't use a model: write a standalone class.

### 2. Add the method string

In `vllm/config/speculative.py`:
- Add your method name to the `SpeculativeMethod` literal type (and to the appropriate sub-literal if it's an EAGLE/MTP variant).
- If your method needs auto-detection from model config (rather than explicit `method=` from the user), add detection logic in `SpeculativeConfig.__post_init__`.

### 3. Create the proposer file

For model-based proposers, the minimal subclass looks like `eagle.py`:

```python
class MyProposer(SpecDecodeBaseProposer):
    def __init__(self, vllm_config, device, runner=None):
        super().__init__(
            vllm_config, device,
            pass_hidden_states_to_model=True,  # or False for draft_model-style
            runner=runner,
        )
```

Override methods only as needed. The most commonly overridden methods are:
- `_create_draft_vllm_config()` — customize config for the draft model
- `_get_model()` — customize model loading
- `_maybe_share_embeddings()` / `_maybe_share_lm_head()` — control weight sharing
- `set_inputs_first_pass()` — reshape inputs before first draft forward pass
- `build_model_inputs_first_pass()` — build model kwargs for first pass
- `build_per_group_and_layer_attn_metadata()` — custom attention metadata per KV cache group
- `dummy_run()` — warmup/profiling run
- `validate_same_kv_cache_group()` — override if draft layers span multiple KV cache groups
- `initialize_attn_backend()` — override if you need per-layer attention group setup

For non-model proposers, implement at minimum:
- `__init__(self, vllm_config: VllmConfig, ...)` 
- `propose(...)` — returns draft token IDs (signature varies by type)
- `load_model(self, *args, **kwargs)` — can be a no-op

### 4. Wire it into the model runner

In `vllm/v1/worker/gpu_model_runner.py`:
- Import your proposer class.
- Add it to the `self.drafter` type union (the big `|` chain around line 541).
- Add an `elif` branch in the method dispatch block (around line 553) to instantiate it.

### 5. Write tests

Tests live in `tests/v1/spec_decode/` for unit tests and `tests/v1/e2e/spec_decode/` for end-to-end tests. Key test files:
- `test_speculators_correctness.py` — output correctness against non-speculative baseline
- `test_eagle.py` — EAGLE-specific unit tests
- `test_ngram.py` — n-gram proposer tests

## Key Base Class Concepts (`SpecDecodeBaseProposer`)

**The `propose()` method** drives the drafting loop:
1. `set_inputs_first_pass()` — prepares input_ids, positions, hidden_states for the first draft forward
2. `build_per_group_and_layer_attn_metadata()` — builds attention metadata from `CommonAttentionMetadata`
3. First forward pass through `self.model`
4. Sample first draft token
5. For `num_speculative_tokens > 1` (and non-parallel): loop, updating positions and slot mappings per step

**`pass_hidden_states_to_model`**: When `True` (EAGLE/MTP), the target model's hidden states are passed as input alongside token IDs. When `False` (draft_model), only token IDs are used.

**`parallel_drafting`**: When `True`, all speculative tokens are produced in a single forward pass (using mask tokens at future positions) rather than autoregressively.

**`needs_extra_input_slots`**: Triggered by draft model or parallel drafting modes. Expands the input tensor to include extra slots per request for padded drafting.

**`constant_draft_positions`**: When `True` (Gemma4 MTP), all draft steps reuse the same position. The positions buffer and attention metadata are built once and reused.

## Supporting Files

- `metadata.py` — `SpecDecodeMetadata` dataclass: draft token IDs, cumulative counts, logit indices for verification.
- `metrics.py` — `SpecDecodingStats`, `SpecDecodingLogging`, `SpecDecodingProm`: per-step stats aggregation, console logging, and Prometheus metrics.
- `utils.py` — Triton kernels for slot mapping updates, input expansion, and position management during draft steps.
