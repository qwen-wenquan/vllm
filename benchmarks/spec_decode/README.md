# Parsed-Draft Speculative Decoding Benchmarks

Benchmarks for the parsed-draft speculative decoder, which uses pre-existing text
(e.g., OCR-extracted from PDF via mupdf) as a zero-cost draft for speculative decoding.

See [INSTALL.md](INSTALL.md) for full environment setup instructions.

## Benchmarks

### `benchmark_parsed_draft.py` — Fidelity Check

Verifies that vLLM's LCS utilities produce **identical results** to the EAGLE reference
implementation. Runs both implementations side-by-side on the same pre-tokenized OCR
blocks and compares accepted tokens, verify steps, and speedup estimates.

No GPU required.

```bash
# Run with default settings (100 blocks, chunk sizes 16/50/200)
.venv/bin/python benchmarks/spec_decode/benchmark_parsed_draft.py

# Custom block count
.venv/bin/python benchmarks/spec_decode/benchmark_parsed_draft.py --n 500

# Custom chunk size
.venv/bin/python benchmarks/spec_decode/benchmark_parsed_draft.py --chunk-size 32
```

**Expected output**: `FIDELITY CHECK PASSED` — 0 mismatches across all configurations.

### `benchmark_parsed_draft_e2e.py` — End-to-End Benchmark

Loads PaddleOCR-VL in vLLM, processes real document images from GTX5k, and compares:

- **Baseline**: autoregressive (no `draft_text`)
- **With draft_text**: same request but with `draft_text` from mupdf OCR in `extra_args`

Measures wall-clock latency, throughput, and output fidelity (edit distance).

Requires GPU and the PaddleOCR-VL model.

```bash
# Required environment variables (when nvcc is not installed)
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_USE_DEEP_GEMM=0

# Run benchmark (50 blocks from 1 document)
.venv/bin/python benchmarks/spec_decode/benchmark_parsed_draft_e2e.py \
    --model /path/to/PaddleOCR-VL_finetune \
    --samples-dir /path/to/samples/GTX5k \
    --max-docs 1 --max-blocks 50

# With repetition detection enabled
.venv/bin/python benchmarks/spec_decode/benchmark_parsed_draft_e2e.py \
    --model /path/to/PaddleOCR-VL_finetune \
    --samples-dir /path/to/samples/GTX5k \
    --max-docs 1 --max-blocks 50 \
    --repetition-detection
```

#### Repetition Detection

The `--repetition-detection` flag enables vLLM's built-in N-gram repetition detection
via `RepetitionDetectionParams`. When the scheduler detects that the output token
sequence contains a repeated N-gram pattern, it stops generation early
(`FINISHED_REPETITION`).

This replaces the custom `ForceEosOnSimpleRepeatVLLM` logits processor used in
`eval_vllm.py`. The key difference:

| Feature | `ForceEosOnSimpleRepeatVLLM` (eval_vllm.py) | `RepetitionDetectionParams` (vLLM built-in) |
| --- | --- | --- |
| Where it runs | Logits processor (per-token, GPU-side) | Scheduler (per-step, CPU-side) |
| How it stops | Forces EOS logit to 0, others to -inf | Sets `FINISHED_REPETITION` status |
| Configuration | `max_simple_cycle`, `repeat_threshold` | `max_pattern_size`, `min_count` |
| Requires custom adapter | Yes (`AdapterLogitsProcessor`) | No |

Default parameters when `--repetition-detection` is enabled:

- `max_pattern_size=10` — detect repeating N-grams up to 10 tokens
- `min_count=5` — trigger after the pattern repeats 5 times

## Data Format

### E2E Benchmark

The E2E benchmark uses GTX5k-style document directories:

```text
samples/GTX5k/
  <doc_id>/
    document.pdf
    det_ocr_results.json
```

Each `det_ocr_results.json` contains page layout blocks with bounding boxes,
category labels, ground-truth text, and OCR-parsed text (`parsed_text`).

### Fidelity Benchmark

The fidelity benchmark uses pre-tokenized JSON from
`EAGLE/bench/sft_ocr_blocks_10k.tokenized.json`:

```json
{
  "blocks": [
    {
      "parsed_text": "OCR-extracted text",
      "pred_text": "model ground truth text",
      "parsed_ids": [token, ids, ...],
      "gt_ids": [token, ids, ...],
      "page": "0",
      "block_idx": 0
    }
  ]
}
```

## Results Summary

Results from 100 blocks on H100 NVL (PaddleOCR-VL 0.3B):

### Measured Latency

Pure model forward pass latency (HuggingFace, fresh KV cache clone, no DynamicCache
concat overhead). These numbers approximate what vLLM's paged KV cache would achieve.

| Operation | Latency | Per token |
| --- | --- | --- |
| Decode 1 token | 10.2 ms | 10.2 ms/tok |
| Verify chunk c=4 | 11.2 ms | 2.8 ms/tok |
| Verify chunk c=16 | 11.2 ms | 0.7 ms/tok |
| Verify chunk c=50 | 11.3 ms | 0.2 ms/tok |

For this small model, verify cost is ~1.1x a single decode step regardless of chunk
size — the forward pass is dominated by kernel launch overhead, not FLOPS. Larger
models would show more separation between decode and verify cost.

### Estimated Speedup

Using decode=10.2 ms/step, verify=11.2 ms/step (measured on H100 NVL):

| Strategy | Chunk Size | Verify Steps | Verify Time | AR Time | **Speedup** |
| --- | --- | --- | --- | --- | --- |
| stop_at_first | 16 | 2,444 | 27.4s | 64.5s | **2.4x** |
| stop_at_first | 50 | 1,729 | 19.4s | 64.5s | **3.3x** |
| stop_at_first | 200 | 1,440 | 16.1s | 64.5s | **4.0x** |
| hybrid (mr=3) | 16 | 1,051 | 11.8s | 64.5s | **5.5x** |
| hybrid (mr=3) | 50 | 955 | 10.7s | 64.5s | **6.0x** |
| hybrid (mr=3) | 200 | 966 | 10.8s | 64.5s | **6.0x** |

AR time = 6,322 tokens x 10.2 ms. Verify time = steps x 11.2 ms.

### Acceptance Rate Simulation

| Strategy | Chunk Size | Accept Rate | Verify Steps | Avg tok/step |
| --- | --- | --- | --- | --- |
| stop_at_first | 16 | 47.4% | 2,444 | 2.2 |
| stop_at_first | 50 | 30.4% | 1,729 | 2.1 |
| stop_at_first | 200 | 20.4% | 1,440 | 1.9 |
| hybrid (mr=3) | 16 | 60.3% | 1,051 | 5.6 |
| hybrid (mr=3) | 50 | 47.5% | 955 | 4.9 |
| hybrid (mr=3) | 200 | 42.5% | 966 | 4.4 |

### Fidelity Check

vLLM's LCS implementation produces **identical results** to the EAGLE reference across
all 6 configurations (0 mismatches on 100 blocks).
