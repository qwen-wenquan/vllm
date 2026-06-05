# Parsed-Draft Speculative Decoding Benchmarks

Benchmarks for the parsed-draft speculative decoder, which uses pre-existing text
(e.g., OCR-extracted from PDF via mupdf) as a zero-cost draft for speculative decoding.

## Benchmarks

### `bench_parsed_draft_fidelity.py` — Fidelity Check

Verifies that vLLM's LCS utilities produce **identical results** to the EAGLE reference
implementation. Runs both implementations side-by-side on the same pre-tokenized OCR
blocks and compares accepted tokens, verify steps, and speedup estimates.

No GPU required.

```bash
# Run with default settings (100 blocks, chunk sizes 16/50/200)
.venv/bin/python bench/bench_parsed_draft_fidelity.py

# Custom block count
.venv/bin/python bench/bench_parsed_draft_fidelity.py --n 500

# Custom chunk size
.venv/bin/python bench/bench_parsed_draft_fidelity.py --chunk-size 32
```

**Expected output**: `FIDELITY CHECK PASSED` — 0 mismatches across all configurations.

### `bench_e2e_parsed_draft.py` — End-to-End Benchmark

Loads PaddleOCR-VL in vLLM, measures real autoregressive decode latency, then simulates
parsed-draft speculative decoding on the same blocks to estimate speedup.

Requires GPU and the PaddleOCR-VL model.

```bash
# Environment setup (required for this machine's CUDA config)
export CUDA_HOME=/tmp/cuda_home
export PATH=".venv/bin:$CUDA_HOME/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0

# Run benchmark (100 blocks)
.venv/bin/python bench/bench_e2e_parsed_draft.py --n 100

# Custom chunk size
.venv/bin/python bench/bench_e2e_parsed_draft.py --n 100 --chunk-size 32

# Different model path
.venv/bin/python bench/bench_e2e_parsed_draft.py \
    --model /path/to/model \
    --data /path/to/tokenized_blocks.json \
    --n 200
```

## Data Format

Both benchmarks use pre-tokenized JSON from `EAGLE/bench/sft_ocr_blocks_10k.tokenized.json`:

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
|---|---|---|
| Decode 1 token | 10.2 ms | 10.2 ms/tok |
| Verify chunk c=4 | 11.2 ms | 2.8 ms/tok |
| Verify chunk c=16 | 11.2 ms | 0.7 ms/tok |
| Verify chunk c=50 | 11.3 ms | 0.2 ms/tok |

For this small model, verify cost is ~1.1× a single decode step regardless of chunk
size — the forward pass is dominated by kernel launch overhead, not FLOPS. Larger
models would show more separation between decode and verify cost.

### Estimated Speedup

Using decode=10.2 ms/step, verify=11.2 ms/step (measured on H100 NVL):

| Strategy | Chunk Size | Verify Steps | Verify Time | AR Time | **Speedup** |
|---|---|---|---|---|---|
| stop_at_first | 16 | 2,444 | 27.4s | 64.5s | **2.4×** |
| stop_at_first | 50 | 1,729 | 19.4s | 64.5s | **3.3×** |
| stop_at_first | 200 | 1,440 | 16.1s | 64.5s | **4.0×** |
| hybrid (mr=3) | 16 | 1,051 | 11.8s | 64.5s | **5.5×** |
| hybrid (mr=3) | 50 | 955 | 10.7s | 64.5s | **6.0×** |
| hybrid (mr=3) | 200 | 966 | 10.8s | 64.5s | **6.0×** |

AR time = 6,322 tokens × 10.2 ms. Verify time = steps × 11.2 ms.

### Acceptance Rate Simulation

| Strategy | Chunk Size | Accept Rate | Verify Steps | Avg tok/step |
|---|---|---|---|---|
| stop_at_first | 16 | 47.4% | 2,444 | 2.2 |
| stop_at_first | 50 | 30.4% | 1,729 | 2.1 |
| stop_at_first | 200 | 20.4% | 1,440 | 1.9 |
| hybrid (mr=3) | 16 | 60.3% | 1,051 | 5.6 |
| hybrid (mr=3) | 50 | 47.5% | 955 | 4.9 |
| hybrid (mr=3) | 200 | 42.5% | 966 | 4.4 |

### Fidelity Check

vLLM's LCS implementation produces **identical results** to the EAGLE reference across
all 6 configurations (0 mismatches on 100 blocks).
