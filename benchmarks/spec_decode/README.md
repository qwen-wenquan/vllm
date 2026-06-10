# Parsed-Draft Speculative Decoding Benchmarks

Benchmarks for the parsed-draft speculative decoder, which uses pre-existing text
(e.g., OCR-extracted from PDF via mupdf) as a zero-cost draft for speculative decoding.

See [INSTALL.md](INSTALL.md) for full environment setup instructions.

## Benchmarks

### `benchmark_parsed_draft.py` — Fidelity Check

Verifies that vLLM's LCS utilities produce **identical results** to the EAGLE reference
implementation. Runs both implementations side-by-side on the same pre-tokenized OCR
blocks and compares accepted tokens, spec steps, AR fallback steps, and end-to-end
speedup.

No GPU required. See [SIMULATION_RESULTS.md](SIMULATION_RESULTS.md) for the latest
simulator results and methodology.

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

Results from the **full 10,000-block** simulator on H100 NVL
(PaddleOCR-VL 0.3B):

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

### Estimated Speedup (from simulator, 10,000 blocks)

End-to-end speedup vs pure-AR baseline as
`n_gt / (spec_steps + ar_fallback_steps)`. See
[SIMULATION_RESULTS.md](SIMULATION_RESULTS.md) for full methodology
and caveats.

| Strategy | Chunk Size | Accept Rate | tok/spec step | Spec Steps | AR Fallback | Forward Passes | **Speedup** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stop_at_first | 16 | 26.2% | 1.45 | 412,687 | 126,649 | 539,336 | 1.35× |
| hybrid (mr=3) | 16 | 33.7% | 2.19 | 291,656 | 89,123 | 380,779 | **1.91×** |
| holdsnap (h=8) | 16 | 37.4% | 1.60 | 445,958 | 13,747 | 459,705 | 1.58× |
| stop_at_first | 50 | 16.2% | 1.44 | 261,660 | 350,211 | 611,871 | 1.19× |
| hybrid (mr=3) | 50 | 28.2% | 2.46 | 191,187 | 255,899 | 447,086 | **1.63×** |
| holdsnap (h=8) | 50 | 19.3% | 1.35 | 391,446 | 198,882 | 590,328 | 1.23× |
| stop_at_first | 200 | 11.6% | 1.44 | 184,932 | 459,709 | 644,641 | 1.13× |
| hybrid (mr=3) | 200 | 21.4% | 2.50 | 137,751 | 382,755 | 520,506 | **1.40×** |
| holdsnap (h=8) | 200 | 13.6% | 1.30 | 323,459 | 307,709 | 631,168 | 1.15× |

`n_gt = 726,871` total ground-truth tokens across all 10,000 blocks.
**hybrid (mr=3) at chunk_size=16 is the production-recommended config:
1.91× end-to-end speedup (41% fewer forward passes than pure AR).**

These assume verify and decode cost are equal, which holds for the 0.3B
PaddleOCR-VL model on H100 NVL (~10.2 ms decode, ~11.2 ms verify,
kernel-launch bound). For larger verifiers, scale spec-step cost by
`verify_ms / decode_ms`.

### Fidelity Check

vLLM's LCS implementation produces **identical results** to the EAGLE reference across
all 9 (strategy × chunk_size) configurations (0 mismatches on 10,000 blocks).

### vLLM E2E Results (H100 NVL, 1000 blocks, sequential)

Measured with `benchmark_parsed_draft_e2e.py --sequential --measure-phases` on
1000 blocks from GTX5k (reparsed with `reparse_ocr_text.py` for clean draft text).

#### stop_at_first (chunk=50)

```text
  Baseline time:    624,432 ms    Spec-decode time: 391,284 ms
  Baseline tok/s:   108.4         Spec-decode tok/s: 173.6
  Overall speedup:  1.60x

  Phase breakdown:
                           Baseline         Spec     Speedup
  Prefill (ms)               35,356       26,572          —
  Decode (ms)               569,614      345,508      1.65x
  Overhead (ms)              19,461       19,204          —

  Decode-only speedup: 1.65x
  Baseline time split: 6% prefill, 91% decode, 3% overhead
```

#### hybrid (mr=3, chunk=50, 20 blocks)

```text
  Baseline time:    13,267 ms     Spec-decode time: 1,704 ms
  Overall speedup:  7.79x

  Phase breakdown:
                           Baseline         Spec     Speedup
  Prefill (ms)                  880          650          —
  Decode (ms)                12,035          677     17.78x
  Overhead (ms)                 352          377          —

  Decode-only speedup: 17.78x
```

#### Analysis

The 0.3B PaddleOCR-VL model has nearly identical cost for verifying 50 tokens
vs decoding 1 token (~10ms vs ~11ms) because the GPU is underutilized at this
model size. This limits stop_at_first's effective speedup. The hybrid strategy
overcomes this by accepting ~2.5 tokens/step instead of ~1.4 (full-10k
simulator), dramatically reducing the number of verify rounds.

For larger models (7B+) where verify cost scales with sequence length, both
strategies will show proportionally larger speedups.
