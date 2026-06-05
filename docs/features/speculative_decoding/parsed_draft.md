# Parsed-Draft Speculation

Parsed-draft speculation uses **pre-existing text** (e.g., OCR-extracted from PDF via
mupdf) as a zero-cost draft for speculative decoding. Instead of running a neural draft
model, the parsed text is tokenized and fed as draft tokens for verification by the
target model.

This is designed for **OCR Vision-Language models** where a fast text extractor (like
mupdf) can provide a high-quality draft that the VL model verifies and corrects. The
draft is free — no GPU cost, no draft model weights.

!!! tip "When to use Parsed-Draft"
    Use this when you have pre-existing text that is close to the expected model output.
    The best use case is OCR: mupdf extracts text from PDFs, and a VL model verifies
    and corrects it. With ~50% token overlap between OCR draft and model output,
    parsed-draft achieves **2–6× speedup** depending on strategy.

!!! tip "LCS Alignment"
    Unlike other speculative methods, parsed-draft uses Longest Common Subsequence (LCS)
    alignment to match draft tokens to model output. This handles OCR insertions and
    deletions that would break simple positional matching.

## Offline (Python API)

Pass `draft_text` via `extra_args` in `SamplingParams`. The engine must be configured
with `speculative_config={"method": "parsed_draft"}`.

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="your-ocr-vl-model",
    speculative_config={
        "method": "parsed_draft",
        "num_speculative_tokens": 16,
    },
)

# Each request can have its own draft text (or none)
params_with_draft = SamplingParams(
    temperature=0.0,
    max_tokens=4096,
    extra_args={"draft_text": "text extracted from PDF by mupdf"},
)
params_no_draft = SamplingParams(temperature=0.0, max_tokens=4096)

# Mixed batch: some requests have draft text, some don't
outputs = llm.generate(
    ["<image prompt with draft>", "<image prompt without draft>"],
    [params_with_draft, params_no_draft],
)
```

## Online (OpenAI-compatible API)

Start the server with `--speculative-config`:

```bash
vllm serve your-ocr-vl-model \
    --speculative-config '{
        "method": "parsed_draft",
        "num_speculative_tokens": 16
    }'
```

Pass `draft_text` in the request body:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1")
response = client.chat.completions.create(
    model="your-ocr-vl-model",
    messages=[{"role": "user", "content": "OCR:"}],
    extra_body={"draft_text": "text extracted from PDF"},
)
```

## Configuration

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `parsed_draft_strategy` | `string` | `stop_at_first` | Decoding strategy. `stop_at_first` guarantees identical output to autoregressive. `hybrid` (Phase 2) trades minimal KV staleness for higher speedup. |
| `parsed_draft_max_reject` | `integer >= 1` | `3` | Max consecutive rejected tokens before bail-out in hybrid strategy. |

Example:

```bash
vllm serve your-ocr-vl-model \
    --speculative-config '{
        "method": "parsed_draft",
        "num_speculative_tokens": 16,
        "parsed_draft_strategy": "stop_at_first"
    }'
```

## How It Works

1. The user provides `draft_text` (e.g., mupdf OCR output) per request
2. The proposer tokenizes `draft_text` and serves chunks of `num_speculative_tokens` tokens
3. The target model verifies the chunk in a single forward pass
4. The rejection sampler accepts matching prefix tokens + 1 bonus token
5. **LCS-guided cursor advancement** skips past OCR insertions/deletions in the draft,
   so the next chunk starts at the right position even when the draft has extra or
   missing tokens
6. Repeat until draft is exhausted, then fall back to autoregressive

Requests without `draft_text` skip speculative decoding entirely and run autoregressively,
even in the same batch as requests that have it.

## Benchmark Results

On 100 OCR blocks (PaddleOCR-VL 0.3B, H100 NVL):

| Strategy | Accept Rate | Avg tokens/step | Speedup estimate |
| --- | --- | --- | --- |
| stop_at_first (c=16) | 47.4% | 2.2 | **2.4×** |
| stop_at_first (c=50) | 30.4% | 2.1 | **3.3×** |
| hybrid mr=3 (c=16) | 60.3% | 5.6 | **5.5×** |
| hybrid mr=3 (c=50) | 47.5% | 4.9 | **6.0×** |
