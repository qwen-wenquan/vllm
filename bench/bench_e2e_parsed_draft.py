#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E benchmark: vLLM baseline vs parsed-draft speculative decoding.

Loads PaddleOCR-VL via vLLM, processes real document images from GTX5k,
and compares:
  - Baseline: autoregressive (no draft_text)
  - Parsed-draft: same request but with draft_text from mupdf OCR

Measures wall-clock latency, throughput, and output fidelity.

Usage:
    export CUDA_HOME=/tmp/cuda_home
    export PATH=".venv/bin:$CUDA_HOME/bin:$PATH"
    export VLLM_USE_FLASHINFER_SAMPLER=0

    .venv/bin/python bench/bench_e2e_parsed_draft.py \
        --model /home/dlisuser/EAGLE/models/PaddleOCR-VL_finetune \
        --samples-dir /home/dlisuser/EAGLE/models/samples/GTX5k \
        --max-docs 1 --max-blocks 50
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Any, cast

# RoPE patch — must run before any transformers/vllm import
try:
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    if "default" not in ROPE_INIT_FUNCTIONS:
        import torch

        def _compute_default_rope_parameters(
            config, device=None, **kwargs
        ):
            base = config.rope_theta
            prf = (
                config.partial_rotary_factor
                if hasattr(config, "partial_rotary_factor")
                else 1.0
            )
            head_dim = getattr(
                config,
                "head_dim",
                config.hidden_size // config.num_attention_heads,
            )
            dim = int(head_dim * prf)
            inv_freq = 1.0 / (
                base
                ** (
                    torch.arange(0, dim, 2, dtype=torch.int64)
                    .float()
                    .to(device)
                    / dim
                )
            )
            return inv_freq, 1.0

        ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters
except ImportError:
    pass

import torch
from PIL import Image
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.entrypoints import chat_utils as vllm_chat_utils
from vllm.inputs import TokensPrompt

ChatCompletionMessageParam = vllm_chat_utils.ChatCompletionMessageParam
parse_chat_messages = vllm_chat_utils.parse_chat_messages

# ── Prompt mapping (matches eval_vllm.py) ──────────────────────────────
PROMPT_MAP = {
    9: "Table Recognition:",
    3: "Formula Recognition:",
}
DEFAULT_PROMPT = "OCR:"
SKIP_CATEGORIES = {7}  # Picture


def get_prompt(category_id: int) -> str:
    return PROMPT_MAP.get(category_id, DEFAULT_PROMPT)


# ── Image utilities (from eval_vllm.py) ────────────────────────────────
def preprocess_image_with_padding(
    image: Image.Image, pad_size: int | None = None
) -> Image.Image:
    width, height = image.size
    if pad_size is None:
        min_dim = min(width, height)
        calculated_pad = int(0.1 * min_dim)
        pad_size = min(200, max(100, calculated_pad))
    new_width = width + 2 * pad_size
    new_height = height + 2 * pad_size
    try:
        padded = Image.new(
            image.mode, (new_width, new_height), color=(255, 255, 255)
        )
    except ValueError:
        image = image.convert("RGB")
        padded = Image.new(
            image.mode, (new_width, new_height), color=(255, 255, 255)
        )
    padded.paste(image, (pad_size, pad_size))
    return padded


def image_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{b64}"


# ── PDF processing (from eval_vllm.py) ─────────────────────────────────
def render_pdf_pages(pdf_path: str, dpi: int = 200) -> dict:
    import fitz

    doc = fitz.open(pdf_path)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pages = {}
    for i in range(len(doc)):
        pix = doc[i].get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        pages[i] = img
    doc.close()
    return pages


def crop_block(page_img: Image.Image, bbox_norm: list) -> Image.Image:
    W, H = page_img.size
    x, y, w, h = bbox_norm
    x1 = max(0, int(x * W))
    y1 = max(0, int(y * H))
    x2 = min(W, int((x + w) * W))
    y2 = min(H, int((y + h) * H))
    return page_img.crop((x1, y1, x2, y2))


def load_document_blocks(
    samples_dir: str,
    dpi: int = 200,
    max_docs: int | None = None,
    max_blocks: int | None = None,
) -> list[dict]:
    """Load blocks from GTX5k-style document directories."""
    samples_path = Path(samples_dir)

    doc_dirs = sorted(
        d
        for d in samples_path.iterdir()
        if d.is_dir() and (d / "det_ocr_results.json").exists()
    )
    if max_docs:
        doc_dirs = doc_dirs[:max_docs]

    blocks: list[dict] = []
    for doc_dir in doc_dirs:
        det_path = doc_dir / "det_ocr_results.json"
        pdf_path = doc_dir / "document.pdf"
        if not pdf_path.exists():
            continue

        with open(det_path) as f:
            det_data = json.load(f)

        orig_to_packed: dict[int, int] = {}
        page_spans = det_data.get("page_spans", [])
        packed_spans = det_data.get("packed_page_spans", [])
        if page_spans and packed_spans:
            for orig_span, packed_span in zip(page_spans, packed_spans):
                for o, p in zip(
                    range(orig_span[0], orig_span[1] + 1),
                    range(packed_span[0], packed_span[1] + 1),
                ):
                    orig_to_packed[o] = p

        page_images = render_pdf_pages(str(pdf_path), dpi=dpi)

        for page_key, page_data in det_data["pages"].items():
            pdf_page_orig = page_data.get("pdf_page", int(page_key))
            pdf_page = orig_to_packed.get(pdf_page_orig, pdf_page_orig)

            for blk_idx, blk in enumerate(page_data["blocks"]):
                cat_id = blk.get("category_id", 0)
                if cat_id in SKIP_CATEGORIES:
                    continue
                parsed_text = blk.get("parsed_text", "").strip()
                gt_text = blk.get("gt_text", "").strip()
                if not gt_text:
                    continue
                if pdf_page not in page_images:
                    continue

                crop = crop_block(page_images[pdf_page], blk["bbox"])
                if crop.size[0] <= 0 or crop.size[1] <= 0:
                    continue

                blocks.append(
                    {
                        "doc_id": doc_dir.name,
                        "page": page_key,
                        "block_idx": blk_idx,
                        "category_name": blk.get("category_name", ""),
                        "category_id": cat_id,
                        "parsed_text": parsed_text.replace("\n", " ").strip(),
                        "gt_text": gt_text,
                        "prompt": get_prompt(cat_id),
                        "crop": crop,
                    }
                )
                if max_blocks and len(blocks) >= max_blocks:
                    return blocks

        del page_images

    return blocks


# ── vLLM request helpers (from eval_vllm.py) ──────────────────────────
def parse_chat_messages_compat(
    tokenizer, model_config, messages, content_format
):
    try:
        return parse_chat_messages(
            messages,
            model_config,
            tokenizer,
            content_format=content_format,
        )
    except TypeError:
        return parse_chat_messages(
            messages,
            model_config,
            content_format=content_format,
        )


def prepare_request(
    tokenizer,
    model_config,
    image: Image.Image,
    prompt: str,
) -> TokensPrompt:
    padded = preprocess_image_with_padding(image.convert("RGB"))
    messages = cast(
        list[ChatCompletionMessageParam],
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_to_base64(padded)},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    conversation, mm_data, mm_uuids = parse_chat_messages_compat(
        tokenizer, model_config, messages, content_format="openai"
    )
    prompt_str = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    prompt_token_ids = tokenizer.encode(
        prompt_str, add_special_tokens=False
    )
    request = TokensPrompt(prompt_token_ids=prompt_token_ids)
    if mm_data is not None:
        request["multi_modal_data"] = mm_data
    if mm_uuids is not None:
        request["multi_modal_uuids"] = mm_uuids
    return request


# ── Edit distance ──────────────────────────────────────────────────────
def edit_distance(a: str, b: str) -> int:
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


# ── Main ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="E2E benchmark: baseline vs parsed-draft"
    )
    parser.add_argument(
        "--model",
        default="/home/dlisuser/EAGLE/models/PaddleOCR-VL_finetune",
    )
    parser.add_argument(
        "--samples-dir",
        default="/home/dlisuser/EAGLE/models/samples/GTX5k",
    )
    parser.add_argument("--max-docs", type=int, default=1)
    parser.add_argument("--max-blocks", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    # ── Load blocks from PDFs ──────────────────────────────────────
    print(f"Loading blocks from {args.samples_dir}...")
    blocks = load_document_blocks(
        args.samples_dir,
        dpi=args.dpi,
        max_docs=args.max_docs,
        max_blocks=args.max_blocks,
    )
    # Only keep blocks with parsed_text (for draft comparison)
    blocks_with_draft = [b for b in blocks if b["parsed_text"]]
    blocks_without_draft = [b for b in blocks if not b["parsed_text"]]
    print(
        f"Loaded {len(blocks)} blocks "
        f"({len(blocks_with_draft)} with parsed_text, "
        f"{len(blocks_without_draft)} without)"
    )

    # ── Load vLLM ──────────────────────────────────────────────────
    print(f"\nLoading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        padding_side="left",
        use_fast=True,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    eos_token_id = (
        tokenizer.eos_token_id
        if tokenizer.eos_token_id is not None
        else 2
    )

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        max_model_len=4096,
        enforce_eager=True,
        gpu_memory_utilization=0.9,
        enable_prefix_caching=False,
    )
    model_config = llm.llm_engine.model_config
    print("Model loaded.\n")

    # ── Prepare requests ───────────────────────────────────────────
    print("Preparing requests...")
    requests_baseline: list[TokensPrompt] = []
    requests_draft: list[TokensPrompt] = []
    params_baseline: list[SamplingParams] = []
    params_draft: list[SamplingParams] = []
    valid_blocks: list[dict] = []

    for blk in blocks_with_draft:
        try:
            req = prepare_request(
                tokenizer, model_config, blk["crop"], blk["prompt"]
            )
        except Exception as e:
            print(f"  SKIP: {e}")
            continue

        # Baseline: no draft_text
        sp_base = SamplingParams(
            temperature=0.0, max_tokens=args.max_new_tokens
        )
        # Parsed-draft: include draft_text in extra_args
        sp_draft = SamplingParams(
            temperature=0.0,
            max_tokens=args.max_new_tokens,
            extra_args={"draft_text": blk["parsed_text"]},
        )

        requests_baseline.append(req)
        requests_draft.append(req)
        params_baseline.append(sp_base)
        params_draft.append(sp_draft)
        valid_blocks.append(blk)

    print(f"Prepared {len(valid_blocks)} request pairs\n")

    if not valid_blocks:
        print("No valid blocks to benchmark.")
        return

    # ── Warmup ─────────────────────────────────────────────────────
    print("Warming up...")
    _ = llm.generate(
        [requests_baseline[0]],
        [SamplingParams(temperature=0.0, max_tokens=10)],
    )
    print("Warmup done.\n")

    # ── Run baseline (autoregressive) ──────────────────────────────
    print(f"{'='*60}")
    print(
        f"BASELINE: {len(valid_blocks)} blocks, "
        f"autoregressive (no draft)"
    )
    print(f"{'='*60}")

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    baseline_outputs = llm.generate(
        requests_baseline, sampling_params=params_baseline
    )
    torch.cuda.synchronize()
    baseline_ms = (time.perf_counter() - t0) * 1000

    baseline_texts = [
        o.outputs[0].text if o.outputs else "" for o in baseline_outputs
    ]
    baseline_token_counts = [
        len(o.outputs[0].token_ids) if o.outputs else 0
        for o in baseline_outputs
    ]
    total_baseline_tokens = sum(baseline_token_counts)

    print(
        f"  Time:       {baseline_ms:.0f} ms"
    )
    print(f"  Tokens:     {total_baseline_tokens}")
    print(
        f"  Throughput: {total_baseline_tokens / (baseline_ms / 1000):.1f} tok/s"
    )
    print(
        f"  Latency:    {baseline_ms / total_baseline_tokens:.2f} ms/tok"
        if total_baseline_tokens
        else ""
    )

    # ── Run with draft_text in extra_args ──────────────────────────
    # NOTE: This doesn't trigger parsed-draft spec decode yet (that
    # requires speculative_config). This run just verifies that
    # draft_text flows through without breaking inference, and
    # produces identical output (since it's ignored without spec config).
    print(f"\n{'='*60}")
    print(
        f"WITH DRAFT_TEXT: {len(valid_blocks)} blocks "
        f"(draft_text in extra_args)"
    )
    print(f"{'='*60}")

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    draft_outputs = llm.generate(
        requests_draft, sampling_params=params_draft
    )
    torch.cuda.synchronize()
    draft_ms = (time.perf_counter() - t0) * 1000

    draft_texts = [
        o.outputs[0].text if o.outputs else "" for o in draft_outputs
    ]
    draft_token_counts = [
        len(o.outputs[0].token_ids) if o.outputs else 0
        for o in draft_outputs
    ]
    total_draft_tokens = sum(draft_token_counts)

    print(f"  Time:       {draft_ms:.0f} ms")
    print(f"  Tokens:     {total_draft_tokens}")
    print(
        f"  Throughput: {total_draft_tokens / (draft_ms / 1000):.1f} tok/s"
    )

    # ── Compare outputs ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")

    exact_matches = 0
    total_edit_dist = 0
    for i, blk in enumerate(valid_blocks):
        bt = baseline_texts[i]
        dt = draft_texts[i]
        if bt == dt:
            exact_matches += 1
        ed = edit_distance(bt, dt)
        total_edit_dist += ed
        if bt != dt and i < 3:
            print(
                f"\n  Block {i} ({blk['category_name']}) DIFFERS:"
            )
            print(f"    baseline: {bt[:80]!r}")
            print(f"    w/draft:  {dt[:80]!r}")
            print(f"    edit_dist: {ed}")

    n = len(valid_blocks)
    print(f"\n  Blocks:           {n}")
    print(
        f"  Exact match:      {exact_matches}/{n} "
        f"({exact_matches/n*100:.1f}%)"
    )
    print(
        f"  Mean edit dist:   {total_edit_dist/n:.1f} chars"
    )
    print(
        f"  Baseline time:    {baseline_ms:.0f} ms"
    )
    print(
        f"  With-draft time:  {draft_ms:.0f} ms"
    )

    # ── Per-block detail: compare model output vs gt_text ──────────
    print(f"\n{'='*60}")
    print("SAMPLE OUTPUTS (first 5 blocks)")
    print(f"{'='*60}")
    for i in range(min(5, n)):
        blk = valid_blocks[i]
        bt = baseline_texts[i]
        gt = blk["gt_text"]
        pt = blk["parsed_text"]
        ed_gt = edit_distance(bt, gt)
        print(
            f"\n  [{i}] {blk['category_name']} "
            f"(p{blk['page']}/b{blk['block_idx']})"
        )
        print(f"    gt:       {gt[:80]!r}")
        print(f"    parsed:   {pt[:80]!r}")
        print(f"    baseline: {bt[:80]!r}")
        print(f"    edit(bl,gt): {ed_gt}")

    del llm


if __name__ == "__main__":
    main()
