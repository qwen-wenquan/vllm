#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E benchmark: vLLM baseline vs parsed-draft speculative decoding.

Loads PaddleOCR-VL via vLLM, processes real document images from GTX5k,
and compares:
  - Baseline: autoregressive decoding (standard LLM)
  - Parsed-draft: speculative decoding with draft_text from mupdf OCR

Supports three batching modes:
  - batched (default): all requests in one generate() call
  - sequential: one request at a time (measures per-request latency)
  - constrained: batched with max_num_seqs limit

Usage:
    export VLLM_USE_FLASHINFER_SAMPLER=0
    export VLLM_USE_DEEP_GEMM=0

    # Batched (default) — measures throughput
    .venv/bin/python benchmarks/spec_decode/benchmark_parsed_draft_e2e.py \\
        --model /path/to/PaddleOCR-VL_finetune \\
        --samples-dir /path/to/samples/GTX5k \\
        --max-docs 1 --max-blocks 50

    # Sequential — measures per-request latency speedup
    .venv/bin/python benchmarks/spec_decode/benchmark_parsed_draft_e2e.py \\
        --model /path/to/PaddleOCR-VL_finetune \\
        --samples-dir /path/to/samples/GTX5k \\
        --max-docs 1 --max-blocks 50 \\
        --sequential

    # Constrained batching — simulate smaller GPU (e.g. A10G)
    .venv/bin/python benchmarks/spec_decode/benchmark_parsed_draft_e2e.py \\
        --model /path/to/PaddleOCR-VL_finetune \\
        --samples-dir /path/to/samples/GTX5k \\
        --max-docs 5 --max-blocks 200 \\
        --max-num-seqs 4
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import cast

import pybase64 as base64

# RoPE patch — must run before any transformers/vllm import
try:
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    if "default" not in ROPE_INIT_FUNCTIONS:
        import torch

        def _compute_default_rope_parameters(config, device=None, **kwargs):
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
                ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim)
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
from vllm.sampling_params import RepetitionDetectionParams

ChatCompletionMessageParam = vllm_chat_utils.ChatCompletionMessageParam
parse_chat_messages = vllm_chat_utils.parse_chat_messages

# ── Prompt mapping (matches eval_vllm.py) ─────────────────────────
PROMPT_MAP = {
    9: "Table Recognition:",
    3: "Formula Recognition:",
}
DEFAULT_PROMPT = "OCR:"
SKIP_CATEGORIES = {7}  # Picture


def get_prompt(category_id: int) -> str:
    return PROMPT_MAP.get(category_id, DEFAULT_PROMPT)


# ── Image utilities (from eval_vllm.py) ───────────────────────────
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
            image.mode,
            (new_width, new_height),
            color=(255, 255, 255),
        )
    except ValueError:
        image = image.convert("RGB")
        padded = Image.new(
            image.mode,
            (new_width, new_height),
            color=(255, 255, 255),
        )
    padded.paste(image, (pad_size, pad_size))
    return padded


def image_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{b64}"


# ── PDF processing (from eval_vllm.py) ────────────────────────────
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
    require_parsed_text: bool = False,
) -> list[dict]:
    """Load blocks from GTX5k-style document directories.

    Args:
        samples_dir: Path to the samples directory.
        dpi: DPI for PDF rendering.
        max_docs: Maximum number of documents to load.
        max_blocks: Maximum total blocks to return.
        require_parsed_text: If True, skip blocks where
            parsed_text is empty. Useful for spec-decode
            benchmarks where blocks without OCR text
            cannot use draft tokens.
    """
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
                if require_parsed_text and not parsed_text:
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


# ── vLLM request helpers (from eval_vllm.py) ─────────────────────
def parse_chat_messages_compat(tokenizer, model_config, messages, content_format):
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
        tokenizer,
        model_config,
        messages,
        content_format="openai",
    )
    prompt_str = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_token_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
    request = TokensPrompt(prompt_token_ids=prompt_token_ids)
    if mm_data is not None:
        request["multi_modal_data"] = mm_data
    if mm_uuids is not None:
        request["multi_modal_uuids"] = mm_uuids
    return request


def prepare_all_requests(
    blocks: list[dict],
    tokenizer,
    model_config,
) -> tuple[list[TokensPrompt], list[dict]]:
    """Prepare vLLM requests for all blocks.

    Returns (requests, valid_blocks) — blocks that failed
    preparation are skipped.
    """
    requests: list[TokensPrompt] = []
    valid: list[dict] = []
    for blk in blocks:
        try:
            req = prepare_request(
                tokenizer,
                model_config,
                blk["crop"],
                blk["prompt"],
            )
            requests.append(req)
            valid.append(blk)
        except Exception as e:
            print(f"  SKIP: {e}")
    return requests, valid


# ── Edit distance ─────────────────────────────────────────────────
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


# ── Phase stats ───────────────────────────────────────────────────
@dataclass
class PhaseStats:
    """Per-block prefill/decode timing from engine metrics."""

    prefill_ms: float
    decode_ms: float
    num_gen_tokens: int


def extract_phase_stats(
    output,
) -> PhaseStats | None:
    """Extract prefill/decode times from RequestOutput.metrics.

    Returns None if metrics are unavailable (log_stats
    disabled).
    """
    m = output.metrics
    if m is None:
        return None
    if m.first_token_ts == 0 or m.scheduled_ts == 0:
        return None
    prefill_s = m.first_token_ts - m.scheduled_ts
    decode_s = (
        m.last_token_ts - m.first_token_ts
        if m.last_token_ts > m.first_token_ts
        else 0.0
    )
    return PhaseStats(
        prefill_ms=prefill_s * 1000,
        decode_ms=decode_s * 1000,
        num_gen_tokens=m.num_generation_tokens,
    )


# ── Run helpers ───────────────────────────────────────────────────
def run_inference(
    llm: LLM,
    requests: list[TokensPrompt],
    sampling_params: list[SamplingParams],
    label: str,
    sequential: bool = False,
) -> tuple[list[str], list[int], float, list[PhaseStats | None]]:
    """Run generate().

    Returns (texts, token_counts, time_ms, phase_stats).
    phase_stats is a list of per-request PhaseStats (or None
    when engine metrics are unavailable).

    If sequential=True, runs one request at a time to measure
    per-request latency (the regime where spec decode shines).
    """
    print(f"\n{'=' * 60}")
    mode = "sequential" if sequential else "batched"
    print(f"{label} [{mode}]: {len(requests)} blocks")
    print(f"{'=' * 60}")

    # Warmup — use the actual first sampling params so that
    # extra_args (e.g. draft_text) are present during warmup.
    warmup_sp = SamplingParams(
        temperature=0.0,
        max_tokens=10,
        extra_args=(sampling_params[0].extra_args if sampling_params else None),
    )
    _ = llm.generate([requests[0]], [warmup_sp])

    torch.accelerator.synchronize()
    t0 = time.perf_counter()

    if sequential:
        # One request at a time — measures per-request latency
        all_outputs = []
        for req, sp in zip(requests, sampling_params):
            out = llm.generate([req], [sp])
            all_outputs.extend(out)
    else:
        # All requests in one call — measures throughput
        all_outputs = llm.generate(requests, sampling_params=sampling_params)

    torch.accelerator.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    texts = [o.outputs[0].text if o.outputs else "" for o in all_outputs]
    token_counts = [
        len(o.outputs[0].token_ids) if o.outputs else 0 for o in all_outputs
    ]
    total_tokens = sum(token_counts)

    # Extract per-request phase stats
    phase_stats = [extract_phase_stats(o) for o in all_outputs]

    print(f"  Time:       {elapsed_ms:.0f} ms")
    print(f"  Tokens:     {total_tokens}")
    if elapsed_ms > 0:
        tps = total_tokens / (elapsed_ms / 1000)
        print(f"  Throughput: {tps:.1f} tok/s")
    if total_tokens > 0:
        lat = elapsed_ms / total_tokens
        print(f"  Latency:    {lat:.2f} ms/tok")
    if sequential and len(requests) > 0:
        avg_lat = elapsed_ms / len(requests)
        print(f"  Avg/block:  {avg_lat:.0f} ms")

    # Print phase breakdown if available
    valid_phases = [p for p in phase_stats if p is not None]
    if valid_phases:
        total_prefill = sum(p.prefill_ms for p in valid_phases)
        total_decode = sum(p.decode_ms for p in valid_phases)
        avg_prefill = total_prefill / len(valid_phases)
        avg_decode = total_decode / len(valid_phases)
        print("  --- Phase breakdown (engine timestamps) ---")
        print(
            f"  Total prefill: {total_prefill:.0f} ms  (avg {avg_prefill:.1f} ms/block)"
        )
        print(
            f"  Total decode:  {total_decode:.0f} ms  (avg {avg_decode:.1f} ms/block)"
        )
        overhead = elapsed_ms - total_prefill - total_decode
        print(f"  Overhead:      {overhead:.0f} ms  (scheduling, IPC, tokenization)")

    return texts, token_counts, elapsed_ms, phase_stats


# ── Main ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description=("E2E benchmark: baseline vs parsed-draft speculative decoding"),
    )
    parser.add_argument(
        "--model",
        default=("/home/dlisuser/EAGLE/models/PaddleOCR-VL_finetune"),
    )
    parser.add_argument(
        "--samples-dir",
        default=("data/GTX5k"),
    )
    parser.add_argument("--max-docs", type=int, default=1)
    parser.add_argument("--max-blocks", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--dpi", type=int, default=200)

    # Speculative decoding configuration
    parser.add_argument(
        "--num-speculative-tokens",
        type=int,
        default=16,
        help="Draft chunk size in tokens (default: 16)",
    )
    parser.add_argument(
        "--parsed-draft-strategy",
        type=str,
        default="stop_at_first",
        choices=["stop_at_first", "hybrid"],
        help=("Parsed-draft decoding strategy (default: stop_at_first)"),
    )
    parser.add_argument(
        "--parsed-draft-max-reject",
        type=int,
        default=3,
        help=(
            "Max consecutive rejects before bail-out in hybrid strategy (default: 3)"
        ),
    )
    parser.add_argument(
        "--parsed-draft-holdsnap",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable holdsnap cursor rule for no-prefix-match case "
            "(SNAP cursor to correction if found in chunk, otherwise "
            "HOLD up to --parsed-draft-max-hold consecutive steps). "
            "Lossless. Default OFF — matches the production default. "
            "Holdsnap is a simulator-tuned rule that doesn't translate "
            "to real verifiers (the simulator's gt-stream verifier "
            "slides forward independently of the draft; the real "
            "verifier re-rejects the same draft on a hold). See "
            "benchmarks/spec_decode/SIMULATION_RESULTS.md for the "
            "1.49x → 1.03x regression measured on GTX5k."
        ),
    )
    parser.add_argument(
        "--parsed-draft-max-hold",
        type=int,
        default=8,
        help=(
            "Maximum consecutive holds before forcing a 1-step cursor "
            "advance when holdsnap is enabled. Matches the benchmark "
            "sweep optimum at chunk_size=16 (default: 8). Larger values "
            "are safe on OCR data."
        ),
    )
    parser.add_argument(
        "--parsed-draft-lcs-backend",
        type=str,
        default="auto",
        choices=["auto", "python", "numpy", "triton", "positional"],
        help=(
            "Backend for LCS draft-cursor advancement. "
            "'auto' (default): python for small batches, "
            "numpy for >= 64 requests. "
            "'triton': GPU Triton kernel. "
            "'positional': DISABLES LCS — cursor advances by "
            "prefix_len+1 (no insertion/deletion recovery). "
            "Use only to A/B the value of LCS alignment."
        ),
    )
    parser.add_argument(
        "--repetition-detection",
        action="store_true",
        default=True,
        help="Enable vLLM's built-in repetition detection.",
    )
    parser.add_argument(
        "--measure-phases",
        action="store_true",
        default=False,
        help=(
            "Break down timing into prefill vs decode "
            "phases using engine-internal timestamps. "
            "Implies --sequential and enables log_stats."
        ),
    )

    # Batching mode
    batch_group = parser.add_mutually_exclusive_group()
    batch_group.add_argument(
        "--sequential",
        action="store_true",
        default=False,
        help=(
            "Process one block at a time instead of "
            "batching all requests. Measures per-request "
            "latency — the regime where spec decode "
            "gives the largest speedup."
        ),
    )
    batch_group.add_argument(
        "--max-num-seqs",
        type=int,
        default=None,
        help=(
            "Limit max concurrent requests in the "
            "engine (continuous batching constraint). "
            "Simulates smaller GPUs like A10G. "
            "E.g. --max-num-seqs 4."
        ),
    )

    args = parser.parse_args()

    # ── Load blocks from PDFs ─────────────────────────────────
    print(f"Loading blocks from {args.samples_dir}...")
    blocks = load_document_blocks(
        args.samples_dir,
        dpi=args.dpi,
        max_docs=args.max_docs,
        max_blocks=args.max_blocks,
        require_parsed_text=True,
    )
    print(f"Loaded {len(blocks)} blocks (all with parsed_text)")
    if not blocks:
        print("No blocks with parsed_text to benchmark.")
        return

    # Build repetition detection params if enabled
    rep_params = None
    if args.repetition_detection:
        rep_params = RepetitionDetectionParams(
            max_pattern_size=10,
            min_count=5,
        )
        print("Repetition detection enabled: max_pattern_size=10, min_count=5")

    # --measure-phases implies sequential mode
    if args.measure_phases:
        args.sequential = True

    # Print batching mode
    if args.sequential:
        print("Batching mode: SEQUENTIAL (1 req at a time)")
    elif args.max_num_seqs:
        print(f"Batching mode: CONSTRAINED (max_num_seqs={args.max_num_seqs})")
    else:
        print("Batching mode: BATCHED (all at once)")
    if args.measure_phases:
        print("Phase measurement: ENABLED")

    # Load tokenizer once (shared across both LLM instances)
    print(f"\nLoading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        padding_side="left",
        use_fast=True,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Common LLM kwargs
    llm_kwargs: dict = dict(
        model=args.model,
        trust_remote_code=True,
        max_model_len=4096,
        enforce_eager=True,
        gpu_memory_utilization=0.9,
        enable_prefix_caching=False,
    )
    if args.max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = args.max_num_seqs
    if args.measure_phases:
        # Enable engine stats so RequestOutput.metrics
        # has prefill/decode timestamps.
        llm_kwargs["disable_log_stats"] = False

    # ══════════════════════════════════════════════════════════
    # PHASE 1: Baseline (autoregressive, no speculative config)
    # ══════════════════════════════════════════════════════════
    print(f"\n{'#' * 60}")
    print("# PHASE 1: Loading baseline LLM (no spec decode)")
    print(f"{'#' * 60}")

    llm_baseline = LLM(**llm_kwargs)
    model_config = llm_baseline.llm_engine.model_config

    # Prepare requests against baseline model_config
    print("Preparing requests...")
    baseline_requests, valid_blocks = prepare_all_requests(
        blocks, tokenizer, model_config
    )
    print(f"Prepared {len(valid_blocks)} requests")

    if not valid_blocks:
        print("No valid blocks to benchmark.")
        del llm_baseline
        return

    # Build baseline sampling params (no draft_text)
    baseline_params = [
        SamplingParams(
            temperature=0.0,
            max_tokens=args.max_new_tokens,
            repetition_detection=rep_params,
        )
        for _ in valid_blocks
    ]

    baseline_texts, baseline_tokens, baseline_ms, bl_phases = run_inference(
        llm_baseline,
        baseline_requests,
        baseline_params,
        "BASELINE (autoregressive)",
        sequential=args.sequential,
    )

    # Free baseline LLM to make room for spec-decode LLM
    del llm_baseline
    gc.collect()
    torch.accelerator.empty_cache()

    # ══════════════════════════════════════════════════════════
    # PHASE 2: Parsed-draft speculative decoding
    # ══════════════════════════════════════════════════════════
    spec_config = {
        "model": "parsed_draft",
        "num_speculative_tokens": (args.num_speculative_tokens),
        "parsed_draft_strategy": (args.parsed_draft_strategy),
        "parsed_draft_max_reject": (args.parsed_draft_max_reject),
        "parsed_draft_holdsnap": (args.parsed_draft_holdsnap),
        "parsed_draft_max_hold": (args.parsed_draft_max_hold),
        "parsed_draft_lcs_backend": (args.parsed_draft_lcs_backend),
    }

    print(f"\n{'#' * 60}")
    print("# PHASE 2: Loading spec-decode LLM")
    print(f"#   speculative_config = {spec_config}")
    print(f"{'#' * 60}")

    llm_spec = LLM(
        **llm_kwargs,
        speculative_config=spec_config,
    )
    spec_model_config = llm_spec.llm_engine.model_config

    # Re-prepare requests against spec-decode model_config
    print("Preparing requests for spec-decode engine...")
    spec_requests, spec_valid_blocks = prepare_all_requests(
        valid_blocks, tokenizer, spec_model_config
    )

    # Pre-tokenize draft text BEFORE the timed run so
    # tokenizer cost is not counted in the speedup.
    print("Pre-tokenizing draft text...")
    draft_token_ids_per_block: list[list[int]] = []
    for blk in spec_valid_blocks:
        ids = tokenizer.encode(blk["parsed_text"], add_special_tokens=False)
        draft_token_ids_per_block.append(ids)
    total_draft_tokens = sum(len(ids) for ids in draft_token_ids_per_block)
    print(
        f"  {len(draft_token_ids_per_block)} blocks, "
        f"{total_draft_tokens} draft tokens total"
    )

    # Build spec-decode sampling params with pre-tokenized IDs
    spec_params = [
        SamplingParams(
            temperature=0.0,
            max_tokens=args.max_new_tokens,
            extra_args={"draft_token_ids": draft_ids},
            repetition_detection=rep_params,
        )
        for draft_ids in draft_token_ids_per_block
    ]

    spec_texts, spec_tokens, spec_ms, sp_phases = run_inference(
        llm_spec,
        spec_requests,
        spec_params,
        (
            f"PARSED-DRAFT SPEC DECODE "
            f"(strategy={args.parsed_draft_strategy}, "
            f"chunk={args.num_speculative_tokens}, "
            f"holdsnap={'on' if args.parsed_draft_holdsnap else 'off'}"
            + (f"/h={args.parsed_draft_max_hold}" if args.parsed_draft_holdsnap else "")
            + ")"
        ),
        sequential=args.sequential,
    )

    del llm_spec
    gc.collect()
    torch.accelerator.empty_cache()

    # ══════════════════════════════════════════════════════════
    # PHASE 3: Compare outputs
    # ══════════════════════════════════════════════════════════
    n = min(len(baseline_texts), len(spec_texts))

    print(f"\n{'=' * 60}")
    print("COMPARISON")
    print(f"{'=' * 60}")

    exact_matches = 0
    total_edit_dist = 0
    for i in range(n):
        bt = baseline_texts[i]
        st = spec_texts[i]
        if bt == st:
            exact_matches += 1
        ed = edit_distance(bt, st)
        total_edit_dist += ed
        if bt != st and i < 3:
            blk = valid_blocks[i]
            print(f"\n  Block {i} ({blk['category_name']}) DIFFERS:")
            print(f"    baseline:  {bt[:80]!r}")
            print(f"    spec:      {st[:80]!r}")
            print(f"    edit_dist: {ed}")

    total_baseline = sum(baseline_tokens[:n])
    total_spec = sum(spec_tokens[:n])
    baseline_tps = total_baseline / (baseline_ms / 1000) if baseline_ms > 0 else 0
    spec_tps = total_spec / (spec_ms / 1000) if spec_ms > 0 else 0
    speedup = baseline_ms / spec_ms if spec_ms > 0 else 0

    print(f"\n  Blocks:           {n}")
    print(f"  Exact match:      {exact_matches}/{n} ({exact_matches / n * 100:.1f}%)")
    print(f"  Mean edit dist:   {total_edit_dist / n:.1f} chars")
    print(f"  Baseline time:    {baseline_ms:.0f} ms")
    print(f"  Spec-decode time: {spec_ms:.0f} ms")
    print(f"  Baseline tok/s:   {baseline_tps:.1f}")
    print(f"  Spec-decode tok/s:{spec_tps:.1f}")
    print(f"  Speedup:          {speedup:.2f}x")

    # ── Phase breakdown (prefill vs decode) ───────────────────
    bl_valid = [p for p in bl_phases[:n] if p is not None]
    sp_valid = [p for p in sp_phases[:n] if p is not None]
    if bl_valid and sp_valid:
        bl_prefill = sum(p.prefill_ms for p in bl_valid)
        bl_decode = sum(p.decode_ms for p in bl_valid)
        sp_prefill = sum(p.prefill_ms for p in sp_valid)
        sp_decode = sum(p.decode_ms for p in sp_valid)

        avg_bl_pre = bl_prefill / len(bl_valid)
        avg_bl_dec = bl_decode / len(bl_valid)
        avg_sp_pre = sp_prefill / len(sp_valid)
        avg_sp_dec = sp_decode / len(sp_valid)

        decode_speedup = bl_decode / sp_decode if sp_decode > 0 else 0

        bl_overhead = baseline_ms - bl_prefill - bl_decode
        sp_overhead = spec_ms - sp_prefill - sp_decode

        print(f"\n  {'─' * 56}")
        print("  PHASE BREAKDOWN (engine timestamps)")
        print(f"  {'─' * 56}")
        print(f"  {'':20s} {'Baseline':>12s} {'Spec':>12s} {'Speedup':>8s}")
        print(
            f"  {'Prefill (ms)':20s} {bl_prefill:>12.0f} {sp_prefill:>12.0f} {'—':>8s}"
        )
        print(
            f"  {'Decode (ms)':20s} "
            f"{bl_decode:>12.0f} "
            f"{sp_decode:>12.0f} "
            f"{decode_speedup:>7.2f}x"
        )
        print(
            f"  {'Overhead (ms)':20s} "
            f"{bl_overhead:>12.0f} "
            f"{sp_overhead:>12.0f} "
            f"{'—':>8s}"
        )
        print(f"  {'Avg prefill/blk':20s} {avg_bl_pre:>11.1f}  {avg_sp_pre:>11.1f}")
        print(f"  {'Avg decode/blk':20s} {avg_bl_dec:>11.1f}  {avg_sp_dec:>11.1f}")
        print(f"\n  Decode-only speedup: {decode_speedup:.2f}x")
        if baseline_ms > 0:
            pct_prefill = bl_prefill / baseline_ms * 100
            pct_decode = bl_decode / baseline_ms * 100
            pct_overhead = bl_overhead / baseline_ms * 100
            print(
                f"  Baseline time split: "
                f"{pct_prefill:.0f}% prefill, "
                f"{pct_decode:.0f}% decode, "
                f"{pct_overhead:.0f}% overhead"
            )
    elif args.measure_phases:
        print("\n  WARNING: No phase stats available. Check that log_stats is enabled.")

    # ── Per-block detail ──────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("SAMPLE OUTPUTS (first 5 blocks)")
    print(f"{'=' * 60}")
    for i in range(min(5, n)):
        blk = valid_blocks[i]
        bt = baseline_texts[i]
        st = spec_texts[i]
        gt = blk["gt_text"]
        ed_bl_gt = edit_distance(bt, gt)
        ed_sp_gt = edit_distance(st, gt)
        print(f"\n  [{i}] {blk['category_name']} (p{blk['page']}/b{blk['block_idx']})")
        print(f"    gt:       {gt[:80]!r}")
        print(f"    parsed:   {blk['parsed_text'][:80]!r}")
        print(f"    baseline: {bt[:80]!r}")
        print(f"    spec:     {st[:80]!r}")
        print(f"    edit(baseline,gt): {ed_bl_gt}  edit(spec,gt): {ed_sp_gt}")


if __name__ == "__main__":
    main()
