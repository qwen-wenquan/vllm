# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Pre-tokenize benchmark data for fast repeated benchmarking.

Tokenizes parsed_text and pred_text/gt_text in parallel using multiprocessing,
saves the token id lists alongside the original data. Subsequent benchmark
runs skip tokenization entirely.

Usage:
    .venv/bin/python benchmarks/spec_decode/pretokenize.py \\
        --input data/sft_ocr_blocks_10k.tokenized.json \\
        --tokenizer /path/to/PaddleOCR-VL_finetune \\
        --output data/sft_ocr_blocks_10k.normalized.jsonl \\
        --workers 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from vllm.v1.spec_decode.parsed_draft import normalize_parsed_text  # noqa: E402


def _init_worker(tokenizer_path: str):
    """Initialize tokenizer in each worker process."""
    global _worker_tokenizer
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    from transformers import AutoTokenizer

    _worker_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)


def _tokenize_one(args):
    block, normalize = args
    parsed = block["parsed_text"]
    if normalize:
        parsed = normalize_parsed_text(parsed)
    gt = block.get("gt_text") or block.get("pred_text", "")

    global _worker_tokenizer
    block["parsed_ids"] = _worker_tokenizer.encode(parsed, add_special_tokens=False)
    block["gt_ids"] = _worker_tokenizer.encode(gt, add_special_tokens=False)
    block["parsed_text_normalized"] = parsed
    return block


def main():
    parser = argparse.ArgumentParser(description="Pre-tokenize benchmark data")
    parser.add_argument(
        "--input", required=True, help="Input JSON (inference_results format)"
    )
    parser.add_argument("--tokenizer", required=True, help="Tokenizer name or path")
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL with token ids (one block per line)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of parallel workers (default: cpu_count)",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable normalize_parsed_text "
        "(by default, NFKC + hyphen + whitespace cleanup runs)",
    )
    args = parser.parse_args()

    n_workers = args.workers or cpu_count()
    normalize = not args.no_normalize

    print(f"Loading data: {args.input}")
    # Load raw JSON directly (not through load_blocks, to preserve structure)
    with open(args.input) as f:
        data = json.load(f)

    blocks = data["blocks"]
    print(f"Blocks: {len(blocks)}")
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Workers: {n_workers}")
    print(f"Normalize: {normalize}")

    # Prepare work items
    work = [(b, normalize) for b in blocks]

    t0 = time.time()
    with Pool(n_workers, initializer=_init_worker, initargs=(args.tokenizer,)) as pool:
        results = []
        for i, block in enumerate(pool.imap(_tokenize_one, work, chunksize=1000)):
            results.append(block)
            if (i + 1) % 50000 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                print(f"  {i + 1}/{len(blocks)} ({rate:.0f} blocks/s)")

    elapsed = time.time() - t0
    print(
        f"Tokenized {len(results)} blocks in {elapsed:.1f}s "
        f"({len(results) / elapsed:.0f} blocks/s)"
    )

    # Save as JSONL (one block per line) for fast streaming reads
    with open(args.output, "w", encoding="utf-8") as f:
        for block in results:
            f.write(json.dumps(block, ensure_ascii=False) + "\n")

    size_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f"Saved: {args.output} ({size_mb:.1f} MB, JSONL format)")


if __name__ == "__main__":
    main()
