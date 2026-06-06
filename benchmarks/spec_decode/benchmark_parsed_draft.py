#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Fidelity benchmark: vLLM ParsedDraftProposer vs EAGLE reference.

Runs both implementations' LCS utilities on the same pre-tokenized
OCR blocks and verifies they produce identical results. Also reports
acceptance rates and speedup estimates.

Usage:
    .venv/bin/python bench/bench_parsed_draft_fidelity.py \
        --data ../EAGLE/bench/sft_ocr_blocks_10k.tokenized.json \
        --n 100
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# ── Imports: vLLM implementation ──────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vllm.v1.spec_decode.parsed_draft import (  # noqa: E402
    IncrementalLCSMatcher,
    _lcs_draft_advance,
)

# ── Imports: EAGLE reference implementation ───────────────────────────────
EAGLE_ROOT = Path(__file__).resolve().parent.parent.parent / "EAGLE"
sys.path.insert(0, str(EAGLE_ROOT))

from eagle.model.parsed_draft_spec_decode import (  # noqa: E402
    IncrementalLCSMatcher as EagleLCSMatcher,
)
from eagle.model.parsed_draft_spec_decode import (  # noqa: E402
    _lcs_draft_advance as eagle_lcs_draft_advance,
)


# ── Simulation: stop-at-first-mismatch with LCS cursor ───────────────────
def simulate_stop_at_first(
    draft_ids: list[int],
    gt_ids: list[int],
    chunk_size: int,
    lcs_advance_fn,
    lcs_matcher_cls,
) -> dict:
    """Simulate stop-at-first-mismatch decoding.

    Returns dict with accepted count, verify steps, tokens produced.
    """
    n_draft = len(draft_ids)
    n_gt = len(gt_ids)
    if n_draft == 0 or n_gt == 0:
        return {
            "accepted": 0,
            "steps": 0,
            "tokens_produced": 0,
            "n_gt": n_gt,
        }

    total_accepted = 0
    total_steps = 0
    total_tokens = 0
    draft_cursor = 0
    gt_cursor = 0

    while draft_cursor < n_draft and gt_cursor < n_gt:
        chunk_d = draft_ids[draft_cursor : draft_cursor + chunk_size]
        chunk_g = gt_ids[gt_cursor : gt_cursor + chunk_size]
        chunk_len = min(len(chunk_d), len(chunk_g))
        if chunk_len == 0:
            break

        matcher = lcs_matcher_cls(chunk_d)
        acc_spans, _ = matcher.add_chunk(chunk_g[:chunk_len])

        if not acc_spans or acc_spans[0][0] != 0:
            # No prefix match — correction only
            total_tokens += 1
            gt_prefix = chunk_g[:1]
            advance = lcs_advance_fn(chunk_d, gt_prefix)
            draft_cursor += max(advance, 1)
            gt_cursor += 1
        else:
            prefix_len = acc_spans[0][1]
            if prefix_len >= chunk_len:
                # Entire chunk matched
                total_accepted += prefix_len
                total_tokens += prefix_len
                draft_cursor += len(chunk_d)
                gt_cursor += prefix_len
            else:
                # Accept prefix + 1 correction
                total_accepted += prefix_len
                total_tokens += prefix_len + 1
                gt_prefix = chunk_g[: prefix_len + 1]
                advance = lcs_advance_fn(chunk_d, gt_prefix)
                draft_cursor += max(advance, 1)
                gt_cursor += prefix_len + 1

        total_steps += 1

    return {
        "accepted": total_accepted,
        "steps": total_steps,
        "tokens_produced": total_tokens,
        "n_gt": n_gt,
    }


# ── Simulation: hybrid with bail-out ─────────────────────────────────────
def simulate_hybrid(
    draft_ids: list[int],
    gt_ids: list[int],
    chunk_size: int,
    max_reject: int,
    lcs_advance_fn,
    lcs_matcher_cls,
) -> dict:
    """Simulate hybrid (whole-chunk with bail-out) decoding."""
    n_draft = len(draft_ids)
    n_gt = len(gt_ids)
    if n_draft == 0 or n_gt == 0:
        return {
            "accepted": 0,
            "steps": 0,
            "tokens_produced": 0,
            "n_gt": n_gt,
        }

    total_accepted = 0
    total_steps = 0
    total_tokens = 0
    draft_cursor = 0
    gt_cursor = 0

    while draft_cursor < n_draft and gt_cursor < n_gt:
        chunk_d = draft_ids[draft_cursor : draft_cursor + chunk_size]
        chunk_g = gt_ids[gt_cursor : gt_cursor + chunk_size]
        chunk_len = min(len(chunk_d), len(chunk_g))
        if chunk_len == 0:
            break

        matcher = lcs_matcher_cls(chunk_d[:chunk_len])
        acc_spans, _ = matcher.add_chunk(chunk_g[:chunk_len])

        matched = set()
        for s, e in acc_spans:
            for pos in range(s, e):
                matched.add(pos)

        n_output = 0
        n_accepted = 0
        consec_reject = 0
        bail_pos = chunk_len

        for pos in range(chunk_len):
            if pos in matched:
                n_accepted += 1
                n_output += 1
                consec_reject = 0
            else:
                consec_reject += 1
                if consec_reject >= max_reject:
                    bail_pos = pos - max_reject + 1 + 1
                    n_output = bail_pos
                    n_accepted = sum(1 for p in range(bail_pos - 1) if p in matched)
                    break
                else:
                    n_output += 1

        total_accepted += n_accepted
        total_tokens += n_output
        total_steps += 1

        if bail_pos < chunk_len:
            gt_prefix = chunk_g[:bail_pos]
            advance = lcs_advance_fn(chunk_d, gt_prefix) if bail_pos > 0 else 1
            draft_cursor += max(advance, 1)
            gt_cursor += bail_pos
        else:
            draft_cursor += len(chunk_d)
            gt_cursor += chunk_len

    return {
        "accepted": total_accepted,
        "steps": total_steps,
        "tokens_produced": total_tokens,
        "n_gt": n_gt,
    }


# ── Load data ─────────────────────────────────────────────────────────────
def load_blocks(path: str, n: int | None = None) -> list[dict]:
    with open(path) as f:
        data = json.load(f)

    blocks = data["blocks"]
    # Filter to blocks with meaningful content
    blocks = [
        b
        for b in blocks
        if len(b.get("parsed_ids", [])) > 0 and len(b.get("gt_ids", [])) > 0
    ]
    if n is not None:
        blocks = blocks[:n]
    return blocks


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Fidelity benchmark: vLLM vs EAGLE parsed-draft LCS"
    )
    parser.add_argument(
        "--data",
        default=str(EAGLE_ROOT / "bench" / "sft_ocr_blocks_10k.tokenized.json"),
        help="Path to pre-tokenized benchmark data",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=100,
        help="Number of blocks to benchmark (default: 100)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=16,
        help="Draft chunk size in tokens (default: 16, matching vLLM config)",
    )
    parser.add_argument(
        "--max-reject",
        type=int,
        default=3,
        help="Max consecutive rejects for hybrid strategy (default: 3)",
    )
    args = parser.parse_args()

    print(f"Loading data from {args.data}")
    blocks = load_blocks(args.data, args.n)
    print(f"Loaded {len(blocks)} blocks\n")

    chunk_sizes = [args.chunk_size, 50, 200]
    strategies = [
        ("stop_at_first", None),
        ("hybrid_mr3", 3),
    ]

    # ── Run both implementations ──────────────────────────────────────
    for cs in chunk_sizes:
        for strategy_name, max_reject in strategies:
            print(f"{'=' * 70}")
            print(f"Strategy: {strategy_name}  |  chunk_size={cs}")
            print(f"{'=' * 70}")

            vllm_stats = {
                "accepted": 0,
                "steps": 0,
                "tokens": 0,
                "n_gt": 0,
            }
            eagle_stats = {
                "accepted": 0,
                "steps": 0,
                "tokens": 0,
                "n_gt": 0,
            }
            mismatches = 0

            t_vllm = 0.0
            t_eagle = 0.0

            for i, block in enumerate(blocks):
                draft_ids = block["parsed_ids"]
                gt_ids = block["gt_ids"]

                if max_reject is None:
                    # stop_at_first
                    t0 = time.perf_counter()
                    v = simulate_stop_at_first(
                        draft_ids,
                        gt_ids,
                        cs,
                        _lcs_draft_advance,
                        IncrementalLCSMatcher,
                    )
                    t_vllm += time.perf_counter() - t0

                    t0 = time.perf_counter()
                    e = simulate_stop_at_first(
                        draft_ids,
                        gt_ids,
                        cs,
                        eagle_lcs_draft_advance,
                        EagleLCSMatcher,
                    )
                    t_eagle += time.perf_counter() - t0
                else:
                    # hybrid
                    t0 = time.perf_counter()
                    v = simulate_hybrid(
                        draft_ids,
                        gt_ids,
                        cs,
                        max_reject,
                        _lcs_draft_advance,
                        IncrementalLCSMatcher,
                    )
                    t_vllm += time.perf_counter() - t0

                    t0 = time.perf_counter()
                    e = simulate_hybrid(
                        draft_ids,
                        gt_ids,
                        cs,
                        max_reject,
                        eagle_lcs_draft_advance,
                        EagleLCSMatcher,
                    )
                    t_eagle += time.perf_counter() - t0

                # Accumulate
                for key in ("accepted", "steps", "n_gt"):
                    vllm_stats[key] += v[key]
                    eagle_stats[key] += e[key]
                vllm_stats["tokens"] += v["tokens_produced"]
                eagle_stats["tokens"] += e["tokens_produced"]

                # Check fidelity
                if (
                    v["accepted"] != e["accepted"]
                    or v["steps"] != e["steps"]
                    or v["tokens_produced"] != e["tokens_produced"]
                ):
                    mismatches += 1
                    if mismatches <= 3:
                        print(
                            f"  MISMATCH block {i}: "
                            f"vllm(acc={v['accepted']},steps={v['steps']},"
                            f"tok={v['tokens_produced']}) vs "
                            f"eagle(acc={e['accepted']},steps={e['steps']},"
                            f"tok={e['tokens_produced']})"
                        )

            # ── Report ────────────────────────────────────────────────
            n = len(blocks)
            v_gt = vllm_stats["n_gt"]
            e_gt = eagle_stats["n_gt"]

            v_rate = vllm_stats["accepted"] / v_gt * 100 if v_gt else 0
            e_rate = eagle_stats["accepted"] / e_gt * 100 if e_gt else 0
            v_tps = (
                vllm_stats["tokens"] / vllm_stats["steps"] if vllm_stats["steps"] else 0
            )
            e_tps = (
                eagle_stats["tokens"] / eagle_stats["steps"]
                if eagle_stats["steps"]
                else 0
            )
            v_speedup = v_gt / vllm_stats["steps"] if vllm_stats["steps"] else 0
            e_speedup = e_gt / eagle_stats["steps"] if eagle_stats["steps"] else 0

            print(f"\n  {'':20s} {'vLLM':>12s} {'EAGLE':>12s} {'Match':>8s}")
            print(f"  {'─' * 55}")
            print(
                f"  {'Accepted tokens':20s} "
                f"{vllm_stats['accepted']:>12d} "
                f"{eagle_stats['accepted']:>12d} "
                f"{'✓' if vllm_stats['accepted'] == eagle_stats['accepted'] else '✗':>8s}"  # noqa: E501
            )
            print(
                f"  {'Accept rate':20s} "
                f"{v_rate:>11.1f}% "
                f"{e_rate:>11.1f}% "
                f"{'✓' if abs(v_rate - e_rate) < 0.01 else '✗':>8s}"
            )
            print(
                f"  {'Verify steps':20s} "
                f"{vllm_stats['steps']:>12d} "
                f"{eagle_stats['steps']:>12d} "
                f"{'✓' if vllm_stats['steps'] == eagle_stats['steps'] else '✗':>8s}"
            )
            print(
                f"  {'Avg tokens/step':20s} "
                f"{v_tps:>12.1f} "
                f"{e_tps:>12.1f} "
                f"{'✓' if abs(v_tps - e_tps) < 0.01 else '✗':>8s}"
            )
            print(
                f"  {'Effective speedup':20s} "
                f"{v_speedup:>11.1f}x "
                f"{e_speedup:>11.1f}x "
                f"{'✓' if abs(v_speedup - e_speedup) < 0.01 else '✗':>8s}"
            )
            print(
                f"  {'Wall-clock (ms)':20s} "
                f"{t_vllm * 1000:>11.1f}  "
                f"{t_eagle * 1000:>11.1f}"
            )
            print(f"  {'Block mismatches':20s} {mismatches:>12d} / {n}")
            print()

    # ── Final verdict ─────────────────────────────────────────────────
    print(f"{'=' * 70}")
    if mismatches == 0:
        print("FIDELITY CHECK PASSED: vLLM and EAGLE produce identical results")
    else:
        print(f"FIDELITY CHECK FAILED: {mismatches} block(s) differ")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
