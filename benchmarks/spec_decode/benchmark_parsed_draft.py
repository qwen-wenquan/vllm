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

from eagle_parsed_draft_spec_decode import (  # noqa: E402
    IncrementalLCSMatcher as EagleLCSMatcher,
)
from eagle_parsed_draft_spec_decode import (
    _find_draft_skip as eagle_find_draft_skip,
)
from eagle_parsed_draft_spec_decode import (
    _lcs_draft_advance as eagle_lcs_draft_advance,
)

from vllm.v1.spec_decode.parsed_draft import (  # noqa: E402
    IncrementalLCSMatcher,
    _find_draft_skip,
    _lcs_draft_advance,
)


# ── Simulation: stop-at-first-mismatch with LCS cursor ───────────────────
def simulate_stop_at_first(
    draft_ids: list[int],
    gt_ids: list[int],
    chunk_size: int,
    lcs_advance_fn,
    lcs_matcher_cls,
    find_draft_skip_fn,
) -> dict:
    """Simulate stop-at-first-mismatch decoding, mirroring EAGLE's
    ``_decode_stop_at_first_inner``.

    Each spec step is one verifier forward pass over a draft chunk. After
    the draft is exhausted we fall back to autoregressive decoding (1
    forward pass per remaining gt token) until the full ``gt_ids``
    sequence has been produced — that's what the real decoder does.

    Returns:
        Dict with:
            accepted          — total spec-accepted tokens (excludes
                                corrections and AR-fallback tokens).
            spec_steps        — number of verifier forward passes during
                                speculative phase.
            ar_steps          — number of forward passes during AR fallback
                                (= n_gt - tokens_produced_in_spec).
            forward_passes    — spec_steps + ar_steps. THE denominator
                                for true wall-clock speedup.
            tokens_produced   — tokens emitted during the spec phase
                                (== final gt_cursor before AR fallback).
            n_gt              — total ground-truth output length.
    """
    n_draft = len(draft_ids)
    n_gt = len(gt_ids)
    if n_draft == 0 or n_gt == 0:
        return {
            "accepted": 0,
            "spec_steps": 0,
            "ar_steps": n_gt,
            "forward_passes": n_gt,
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
            # No prefix match — take the correction (gt[0]) and advance
            # the draft cursor by locating that correction in the chunk
            # (EAGLE uses _find_draft_skip here, NOT _lcs_draft_advance).
            correction = chunk_g[0]
            total_tokens += 1
            skip = find_draft_skip_fn(chunk_d, correction, 0)
            draft_cursor += max(skip, 1)
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

    # AR fallback for remaining gt tokens (one forward pass each).
    ar_steps = max(0, n_gt - gt_cursor)

    return {
        "accepted": total_accepted,
        "spec_steps": total_steps,
        "ar_steps": ar_steps,
        "forward_passes": total_steps + ar_steps,
        "tokens_produced": total_tokens,
        "n_gt": n_gt,
    }


# ── Simulation: stop-at-first with hold+snap cursor rule ─────────────────
def simulate_stop_at_first_holdsnap(
    draft_ids: list[int],
    gt_ids: list[int],
    chunk_size: int,
    lcs_advance_fn,
    lcs_matcher_cls,
    max_hold: int = 8,
) -> dict:
    """Variant of stop_at_first with a smarter no-prefix-match cursor rule.

    The baseline ``simulate_stop_at_first`` calls ``_find_draft_skip``
    when LCS finds no prefix match. That function advances the draft
    cursor by 1 when the correction token is not in the draft chunk,
    which burns through draft tokens during gt-side insertions
    (e.g. the model emits a few extra formatting tokens). By the time
    gt rejoins the draft text, the matching draft tokens are gone.

    Hold + snap rule (no-prefix-match branch only):
      - Search the current draft chunk for the correction token.
      - FOUND at position i: SNAP — advance cursor by i, so the next
        spec step starts with the correction token at position 0
        (the verifier can then confirm it as a real prefix match).
      - NOT FOUND: HOLD — leave cursor parked, re-verify the same
        draft chunk against the shifted gt window next step.
      - Safety: after ``max_hold`` consecutive holds, advance by 1
        to avoid stalling when draft and gt have no overlap.

    Returns the same dict shape as ``simulate_stop_at_first``.
    """
    n_draft = len(draft_ids)
    n_gt = len(gt_ids)
    if n_draft == 0 or n_gt == 0:
        return {
            "accepted": 0,
            "spec_steps": 0,
            "ar_steps": n_gt,
            "forward_passes": n_gt,
            "tokens_produced": 0,
            "n_gt": n_gt,
        }

    total_accepted = 0
    total_steps = 0
    total_tokens = 0
    draft_cursor = 0
    gt_cursor = 0
    consecutive_holds = 0

    while draft_cursor < n_draft and gt_cursor < n_gt:
        chunk_d = draft_ids[draft_cursor : draft_cursor + chunk_size]
        chunk_g = gt_ids[gt_cursor : gt_cursor + chunk_size]
        chunk_len = min(len(chunk_d), len(chunk_g))
        if chunk_len == 0:
            break

        matcher = lcs_matcher_cls(chunk_d)
        acc_spans, _ = matcher.add_chunk(chunk_g[:chunk_len])

        if not acc_spans or acc_spans[0][0] != 0:
            # No prefix match — apply hold+snap rule.
            correction = chunk_g[0]
            total_tokens += 1

            snap_pos = -1
            for i, tok in enumerate(chunk_d):
                if tok == correction:
                    snap_pos = i
                    break

            if snap_pos >= 0:
                # SNAP: jump TO the correction so it becomes the next
                # chunk's position-0 token (verifier can confirm as prefix).
                draft_cursor += snap_pos
                consecutive_holds = 0
            else:
                # HOLD: don't advance — gt-side insertion in progress.
                if consecutive_holds >= max_hold:
                    draft_cursor += 1
                    consecutive_holds = 0
                else:
                    consecutive_holds += 1
                    # draft_cursor unchanged
            gt_cursor += 1
        else:
            prefix_len = acc_spans[0][1]
            consecutive_holds = 0
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

    ar_steps = max(0, n_gt - gt_cursor)

    return {
        "accepted": total_accepted,
        "spec_steps": total_steps,
        "ar_steps": ar_steps,
        "forward_passes": total_steps + ar_steps,
        "tokens_produced": total_tokens,
        "n_gt": n_gt,
    }


# ── Simulation: adaptive chunk-size + holdsnap ───────────────────────────
def simulate_holdsnap_adaptive(
    draft_ids: list[int],
    gt_ids: list[int],
    initial_chunk_size: int,
    lcs_advance_fn,
    lcs_matcher_cls,
    grow_factor: int = 2,
    shrink_factor: int = 2,
    max_chunk_multiplier: int = 2,
    recover_to_initial: bool = False,
) -> dict:
    """Adaptive chunk-size variant of holdsnap (lossless).

    Same cursor rule as ``simulate_stop_at_first_holdsnap``, but the
    chunk size adapts after each step based on the verifier outcome,
    and ``max_hold`` is re-evaluated each step as the current chunk
    size.

    Rules:
      - Initial chunk size = ``initial_chunk_size``.
      - Bounds: ``min = 2``,
        ``max = max_chunk_multiplier * initial_chunk_size``.
      - After a "nothing hit" step (no-prefix-match branch, i.e. the
        first draft token didn't match the first gt token):
        ``cs //= shrink_factor`` (clamped at min).
      - After a "prefix hit" step (any prefix_len >= 1):
        - If ``recover_to_initial``: jump cs back to
          ``initial_chunk_size`` (any partial match is treated as a
          signal that we re-aligned and can restore the original
          lookahead window).
        - Else, the default behavior is split by hit type:
            * full-chunk hit  → ``cs *= grow_factor`` (clamped at max).
            * partial hit     → keep cs unchanged.
      - ``max_hold`` = current chunk size, re-evaluated each step.

    Returns same dict shape as ``simulate_stop_at_first``, plus:
      - mean_cs: mean chunk size actually used across spec steps.
      - max_cs_observed / min_cs_observed: extremes used.
    """
    n_draft = len(draft_ids)
    n_gt = len(gt_ids)
    if n_draft == 0 or n_gt == 0:
        return {
            "accepted": 0,
            "spec_steps": 0,
            "ar_steps": n_gt,
            "forward_passes": n_gt,
            "tokens_produced": 0,
            "n_gt": n_gt,
            "mean_cs": 0.0,
            "min_cs_observed": 0,
            "max_cs_observed": 0,
        }

    cs_min = 2
    cs_max = max_chunk_multiplier * initial_chunk_size
    cs = max(cs_min, min(initial_chunk_size, cs_max))

    total_accepted = 0
    total_steps = 0
    total_tokens = 0
    draft_cursor = 0
    gt_cursor = 0
    consecutive_holds = 0
    cs_sum = 0
    cs_obs_min = cs
    cs_obs_max = cs

    while draft_cursor < n_draft and gt_cursor < n_gt:
        chunk_d = draft_ids[draft_cursor : draft_cursor + cs]
        chunk_g = gt_ids[gt_cursor : gt_cursor + cs]
        chunk_len = min(len(chunk_d), len(chunk_g))
        if chunk_len == 0:
            break

        cs_sum += cs
        cs_obs_min = min(cs_obs_min, cs)
        cs_obs_max = max(cs_obs_max, cs)
        max_hold = cs

        matcher = lcs_matcher_cls(chunk_d)
        acc_spans, _ = matcher.add_chunk(chunk_g[:chunk_len])

        if not acc_spans or acc_spans[0][0] != 0:
            # No prefix match — apply hold+snap rule.
            correction = chunk_g[0]
            total_tokens += 1

            snap_pos = -1
            for i, tok in enumerate(chunk_d):
                if tok == correction:
                    snap_pos = i
                    break

            if snap_pos >= 0:
                draft_cursor += snap_pos
                consecutive_holds = 0
            else:
                if consecutive_holds >= max_hold:
                    draft_cursor += 1
                    consecutive_holds = 0
                else:
                    consecutive_holds += 1
            gt_cursor += 1
            # "Nothing hit" → shrink chunk
            cs = max(cs_min, cs // shrink_factor)
        else:
            prefix_len = acc_spans[0][1]
            consecutive_holds = 0
            if prefix_len >= chunk_len:
                # Entire chunk matched → grow chunk
                total_accepted += prefix_len
                total_tokens += prefix_len
                draft_cursor += len(chunk_d)
                gt_cursor += prefix_len
                if recover_to_initial:
                    cs = max(cs_min, min(initial_chunk_size, cs_max))
                else:
                    cs = min(cs_max, cs * grow_factor)
            else:
                # Partial match → keep chunk size (or recover)
                total_accepted += prefix_len
                total_tokens += prefix_len + 1
                gt_prefix = chunk_g[: prefix_len + 1]
                advance = lcs_advance_fn(chunk_d, gt_prefix)
                draft_cursor += max(advance, 1)
                gt_cursor += prefix_len + 1
                if recover_to_initial:
                    cs = max(cs_min, min(initial_chunk_size, cs_max))

        total_steps += 1

    ar_steps = max(0, n_gt - gt_cursor)
    mean_cs = cs_sum / total_steps if total_steps else 0.0

    return {
        "accepted": total_accepted,
        "spec_steps": total_steps,
        "ar_steps": ar_steps,
        "forward_passes": total_steps + ar_steps,
        "tokens_produced": total_tokens,
        "n_gt": n_gt,
        "mean_cs": mean_cs,
        "min_cs_observed": cs_obs_min,
        "max_cs_observed": cs_obs_max,
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
    """Simulate hybrid (whole-chunk with bail-out) decoding.

    Same return shape as ``simulate_stop_at_first``. After draft is
    exhausted, falls back to autoregressive decoding (one forward pass
    per remaining gt token).
    """
    n_draft = len(draft_ids)
    n_gt = len(gt_ids)
    if n_draft == 0 or n_gt == 0:
        return {
            "accepted": 0,
            "spec_steps": 0,
            "ar_steps": n_gt,
            "forward_passes": n_gt,
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

    ar_steps = max(0, n_gt - gt_cursor)

    return {
        "accepted": total_accepted,
        "spec_steps": total_steps,
        "ar_steps": ar_steps,
        "forward_passes": total_steps + ar_steps,
        "tokens_produced": total_tokens,
        "n_gt": n_gt,
    }


# ── Load data ─────────────────────────────────────────────────────────────
def load_blocks(path: str, n: int | None = None) -> list[dict]:
    """Load blocks from either a JSON (legacy) or JSONL (pretokenized) file.

    JSON format: ``{"blocks": [...], ...}`` — read all at once.
    JSONL format: one block per line — streamed, can early-exit at ``n``.
    Detection: ``.jsonl`` extension → JSONL; anything else → JSON.
    """
    if path.endswith(".jsonl"):
        blocks = []
        with open(path) as f:
            for line in f:
                b = json.loads(line)
                if len(b.get("parsed_ids", [])) > 0 and len(b.get("gt_ids", [])) > 0:
                    blocks.append(b)
                    if n is not None and len(blocks) >= n:
                        break
        return blocks

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
        default="data/sft_ocr_blocks_10k.normalized.jsonl",
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
    # (display_name, kind, param)
    #   kind = "stop"      → simulate_stop_at_first (production baseline)
    #   kind = "hybrid"    → simulate_hybrid (param = max_reject)
    #   kind = "holdsnap"  → simulate_stop_at_first_holdsnap
    #                        (param = max_hold; lossless cursor improvement
    #                        over the production baseline)
    strategies = [
        ("stop_at_first", "stop", None),
        ("hybrid_mr3", "hybrid", 3),
        ("holdsnap_h8", "holdsnap", 8),
    ]

    # ── Run both implementations ──────────────────────────────────────
    for cs in chunk_sizes:
        for strategy_name, kind, param in strategies:
            print(f"{'=' * 70}")
            print(f"Strategy: {strategy_name}  |  chunk_size={cs}")
            print(f"{'=' * 70}")

            vllm_stats = {
                "accepted": 0,
                "spec_steps": 0,
                "ar_steps": 0,
                "forward_passes": 0,
                "tokens": 0,
                "n_gt": 0,
            }
            eagle_stats = {
                "accepted": 0,
                "spec_steps": 0,
                "ar_steps": 0,
                "forward_passes": 0,
                "tokens": 0,
                "n_gt": 0,
            }
            # Adaptive-only diagnostics
            adaptive_cs_sum = 0.0
            adaptive_cs_min = None
            adaptive_cs_max = None
            mismatches = 0

            t_vllm = 0.0
            t_eagle = 0.0

            for i, block in enumerate(blocks):
                draft_ids = block["parsed_ids"]
                gt_ids = block["gt_ids"]

                if kind == "stop":
                    t0 = time.perf_counter()
                    v = simulate_stop_at_first(
                        draft_ids,
                        gt_ids,
                        cs,
                        _lcs_draft_advance,
                        IncrementalLCSMatcher,
                        _find_draft_skip,
                    )
                    t_vllm += time.perf_counter() - t0

                    t0 = time.perf_counter()
                    e = simulate_stop_at_first(
                        draft_ids,
                        gt_ids,
                        cs,
                        eagle_lcs_draft_advance,
                        EagleLCSMatcher,
                        eagle_find_draft_skip,
                    )
                    t_eagle += time.perf_counter() - t0
                elif kind == "hybrid":
                    t0 = time.perf_counter()
                    v = simulate_hybrid(
                        draft_ids,
                        gt_ids,
                        cs,
                        param,
                        _lcs_draft_advance,
                        IncrementalLCSMatcher,
                    )
                    t_vllm += time.perf_counter() - t0

                    t0 = time.perf_counter()
                    e = simulate_hybrid(
                        draft_ids,
                        gt_ids,
                        cs,
                        param,
                        eagle_lcs_draft_advance,
                        EagleLCSMatcher,
                    )
                    t_eagle += time.perf_counter() - t0
                elif kind == "holdsnap":
                    # Experimental cursor rule (not in production yet).
                    # Run with both LCS impls so the fidelity columns still
                    # validate matcher equivalence.
                    t0 = time.perf_counter()
                    v = simulate_stop_at_first_holdsnap(
                        draft_ids,
                        gt_ids,
                        cs,
                        _lcs_draft_advance,
                        IncrementalLCSMatcher,
                        max_hold=param,
                    )
                    t_vllm += time.perf_counter() - t0

                    t0 = time.perf_counter()
                    e = simulate_stop_at_first_holdsnap(
                        draft_ids,
                        gt_ids,
                        cs,
                        eagle_lcs_draft_advance,
                        EagleLCSMatcher,
                        max_hold=param,
                    )
                    t_eagle += time.perf_counter() - t0
                elif kind == "adaptive":
                    # Adaptive chunk-size + holdsnap (experimental, lossless).
                    # cs adapts in [2, max_chunk_multiplier*initial] based on
                    # hit/miss outcome.
                    kwargs = param or {}
                    t0 = time.perf_counter()
                    v = simulate_holdsnap_adaptive(
                        draft_ids,
                        gt_ids,
                        cs,
                        _lcs_draft_advance,
                        IncrementalLCSMatcher,
                        **kwargs,
                    )
                    t_vllm += time.perf_counter() - t0

                    t0 = time.perf_counter()
                    e = simulate_holdsnap_adaptive(
                        draft_ids,
                        gt_ids,
                        cs,
                        eagle_lcs_draft_advance,
                        EagleLCSMatcher,
                        **kwargs,
                    )
                    t_eagle += time.perf_counter() - t0
                else:
                    raise ValueError(f"unknown strategy kind: {kind}")

                # Accumulate
                for key in (
                    "accepted",
                    "spec_steps",
                    "ar_steps",
                    "forward_passes",
                    "n_gt",
                ):
                    vllm_stats[key] += v[key]
                    eagle_stats[key] += e[key]
                vllm_stats["tokens"] += v["tokens_produced"]
                eagle_stats["tokens"] += e["tokens_produced"]

                # Adaptive diagnostics (only present when kind == "adaptive")
                if kind == "adaptive" and v["spec_steps"] > 0:
                    adaptive_cs_sum += v["mean_cs"] * v["spec_steps"]
                    if adaptive_cs_min is None:
                        adaptive_cs_min = v["min_cs_observed"]
                        adaptive_cs_max = v["max_cs_observed"]
                    else:
                        adaptive_cs_min = min(adaptive_cs_min, v["min_cs_observed"])
                        adaptive_cs_max = max(adaptive_cs_max, v["max_cs_observed"])

                # Check fidelity
                if (
                    v["accepted"] != e["accepted"]
                    or v["spec_steps"] != e["spec_steps"]
                    or v["tokens_produced"] != e["tokens_produced"]
                ):
                    mismatches += 1
                    if mismatches <= 3:
                        print(
                            f"  MISMATCH block {i}: "
                            f"vllm(acc={v['accepted']},"
                            f"steps={v['spec_steps']},"
                            f"tok={v['tokens_produced']}) vs "
                            f"eagle(acc={e['accepted']},"
                            f"steps={e['spec_steps']},"
                            f"tok={e['tokens_produced']})"
                        )

            # ── Report ────────────────────────────────────────────────
            n = len(blocks)
            v_gt = vllm_stats["n_gt"]
            e_gt = eagle_stats["n_gt"]

            v_rate = vllm_stats["accepted"] / v_gt * 100 if v_gt else 0
            e_rate = eagle_stats["accepted"] / e_gt * 100 if e_gt else 0
            v_tps = (
                vllm_stats["tokens"] / vllm_stats["spec_steps"]
                if vllm_stats["spec_steps"]
                else 0
            )
            e_tps = (
                eagle_stats["tokens"] / eagle_stats["spec_steps"]
                if eagle_stats["spec_steps"]
                else 0
            )
            # Honest end-to-end speedup vs pure AR baseline:
            # baseline takes n_gt forward passes; speculative takes
            # spec_steps + ar_steps. The previous metric (n_gt / spec_steps)
            # silently ignored the AR-fallback work needed to produce the
            # tokens not covered by the draft.
            v_speedup = (
                v_gt / vllm_stats["forward_passes"]
                if vllm_stats["forward_passes"]
                else 0
            )
            e_speedup = (
                e_gt / eagle_stats["forward_passes"]
                if eagle_stats["forward_passes"]
                else 0
            )

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
                f"  {'Spec steps':20s} "
                f"{vllm_stats['spec_steps']:>12d} "
                f"{eagle_stats['spec_steps']:>12d} "
                f"{'✓' if vllm_stats['spec_steps'] == eagle_stats['spec_steps'] else '✗':>8s}"  # noqa: E501
            )
            print(
                f"  {'AR fallback steps':20s} "
                f"{vllm_stats['ar_steps']:>12d} "
                f"{eagle_stats['ar_steps']:>12d} "
                f"{'✓' if vllm_stats['ar_steps'] == eagle_stats['ar_steps'] else '✗':>8s}"  # noqa: E501
            )
            print(
                f"  {'Forward passes':20s} "
                f"{vllm_stats['forward_passes']:>12d} "
                f"{eagle_stats['forward_passes']:>12d} "
                f"{'✓' if vllm_stats['forward_passes'] == eagle_stats['forward_passes'] else '✗':>8s}"  # noqa: E501
            )
            print(
                f"  {'Avg tokens/spec step':20s} "
                f"{v_tps:>12.2f} "
                f"{e_tps:>12.2f} "
                f"{'✓' if abs(v_tps - e_tps) < 0.01 else '✗':>8s}"
            )
            print(
                f"  {'End-to-end speedup':20s} "
                f"{v_speedup:>11.2f}x "
                f"{e_speedup:>11.2f}x "
                f"{'✓' if abs(v_speedup - e_speedup) < 0.01 else '✗':>8s}"
            )
            print(
                f"  {'Wall-clock (ms)':20s} "
                f"{t_vllm * 1000:>11.1f}  "
                f"{t_eagle * 1000:>11.1f}"
            )
            print(f"  {'Block mismatches':20s} {mismatches:>12d} / {n}")
            if kind == "adaptive" and vllm_stats["spec_steps"] > 0:
                mean_cs = adaptive_cs_sum / vllm_stats["spec_steps"]
                kwargs = param or {}
                mult = kwargs.get("max_chunk_multiplier", 2)
                grow = kwargs.get("grow_factor", 2)
                shrink = kwargs.get("shrink_factor", 2)
                print(
                    f"  {'Adaptive cs':20s} "
                    f"mean={mean_cs:.2f}  "
                    f"range=[{adaptive_cs_min}, {adaptive_cs_max}]  "
                    f"bounds=[2, {mult * cs}]  "
                    f"grow=x{grow}/shrink=/{shrink}"
                )
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
