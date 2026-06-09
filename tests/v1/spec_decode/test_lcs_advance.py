# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for incremental LCS advance correctness and throughput.

Validates that _incremental_lcs_advance produces identical results
to _lcs_draft_advance for all inputs, then benchmarks throughput.
"""

import random
import time

import pytest

from vllm.v1.spec_decode.parsed_draft import (
    _incremental_lcs_advance,
    _lcs_draft_advance,
)

# ---------------------------------------------------------------------------
# Correctness: incremental vs original must agree
# ---------------------------------------------------------------------------


class TestIncrementalLcsAdvanceCorrectness:
    """Compare _incremental_lcs_advance against _lcs_draft_advance."""

    @staticmethod
    def _assert_equal(draft: list[int], gt: list[int]):
        expected = _lcs_draft_advance(draft, gt)
        actual = _incremental_lcs_advance(draft, gt)
        assert actual == expected, (
            f"Mismatch for draft={draft}, gt={gt}: expected={expected}, got={actual}"
        )

    # ── Edge cases ──────────────────────────────────────

    def test_both_empty(self):
        self._assert_equal([], [])

    def test_empty_draft(self):
        self._assert_equal([], [1, 2, 3])

    def test_empty_gt(self):
        self._assert_equal([1, 2, 3], [])

    def test_single_match(self):
        self._assert_equal([5], [5])

    def test_single_mismatch(self):
        self._assert_equal([5], [6])

    # ── Deterministic cases ─────────────────────────────

    def test_perfect_match(self):
        self._assert_equal([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])

    def test_no_match(self):
        self._assert_equal([1, 2, 3], [4, 5, 6])

    def test_insertion_in_draft(self):
        # draft has extra token X between B and C
        self._assert_equal([1, 99, 2, 3], [1, 2, 3])

    def test_multiple_insertions(self):
        self._assert_equal([1, 90, 91, 2, 92, 3], [1, 2, 3])

    def test_deletion_in_draft(self):
        # draft missing token 2
        self._assert_equal([1, 3, 4], [1, 2, 3, 4])

    def test_substitution(self):
        self._assert_equal([1, 2, 99, 4], [1, 2, 3, 4])

    def test_correction_at_end(self):
        self._assert_equal([1, 2, 90], [1, 2, 80])

    def test_single_match_at_start(self):
        self._assert_equal([1, 90, 91], [1, 80])

    def test_gt_longer_than_draft(self):
        self._assert_equal([1, 2], [1, 2, 3, 4, 5])

    def test_draft_longer_than_gt(self):
        self._assert_equal([1, 2, 3, 4, 5], [1, 2])

    def test_repeated_tokens(self):
        self._assert_equal([1, 1, 1, 2, 2], [1, 2, 1, 2])

    def test_interleaved(self):
        self._assert_equal([1, 90, 2, 91, 3], [1, 80, 2, 81, 3])

    # ── Randomized fuzz test ────────────────────────────

    @pytest.mark.parametrize("seed", range(200))
    def test_random_pair(self, seed):
        """Fuzz test: random draft/gt pairs, M=1-50, N=1-10."""
        rng = random.Random(seed)
        vocab = list(range(1, 20))  # small vocab to get matches
        m = rng.randint(1, 50)
        n = rng.randint(1, 10)
        draft = [rng.choice(vocab) for _ in range(m)]
        gt = [rng.choice(vocab) for _ in range(n)]
        self._assert_equal(draft, gt)

    @pytest.mark.parametrize("seed", range(50))
    def test_random_with_high_overlap(self, seed):
        """Fuzz with high overlap — draft and gt share a base."""
        rng = random.Random(seed + 1000)
        base_len = rng.randint(5, 30)
        base = [rng.randint(1, 10) for _ in range(base_len)]

        # Draft = base with random insertions
        draft = []
        for tok in base:
            if rng.random() < 0.3:
                draft.append(rng.randint(50, 60))  # insertion
            draft.append(tok)

        # GT = base with random substitutions
        gt = []
        for tok in base:
            if rng.random() < 0.2:
                gt.append(rng.randint(70, 80))  # substitution
            else:
                gt.append(tok)

        self._assert_equal(draft, gt)


# ---------------------------------------------------------------------------
# Throughput benchmark (not run by default)
# ---------------------------------------------------------------------------


class TestIncrementalLcsAdvanceThroughput:
    """Throughput benchmark — run with pytest -s to see output."""

    @pytest.mark.parametrize(
        "m,n",
        [(10, 3), (20, 5), (50, 10), (50, 5)],
    )
    def test_throughput_comparison(self, m, n):
        """Compare per-call latency of old vs new implementation."""
        rng = random.Random(42)
        vocab = list(range(1, 20))
        n_trials = 500
        pairs = [
            (
                [rng.choice(vocab) for _ in range(m)],
                [rng.choice(vocab) for _ in range(n)],
            )
            for _ in range(n_trials)
        ]

        # Warmup
        for draft, gt in pairs[:10]:
            _lcs_draft_advance(draft, gt)
            _incremental_lcs_advance(draft, gt)

        # Benchmark original
        t0 = time.perf_counter()
        for draft, gt in pairs:
            _lcs_draft_advance(draft, gt)
        t_orig = time.perf_counter() - t0

        # Benchmark incremental
        t0 = time.perf_counter()
        for draft, gt in pairs:
            _incremental_lcs_advance(draft, gt)
        t_incr = time.perf_counter() - t0

        us_orig = t_orig / n_trials * 1e6
        us_incr = t_incr / n_trials * 1e6
        ratio = t_orig / t_incr if t_incr > 0 else float("inf")

        print(
            f"\n  M={m}, N={n}: "
            f"original={us_orig:.1f}us, "
            f"incremental={us_incr:.1f}us, "
            f"speedup={ratio:.2f}x"
        )
