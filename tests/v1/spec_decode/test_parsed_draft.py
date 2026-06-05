# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for parsed-draft speculative decoding components."""

from vllm.v1.spec_decode.parsed_draft import (
    IncrementalLCSMatcher,
    ParsedDraftProposer,
    ParsedDraftProvider,
    _find_draft_skip,
    _lcs_draft_advance,
    find_prefix_match,
)

# ---------------------------------------------------------------------------
# IncrementalLCSMatcher tests
# ---------------------------------------------------------------------------


class TestIncrementalLCSMatcher:
    def test_identical_sequences(self):
        matcher = IncrementalLCSMatcher([1, 2, 3, 4])
        accepted, rejected = matcher.add_chunk([1, 2, 3, 4])
        assert accepted == [(0, 4)]
        assert rejected == []

    def test_completely_different(self):
        matcher = IncrementalLCSMatcher([1, 2, 3])
        accepted, rejected = matcher.add_chunk([4, 5, 6])
        assert accepted == []
        assert rejected == [(0, 3)]

    def test_insertion_in_draft(self):
        # Draft has extra token X between B and C
        # draft: [A, B, X, C, D]  gt: [A, B, C, D]
        matcher = IncrementalLCSMatcher([10, 20, 99, 30, 40])
        accepted, rejected = matcher.add_chunk([10, 20, 30, 40])
        # LCS should match A, B, C, D (all 4 gt tokens)
        assert accepted == [(0, 4)]
        assert rejected == []

    def test_deletion_in_draft(self):
        # Draft missing token C that GT has
        # draft: [A, B, D]  gt: [A, B, C, D]
        matcher = IncrementalLCSMatcher([10, 20, 40])
        accepted, rejected = matcher.add_chunk([10, 20, 30, 40])
        # LCS matches A, B, D (3 of 4 gt tokens)
        # gt[2]=C is rejected
        total_matched = sum(e - s for s, e in accepted)
        assert total_matched == 3
        total_rejected = sum(e - s for s, e in rejected)
        assert total_rejected == 1

    def test_substitution(self):
        # draft: [A, B, X, D]  gt: [A, B, C, D]
        matcher = IncrementalLCSMatcher([10, 20, 99, 40])
        accepted, rejected = matcher.add_chunk([10, 20, 30, 40])
        total_matched = sum(e - s for s, e in accepted)
        assert total_matched == 3  # A, B, D matched
        total_rejected = sum(e - s for s, e in rejected)
        assert total_rejected == 1  # C not in draft

    def test_empty_draft(self):
        matcher = IncrementalLCSMatcher([])
        accepted, rejected = matcher.add_chunk([1, 2, 3])
        assert accepted == []
        assert rejected == [(0, 3)]

    def test_empty_gt_chunk(self):
        matcher = IncrementalLCSMatcher([1, 2, 3])
        accepted, rejected = matcher.add_chunk([])
        assert accepted == []
        assert rejected == []

    def test_single_token_match(self):
        matcher = IncrementalLCSMatcher([5])
        accepted, rejected = matcher.add_chunk([5])
        assert accepted == [(0, 1)]
        assert rejected == []

    def test_single_token_mismatch(self):
        matcher = IncrementalLCSMatcher([5])
        accepted, rejected = matcher.add_chunk([6])
        assert accepted == []
        assert rejected == [(0, 1)]

    def test_multiple_insertions(self):
        # draft: [A, X, Y, B, C]  gt: [A, B, C]
        matcher = IncrementalLCSMatcher([1, 90, 91, 2, 3])
        accepted, rejected = matcher.add_chunk([1, 2, 3])
        assert accepted == [(0, 3)]  # all gt tokens matched

    def test_reset_history(self):
        matcher = IncrementalLCSMatcher([1, 2, 3, 4, 5])
        matcher.add_chunk([1, 2, 3])
        matcher.reset_history()
        # dp_row is preserved, history is cleared
        assert len(matcher.dp_history) == 1
        assert len(matcher.gt_so_far) == 0

    def test_interleaved_accept_reject(self):
        # draft: [A, X, B, Y, C]  gt: [A, Z, B, W, C]
        matcher = IncrementalLCSMatcher([1, 90, 2, 91, 3])
        accepted, rejected = matcher.add_chunk([1, 80, 2, 81, 3])
        # LCS matches A(0), B(2), C(4) — 3 accepted
        total_matched = sum(e - s for s, e in accepted)
        assert total_matched == 3
        total_rejected = sum(e - s for s, e in rejected)
        assert total_rejected == 2


# ---------------------------------------------------------------------------
# find_prefix_match tests
# ---------------------------------------------------------------------------


class TestFindPrefixMatch:
    def test_full_match(self):
        import torch

        draft = [1, 2, 3, 4]
        logits = torch.zeros(4, 10)
        for i, t in enumerate(draft):
            logits[i, t] = 10.0  # make argmax match
        assert find_prefix_match(draft, logits) == 4

    def test_first_mismatch(self):
        import torch

        draft = [1, 2, 3]
        logits = torch.zeros(3, 10)
        logits[0, 5] = 10.0  # mismatch at pos 0
        assert find_prefix_match(draft, logits) == 0

    def test_partial_match(self):
        import torch

        draft = [1, 2, 3, 4]
        logits = torch.zeros(4, 10)
        logits[0, 1] = 10.0  # match
        logits[1, 2] = 10.0  # match
        logits[2, 9] = 10.0  # mismatch at pos 2
        logits[3, 4] = 10.0  # would match but we stop
        assert find_prefix_match(draft, logits) == 2

    def test_empty_draft(self):
        import torch

        assert find_prefix_match([], torch.zeros(5, 10)) == 0

    def test_empty_logits(self):
        import torch

        assert find_prefix_match([1, 2], torch.zeros(0, 10)) == 0


# ---------------------------------------------------------------------------
# _lcs_draft_advance tests
# ---------------------------------------------------------------------------


class TestLcsDraftAdvance:
    def test_perfect_match(self):
        # All matched, advance past all
        assert _lcs_draft_advance([1, 2, 3], [1, 2, 3]) == 3

    def test_with_insertion(self):
        # draft: [A, X, B, C]  gt: [A, B, C]
        # LCS matches A(0), B(2), C(3) → advance = 3 + 1 = 4
        result = _lcs_draft_advance([1, 99, 2, 3], [1, 2, 3])
        assert result == 4  # past last matched draft pos

    def test_correction_at_end(self):
        # draft: [A, B, X]  gt: [A, B, Y]
        # LCS matches A(0), B(1) → advance = 1 + 1 = 2
        result = _lcs_draft_advance([1, 2, 90], [1, 2, 80])
        assert result == 2

    def test_no_match(self):
        # No tokens match at all
        result = _lcs_draft_advance([1, 2, 3], [4, 5, 6])
        assert result == 1  # fallback minimum

    def test_empty_draft(self):
        assert _lcs_draft_advance([], [1, 2]) == 1

    def test_empty_gt(self):
        assert _lcs_draft_advance([1, 2], []) == 1

    def test_single_match_at_start(self):
        # draft: [A, X, Y]  gt: [A, Z]
        # LCS matches A(0) → advance = 0 + 1 = 1
        result = _lcs_draft_advance([1, 90, 91], [1, 80])
        assert result == 1


# ---------------------------------------------------------------------------
# _find_draft_skip tests
# ---------------------------------------------------------------------------


class TestFindDraftSkip:
    def test_correction_found(self):
        # Correction token 30 found at index 2
        assert _find_draft_skip([10, 20, 30, 40], 30, 0) == 3

    def test_correction_not_found(self):
        assert _find_draft_skip([10, 20, 30], 99, 0) == 1

    def test_search_from_offset(self):
        # Skip past index 0, find correction at index 2
        assert _find_draft_skip([10, 20, 30, 40], 30, 1) == 3

    def test_correction_at_start(self):
        assert _find_draft_skip([10, 20, 30], 10, 0) == 1


# ---------------------------------------------------------------------------
# ParsedDraftProvider tests
# ---------------------------------------------------------------------------


class MockTokenizer:
    """Minimal tokenizer mock for testing."""

    def encode(self, text, add_special_tokens=False):
        # Simple: each character becomes its ord value
        return [ord(c) for c in text]


class TestParsedDraftProvider:
    def test_basic_chunk(self):
        provider = ParsedDraftProvider("hello", MockTokenizer())
        chunk = provider.get_next_chunk(3)
        assert len(chunk) == 3
        assert not provider.is_exhausted()

    def test_exhaustion(self):
        provider = ParsedDraftProvider("ab", MockTokenizer())
        chunk = provider.get_next_chunk(10)
        assert len(chunk) == 2
        provider.advance(2)
        assert provider.is_exhausted()
        assert provider.get_next_chunk(5) == []

    def test_cursor_advance(self):
        provider = ParsedDraftProvider("abcd", MockTokenizer())
        c1 = provider.get_next_chunk(2)
        assert len(c1) == 2
        provider.advance(2)
        c2 = provider.get_next_chunk(2)
        assert len(c2) == 2
        assert c1 != c2

    def test_remaining_tokens(self):
        provider = ParsedDraftProvider("abcde", MockTokenizer())
        assert provider.remaining_tokens == 5
        provider.advance(3)
        assert provider.remaining_tokens == 2

    def test_advance_past_end(self):
        provider = ParsedDraftProvider("ab", MockTokenizer())
        provider.advance(100)
        assert provider.is_exhausted()
        assert provider.remaining_tokens == 0

    def test_empty_text(self):
        provider = ParsedDraftProvider("", MockTokenizer())
        assert provider.is_exhausted()
        assert provider.get_next_chunk(5) == []


# ---------------------------------------------------------------------------
# ParsedDraftProposer tests
# ---------------------------------------------------------------------------


class TestParsedDraftProposer:
    """Test ParsedDraftProposer with minimal mocks."""

    def _make_mock_request(self, draft_text=None):
        """Create a mock CachedRequestState-like object."""

        class MockSamplingParams:
            def __init__(self, dt):
                self.extra_args = {"draft_text": dt} if dt else None

        class MockReqState:
            def __init__(self, dt):
                self.sampling_params = MockSamplingParams(dt)

        return MockReqState(draft_text)

    def _make_mock_input_batch(self, req_ids):
        """Create a mock InputBatch-like object."""

        class MockBatch:
            def __init__(self, rids):
                self.req_ids = rids
                self.num_reqs = len(rids)
                self.req_id_to_index = {rid: i for i, rid in enumerate(rids)}

        return MockBatch(req_ids)

    def _make_proposer(self, chunk_size=4):
        """Create a ParsedDraftProposer with given chunk size."""
        from unittest.mock import MagicMock

        config = MagicMock()
        config.speculative_config.num_speculative_tokens = chunk_size
        config.speculative_config.parsed_draft_strategy = "stop_at_first"
        config.speculative_config.parsed_draft_max_reject = 3
        return ParsedDraftProposer(config)

    def test_propose_with_draft_text(self):
        proposer = self._make_proposer(chunk_size=3)
        req_id = "req-1"
        requests = {req_id: self._make_mock_request("hello")}
        batch = self._make_mock_input_batch([req_id])
        tokenizer = MockTokenizer()

        # First call — no previous draft to advance
        drafts = proposer.propose([[1]], requests, batch, tokenizer)
        assert len(drafts) == 1
        assert len(drafts[0]) == 3  # chunk_size=3, "hello" has 5 tokens

    def test_propose_without_draft_text(self):
        proposer = self._make_proposer()
        req_id = "req-1"
        requests = {req_id: self._make_mock_request(None)}
        batch = self._make_mock_input_batch([req_id])

        drafts = proposer.propose([[1]], requests, batch, MockTokenizer())
        assert drafts == [[]]  # no draft text → empty

    def test_mixed_batch(self):
        proposer = self._make_proposer(chunk_size=3)
        requests = {
            "req-1": self._make_mock_request("hello"),
            "req-2": self._make_mock_request(None),
            "req-3": self._make_mock_request("world"),
        }
        batch = self._make_mock_input_batch(["req-1", "req-2", "req-3"])

        drafts = proposer.propose([[1], [2], [3]], requests, batch, MockTokenizer())
        assert len(drafts) == 3
        assert len(drafts[0]) == 3  # has draft
        assert drafts[1] == []  # no draft
        assert len(drafts[2]) == 3  # has draft

    def test_partial_prefill_skipped(self):
        proposer = self._make_proposer()
        req_id = "req-1"
        requests = {req_id: self._make_mock_request("test")}
        batch = self._make_mock_input_batch([req_id])

        # Empty sampled_token_ids means partial prefill
        drafts = proposer.propose([[]], requests, batch, MockTokenizer())
        assert drafts == [[]]

    def test_cleanup_finished_requests(self):
        proposer = self._make_proposer(chunk_size=3)
        tokenizer = MockTokenizer()

        # First batch with req-1
        requests = {"req-1": self._make_mock_request("hello")}
        batch = self._make_mock_input_batch(["req-1"])
        proposer.propose([[1]], requests, batch, tokenizer)
        assert "req-1" in proposer._providers

        # Second batch without req-1 (finished)
        requests = {"req-2": self._make_mock_request("world")}
        batch = self._make_mock_input_batch(["req-2"])
        proposer.propose([[1]], requests, batch, tokenizer)
        assert "req-1" not in proposer._providers
        assert "req-2" in proposer._providers

    def test_cursor_advances_across_calls(self):
        proposer = self._make_proposer(chunk_size=2)
        tokenizer = MockTokenizer()
        req_id = "req-1"
        requests = {req_id: self._make_mock_request("abcd")}
        batch = self._make_mock_input_batch([req_id])

        # First call
        drafts1 = proposer.propose([[1]], requests, batch, tokenizer)
        chunk1 = drafts1[0]
        assert len(chunk1) == 2

        # Second call — sampled has 2 tokens (all accepted)
        # LCS advance should move cursor forward
        drafts2 = proposer.propose([chunk1], requests, batch, tokenizer)
        chunk2 = drafts2[0]
        assert len(chunk2) == 2
        assert chunk2 != chunk1  # different tokens
