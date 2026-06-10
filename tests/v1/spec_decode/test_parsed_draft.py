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
    holdsnap_advance,
    normalize_parsed_text,
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
# holdsnap_advance tests
# ---------------------------------------------------------------------------


class TestHoldsnapAdvance:
    def test_prefix_match_uses_lcs_advance(self):
        # When sampled[0] == last_draft[0] we're in the prefix-match
        # branch — defer to the LCS advance value (with floor 1).
        adv, holds = holdsnap_advance(
            last_draft=[1, 2, 3, 4],
            sampled=[1, 2, 99],
            lcs_advance=3,
            consecutive_holds=2,
            max_hold=8,
        )
        assert adv == 3
        assert holds == 0  # always reset on prefix match

    def test_prefix_match_with_zero_lcs_advance_floors_to_one(self):
        # Defensive: lcs_advance=0 in the prefix-match branch is
        # nonsensical (we know position 0 matches), but the function
        # should still produce a forward advance.
        adv, holds = holdsnap_advance(
            last_draft=[1, 2, 3],
            sampled=[1, 99],
            lcs_advance=0,
            consecutive_holds=0,
            max_hold=8,
        )
        assert adv == 1
        assert holds == 0

    def test_snap_to_correction_position(self):
        # Correction (sampled[0]=5) appears at draft position 2 →
        # advance by 2 so the next chunk starts with token 5.
        adv, holds = holdsnap_advance(
            last_draft=[11, 12, 5, 6],
            sampled=[5, 6],
            lcs_advance=99,  # ignored when snap fires
            consecutive_holds=4,  # any value, gets reset
            max_hold=8,
        )
        assert adv == 2
        assert holds == 0  # snap resets the counter

    def test_snap_to_first_position_correction(self):
        # Correction is at draft[1] → advance by 1 (not 0).
        # This matters because advance=0 would mean "hold" which
        # has different semantics.
        adv, holds = holdsnap_advance(
            last_draft=[11, 5, 6],
            sampled=[5],
            lcs_advance=99,
            consecutive_holds=0,
            max_hold=8,
        )
        assert adv == 1
        assert holds == 0

    def test_hold_when_correction_absent(self):
        # Correction (99) not in [11, 12, 5, 6] → hold (advance 0).
        adv, holds = holdsnap_advance(
            last_draft=[11, 12, 5, 6],
            sampled=[99],
            lcs_advance=1,  # ignored
            consecutive_holds=0,
            max_hold=8,
        )
        assert adv == 0
        assert holds == 1  # incremented

    def test_hold_increments_counter(self):
        # Multiple holds in a row should keep incrementing.
        adv, holds = holdsnap_advance(
            last_draft=[1, 2, 3],
            sampled=[99],
            lcs_advance=1,
            consecutive_holds=5,
            max_hold=8,
        )
        assert adv == 0
        assert holds == 6

    def test_forced_advance_at_max_hold(self):
        # When consecutive_holds reaches max_hold, force a 1-step
        # advance to avoid stalling.
        adv, holds = holdsnap_advance(
            last_draft=[1, 2, 3],
            sampled=[99],
            lcs_advance=1,
            consecutive_holds=8,
            max_hold=8,
        )
        assert adv == 1
        assert holds == 0  # reset after forced advance

    def test_forced_advance_at_max_hold_exceeded(self):
        # Defensive: state can exceed max_hold (e.g. max_hold lowered
        # mid-flight). Still trigger forced advance.
        adv, holds = holdsnap_advance(
            last_draft=[1, 2, 3],
            sampled=[99],
            lcs_advance=1,
            consecutive_holds=20,
            max_hold=8,
        )
        assert adv == 1
        assert holds == 0

    def test_empty_sampled_defensive(self):
        # Caller should filter empty sampled, but if we land here
        # produce a forward advance using the LCS hint.
        adv, holds = holdsnap_advance(
            last_draft=[1, 2, 3],
            sampled=[],
            lcs_advance=2,
            consecutive_holds=3,
            max_hold=8,
        )
        assert adv == 2
        assert holds == 0

    def test_empty_draft_defensive(self):
        adv, holds = holdsnap_advance(
            last_draft=[],
            sampled=[1],
            lcs_advance=5,
            consecutive_holds=3,
            max_hold=8,
        )
        assert adv == 5
        assert holds == 0


# ---------------------------------------------------------------------------
# normalize_parsed_text tests
# ---------------------------------------------------------------------------


class TestNormalizeParsedText:
    def test_ligature_decomposition(self):
        # NFKC decomposes ﬁ (U+FB01) to "fi", ﬂ (U+FB02) to "fl".
        assert normalize_parsed_text("ofﬁce") == "office"
        assert normalize_parsed_text("inﬂation") == "inflation"
        assert normalize_parsed_text("ﬀ ﬁ ﬂ ﬃ ﬄ") == "ff fi fl ffi ffl"

    def test_newline_to_space(self):
        # PyMuPDF emits \n at every visual line boundary even within
        # a paragraph; these get joined with spaces.
        assert normalize_parsed_text("a\nb\nc") == "a b c"
        # Empty lines are dropped.
        assert normalize_parsed_text("a\n\n\nb") == "a b"
        # Trailing whitespace on each line is stripped first.
        assert normalize_parsed_text("a  \n  b") == "a b"

    def test_residual_hyphen_collapse(self):
        # PyMuPDF leaves compound-word hyphens with a space after the
        # dash; we want to glue them back together.
        assert normalize_parsed_text("MALE- IDENTIFIED") == "MALE-IDENTIFIED"
        # Only triggers between word chars on both sides — does NOT
        # collapse "foo - bar" (free-standing dash, e.g. a list item).
        assert normalize_parsed_text("foo - bar") == "foo - bar"

    def test_multi_space_collapse(self):
        assert normalize_parsed_text("a   b    c") == "a b c"
        assert (
            normalize_parsed_text("  leading   middle  trailing  ")
            == "leading middle trailing"
        )

    def test_nfc_accent_composition(self):
        # NFKC also composes — combining accent + letter form the
        # single accented char.
        decomposed = "café"  # café with combining acute
        composed = "café"
        assert normalize_parsed_text(decomposed) == composed

    def test_idempotent(self):
        # Calling twice gives the same result as calling once.
        s = "ofﬁce  with-- MALE- IDENTIFIED  text"
        once = normalize_parsed_text(s)
        twice = normalize_parsed_text(once)
        assert once == twice

    def test_empty_and_whitespace(self):
        assert normalize_parsed_text("") == ""
        assert normalize_parsed_text("   ") == ""

    def test_no_change_on_clean_text(self):
        # ASCII text with single spaces and no ligatures should pass
        # through unchanged.
        clean = "The quick brown fox jumps over the lazy dog."
        assert normalize_parsed_text(clean) == clean


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

    def test_normalize_via_draft_text(self):
        # draft_text path: normalize=True (default) should NFKC the
        # input before tokenizing, so a ligature in the input becomes
        # the ASCII pair after tokenization.
        tok = MockTokenizer()
        with_norm = ParsedDraftProvider(draft_text="oﬃce  hi", tokenizer=tok)
        # "ofﬁce  hi" → "ofﬁce hi" (NFKC turns ﬃ into ffi, then
        # double-space collapses): "office hi" → 9 chars (= 9 tokens
        # for MockTokenizer).
        assert with_norm.draft_ids == [ord(c) for c in "office hi"]

        without_norm = ParsedDraftProvider(
            draft_text="oﬃce  hi", tokenizer=tok, normalize=False
        )
        # No normalization: raw 7-char string → 7 tokens.
        assert without_norm.draft_ids == [ord(c) for c in "oﬃce  hi"]

    def test_normalize_does_not_apply_to_draft_ids_path(self):
        # When IDs are supplied directly, ``normalize`` has no effect —
        # the caller is responsible for normalization upstream.
        ids = [1, 2, 3]
        p_default = ParsedDraftProvider(draft_ids=ids)
        p_no_norm = ParsedDraftProvider(draft_ids=ids, normalize=False)
        assert p_default.draft_ids == ids
        assert p_no_norm.draft_ids == ids

    def test_consecutive_holds_starts_at_zero(self):
        # Providers begin with no held steps.
        p = ParsedDraftProvider(draft_ids=[1, 2, 3])
        assert p.consecutive_holds == 0

    def test_consecutive_holds_state_is_writable(self):
        # The proposer mutates this directly after each holdsnap_advance
        # call; verify the field is in fact writable.
        p = ParsedDraftProvider(draft_ids=[1, 2, 3])
        p.consecutive_holds = 5
        assert p.consecutive_holds == 5


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


# ---------------------------------------------------------------------------
# _hybrid_accept_one return-shape tests
# ---------------------------------------------------------------------------


class _MinimalProposer:
    """Just enough to call _hybrid_accept_one without loading a model."""

    def __init__(self, strategy: str = "stop_at_first", max_reject: int = 3):
        self.strategy = strategy
        self.max_reject = max_reject

    _hybrid_accept_one = ParsedDraftProposer._hybrid_accept_one


class TestHybridAcceptOneReturnShape:
    """Regression tests for the bonus-token correctness bug.

    ``_hybrid_accept_one`` must return ``(accepted_tokens, all_matched)``
    where ``all_matched`` is True only when every emitted token was an
    accepted draft token (no correction). The caller relies on this to
    decide whether to append the bonus token — appending after a
    correction would emit a token whose context (in the verifier's
    forward pass) assumed the rejected draft token, diverging from AR.
    """

    def test_stop_at_first_all_matched_returns_true(self):
        p = _MinimalProposer("stop_at_first")
        accepted, all_matched = p._hybrid_accept_one([1, 2, 3], [1, 2, 3])
        assert accepted == [1, 2, 3]
        assert all_matched is True

    def test_stop_at_first_mismatch_at_start_returns_false(self):
        p = _MinimalProposer("stop_at_first")
        accepted, all_matched = p._hybrid_accept_one([1, 2, 3], [99, 88, 77])
        assert accepted == [99]
        assert all_matched is False

    def test_stop_at_first_mismatch_at_end_returns_false(self):
        # The buggy version would have returned True here because
        # len(accepted) == n_draft, even though position 2 was corrected.
        p = _MinimalProposer("stop_at_first")
        accepted, all_matched = p._hybrid_accept_one([1, 2, 3], [1, 2, 99])
        assert accepted == [1, 2, 99]
        assert all_matched is False

    def test_stop_at_first_mismatch_at_middle_returns_false(self):
        p = _MinimalProposer("stop_at_first")
        accepted, all_matched = p._hybrid_accept_one([1, 2, 3, 4], [1, 99, 3, 4])
        assert accepted == [1, 99]
        assert all_matched is False

    def test_stop_at_first_single_token_match_returns_true(self):
        # n_draft == 1 + all matched → bonus is valid
        p = _MinimalProposer("stop_at_first")
        accepted, all_matched = p._hybrid_accept_one([5], [5])
        assert accepted == [5]
        assert all_matched is True

    def test_stop_at_first_single_token_mismatch_returns_false(self):
        # n_draft == 1 + correction → bonus would be wrong (used to fire
        # the buggy len(accepted) == n_draft branch).
        p = _MinimalProposer("stop_at_first")
        accepted, all_matched = p._hybrid_accept_one([5], [99])
        assert accepted == [99]
        assert all_matched is False

    def test_stop_at_first_empty_chunk(self):
        p = _MinimalProposer("stop_at_first")
        accepted, all_matched = p._hybrid_accept_one([], [])
        assert accepted == []
        assert all_matched is False

    def test_hybrid_all_matched_returns_true(self):
        p = _MinimalProposer("hybrid", max_reject=3)
        accepted, all_matched = p._hybrid_accept_one([1, 2, 3, 4], [1, 2, 3, 4])
        assert accepted == [1, 2, 3, 4]
        assert all_matched is True

    def test_hybrid_with_correction_returns_false(self):
        # One LCS-mismatch in the middle → correction taken,
        # all_matched must be False. Bug would have triggered
        # the bonus-append because len(output) == n_draft.
        p = _MinimalProposer("hybrid", max_reject=3)
        # Use distinct tokens: draft[1]=22, target[1]=99 → mismatch at pos 1.
        # LCS on [10,22,30,40] vs [10,99,30,40] matches positions 0,2,3
        # (since 22 ≠ 99). At pos 1 we take the correction.
        accepted, all_matched = p._hybrid_accept_one([10, 22, 30, 40], [10, 99, 30, 40])
        # Output length still 4, but pos 1 is a correction → not all_matched.
        assert len(accepted) == 4
        assert all_matched is False

    def test_hybrid_bailout_returns_false(self):
        # max_reject=2: two consecutive mismatches trigger bail.
        # Output is truncated and includes a correction.
        p = _MinimalProposer("hybrid", max_reject=2)
        accepted, all_matched = p._hybrid_accept_one(
            [10, 20, 30, 40, 50],
            [10, 99, 88, 40, 50],
        )
        # Bail at pos 2 (2 consecutive mismatches): output truncated.
        assert len(accepted) < 5
        assert all_matched is False
