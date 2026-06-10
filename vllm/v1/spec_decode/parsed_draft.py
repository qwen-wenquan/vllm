# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Parsed-Draft Speculative Decoding.

Uses pre-existing text (e.g., OCR-extracted from PDF) as a zero-cost
"draft" for speculative decoding. The target model verifies chunks of
draft tokens; LCS alignment matches draft tokens to verifier output,
allowing acceptance even when the draft has insertions/deletions.

Phase 1: stop_at_first strategy (greedy-exact, identical to AR output).
"""

from __future__ import annotations

import unicodedata
from typing import Any

import numpy as np
import regex as re
import torch
from transformers import AutoTokenizer

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.outputs import SamplerOutput
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.worker.gpu_input_batch import InputBatch

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Text normalization for parsed draft input
# ---------------------------------------------------------------------------

# Pre-compile once at module load.
#
# - ``_RESIDUAL_HYPHEN_RE``: fixes compound-word hyphens that survived
#   PyMuPDF's TEXT_DEHYPHENATE — patterns like ``"MALE- IDENTIFIED"``
#   (a line-broken compound) collapse to ``"MALE-IDENTIFIED"``.
# - ``_MULTI_SPACE_RE``: collapses runs of 2+ spaces to a single space.
_RESIDUAL_HYPHEN_RE = re.compile(r"(\w)- (\w)")
_MULTI_SPACE_RE = re.compile(r" {2,}")


def normalize_parsed_text(text: str) -> str:
    """Normalize parsed-draft text to maximize token-level alignment
    with target model output.

    Applies four idempotent, content-preserving transformations that
    remove sources of token-level disagreement that don't affect
    meaning:

    1. **Visual line breaks → spaces.** PyMuPDF emits ``\\n`` at every
       visual line boundary, even within a paragraph. These are
       layout artifacts, not semantic breaks. Lines are stripped and
       joined with a single space.
    2. **NFKC unicode normalization.** Decomposes compatibility
       characters — most importantly ligatures like ``ﬁ`` → ``fi``
       and ``ﬂ`` → ``fl`` (common in academic PDFs but rare in
       VL-model output). Also composes accented characters to their
       canonical form.
    3. **Residual hyphen fix.** PyMuPDF's ``TEXT_DEHYPHENATE`` flag
       handles word-level line-break hyphens (``pho-\\ntonic`` →
       ``photonic``) but misses compound-word hyphens. After joining
       lines with spaces, those appear as ``"MALE- IDENTIFIED"``;
       this collapses them to ``"MALE-IDENTIFIED"``.
    4. **Whitespace collapse.** Runs of multiple spaces collapse to
       a single space, and leading/trailing whitespace is stripped.

    These mirror the rules in
    ``benchmarks/spec_decode/pymupdf_utils.py::_clean_paragraph_text``,
    applied so the draft matches what the target VL model naturally
    emits from the same image crop.

    Measured impact (full 10,000-block OCR sample,
    ``sft_ocr_blocks_10k.tokenized.json``, hybrid_mr=3 strategy):

    +---------------+---------------+----------------+---------+
    | chunk_size    | baseline      | normalized     | gain    |
    +===============+===============+================+=========+
    | 16            | 3.54×         | 3.93×          | +14 %   |
    | 50            | 3.01×         | 3.76×          | +25 %   |
    | 200           | 2.24×         | 3.28×          | +46 %   |
    +---------------+---------------+----------------+---------+

    The function is safe to call on already-normalized text — repeated
    application is a no-op after the first call.
    """
    # 1. Visual line breaks → spaces.
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    text = " ".join(lines)
    # 2. NFKC unicode normalization (ligatures, compatibility forms).
    text = unicodedata.normalize("NFKC", text)
    # 3. Residual hyphen fix.
    text = _RESIDUAL_HYPHEN_RE.sub(r"\1-\2", text)
    # 4. Whitespace collapse.
    text = _MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# LCS utilities (ported from EAGLE/eagle/model/parsed_draft_spec_decode.py)
# ---------------------------------------------------------------------------


def find_prefix_match(
    draft_tokens: list[int],
    logits: torch.Tensor,
) -> int:
    """Find the length of the matching prefix between draft and verifier.

    Args:
        draft_tokens: Draft token ids for this chunk.
        logits: [chunk_len, vocab_size] logits from verifier.

    Returns:
        Number of consecutive matching tokens from the start.
    """
    n = min(len(draft_tokens), logits.shape[0])
    if n == 0:
        return 0

    predicted = torch.argmax(logits[:n], dim=-1)
    draft_tensor = torch.tensor(draft_tokens[:n], device=predicted.device)
    matches = (predicted == draft_tensor).cpu().tolist()

    for i, m in enumerate(matches):
        if not m:
            return i
    return n


def _lcs_draft_advance(draft_tokens: list[int], gt_prefix: list[int]) -> int:
    """Find how many draft tokens are consumed by matching a GT prefix
    via LCS.

    Runs a small LCS between draft_tokens and gt_prefix, then finds
    the draft position of the last matched token + 1.

    Example:
        draft:     [A, X, B, C, D, ...]   (X is OCR insertion)
        gt_prefix: [A, B, C, Y]           (A,B,C matched, Y is correction)

        LCS matches A(0), B(2), C(3) → last matched at draft pos 3
        Advance = 3 + 1 = 4

    Returns:
        Number of draft tokens to advance (>= 1).
    """
    m_len = len(draft_tokens)
    n_len = len(gt_prefix)
    if m_len == 0 or n_len == 0:
        return 1

    # Build DP table
    dp = [[0] * (n_len + 1) for _ in range(m_len + 1)]
    for i in range(1, m_len + 1):
        for j in range(1, n_len + 1):
            if draft_tokens[i - 1] == gt_prefix[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Backtrack to find last matched draft position
    last_draft_pos = 0
    i, j = m_len, n_len
    while i > 0 and j > 0:
        if draft_tokens[i - 1] == gt_prefix[j - 1] and dp[i][j] == dp[i - 1][j - 1] + 1:
            last_draft_pos = max(last_draft_pos, i - 1)
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return last_draft_pos + 1


def _incremental_lcs_advance(
    draft_tokens: list[int],
    gt_tokens: list[int],
) -> int:
    """Find draft cursor advance using incremental (row-by-row) DP.

    Same semantics as ``_lcs_draft_advance`` but builds the DP table
    one GT token at a time (row = GT, column = draft). This layout
    enables batched vectorization (Phase 2) and GPU execution
    (Phase 3).

    Each GT token extends a single DP row of length ``M + 1`` at
    O(M) cost. Total work is still O(M * N), but memory is O(M)
    for the live row plus O(N * M) for backtrack history (only
    the current chunk's history, cleared after each call).

    Returns:
        Number of draft tokens to advance (>= 1).
    """
    m_len = len(draft_tokens)
    n_len = len(gt_tokens)
    if m_len == 0 or n_len == 0:
        return 1

    # Row-by-row DP: process one GT token at a time, extending
    # a single row of length M+1. dp_history[j][i] corresponds
    # to dp[i][j] in the classic 2D formulation.
    dp_row = [0] * (m_len + 1)
    dp_history: list[list[int]] = [list(dp_row)]

    for gt_tok in gt_tokens:
        prev_row = dp_row
        dp_row = [0] * (m_len + 1)
        for i in range(1, m_len + 1):
            if draft_tokens[i - 1] == gt_tok:
                dp_row[i] = prev_row[i - 1] + 1
            else:
                dp_row[i] = max(dp_row[i - 1], prev_row[i])
        dp_history.append(list(dp_row))

    # Backtrack to find last matched DRAFT position.
    # dp_history[j][i] == dp[i][j] in the classic layout.
    # Original: dp[i-1][j] >= dp[i][j-1] → i -= 1
    # Transposed: dp_history[j][i-1] >= dp_history[j-1][i] → i -= 1
    last_draft_pos = 0
    i, j = m_len, n_len
    while i > 0 and j > 0:
        if (
            draft_tokens[i - 1] == gt_tokens[j - 1]
            and dp_history[j][i] == dp_history[j - 1][i - 1] + 1
        ):
            last_draft_pos = max(last_draft_pos, i - 1)
            i -= 1
            j -= 1
        elif dp_history[j][i - 1] >= dp_history[j - 1][i]:
            i -= 1
        else:
            j -= 1

    return max(last_draft_pos + 1, 1)


def _batched_lcs_advance(
    draft_tokens_list: list[list[int]],
    gt_tokens_list: list[list[int]],
) -> list[int]:
    """Batched LCS draft advance using numpy vectorisation.

    Processes *all* requests in the batch simultaneously: for each
    column ``i`` of the DP table the update across all B requests
    is a single numpy vectorised operation.

    Backtracking is still per-request (the path is data-dependent
    and cheap at O(M + N) per request).

    Args:
        draft_tokens_list: Per-request draft token lists.
        gt_tokens_list:    Per-request GT (sampled) token lists.

    Returns:
        Per-request cursor advance amounts (``len == len(draft_tokens_list)``).
    """
    B = len(draft_tokens_list)
    if B == 0:
        return []

    max_m = max((len(d) for d in draft_tokens_list), default=0)
    max_n = max((len(g) for g in gt_tokens_list), default=0)
    if max_m == 0 or max_n == 0:
        return [1] * B

    # Pad draft and gt into [B, max_m] / [B, max_n] numpy arrays.
    # Use -1 as padding (no valid token id is -1).
    draft_arr = np.full((B, max_m), -1, dtype=np.int64)
    gt_arr = np.full((B, max_n), -1, dtype=np.int64)
    draft_lens = np.empty(B, dtype=np.int64)
    gt_lens = np.empty(B, dtype=np.int64)

    for b in range(B):
        d = draft_tokens_list[b]
        g = gt_tokens_list[b]
        draft_arr[b, : len(d)] = d
        gt_arr[b, : len(g)] = g
        draft_lens[b] = len(d)
        gt_lens[b] = len(g)

    # ── Forward pass: row-by-row DP (one row per GT token) ──
    #
    # dp_row[b, i] = LCS(draft_b[0:i], gt_b[0:j]) after processing
    # j GT tokens.  We store dp_history[j][b, :] for backtracking.
    dp_row = np.zeros((B, max_m + 1), dtype=np.int32)
    # dp_history[0] = all-zeros row (before any GT token)
    dp_history = [dp_row.copy()]

    for j in range(max_n):
        gt_col = gt_arr[:, j]  # [B]
        # Which requests still have GT tokens at position j?
        active = j < gt_lens  # [B] bool

        prev_row = dp_row.copy()  # dp_row *before* this GT token
        for i in range(1, max_m + 1):
            valid = active & (i <= draft_lens)
            matched = valid & (draft_arr[:, i - 1] == gt_col)
            dp_row[:, i] = np.where(
                matched,
                prev_row[:, i - 1] + 1,
                np.where(
                    valid,
                    np.maximum(dp_row[:, i - 1], prev_row[:, i]),
                    dp_row[:, i],
                ),
            )
        dp_history.append(dp_row.copy())

    # ── Backtrack per request ──
    advances = []
    for b in range(B):
        m = int(draft_lens[b])
        n = int(gt_lens[b])
        if m == 0 or n == 0:
            advances.append(1)
            continue

        last_draft_pos = 0
        i, j = m, n
        while i > 0 and j > 0:
            if (
                draft_arr[b, i - 1] == gt_arr[b, j - 1]
                and dp_history[j][b, i] == dp_history[j - 1][b, i - 1] + 1
            ):
                last_draft_pos = max(last_draft_pos, i - 1)
                i -= 1
                j -= 1
            elif dp_history[j][b, i - 1] >= dp_history[j - 1][b, i]:
                i -= 1
            else:
                j -= 1

        advances.append(max(last_draft_pos + 1, 1))

    return advances


def _find_draft_skip(draft_tokens: list[int], correction: int, start: int) -> int:
    """Find how far to advance draft cursor when no prefix matched.

    Searches for the correction token in the draft starting from
    *start*. If found, skip past it. Otherwise skip 1.

    Returns:
        Number of draft tokens to skip (>= 1).
    """
    for i in range(start, len(draft_tokens)):
        if draft_tokens[i] == correction:
            return i + 1
    return 1


def holdsnap_advance(
    last_draft: list[int],
    sampled: list[int],
    lcs_advance: int,
    consecutive_holds: int,
    max_hold: int,
) -> tuple[int, int]:
    """Hold+snap cursor-advance rule (simulator-tuned; OFF by default
    in production — see "Production caveat" below).

    Used in the no-prefix-match branch when ``sampled[0] != last_draft[0]``
    (the verifier rejected the very first draft token). In that case the
    LCS-based advance ``_lcs_draft_advance(last_draft, sampled)`` would
    blindly advance the cursor by ``max(adv, 1)``, consuming a draft
    token that may match a gt token a few positions later. The
    hold+snap rule looks at ``sampled[0]`` (the correction emitted by
    the verifier) and:

    - **SNAP**: if ``sampled[0]`` appears at position ``i`` in
      ``last_draft``, advance by exactly ``i`` so the next chunk starts
      with that token. The verifier can then confirm it as a real
      prefix match.
    - **HOLD**: if ``sampled[0]`` is not in ``last_draft``, advance by 0
      (keep cursor parked) and re-verify the same draft chunk against
      the shifted gt window on the next step. Bounded by ``max_hold``
      consecutive holds — once exceeded, fall back to advancing by 1
      to avoid stalling.

    Lossless: changing only the cursor never changes which tokens the
    verifier accepts — the spec decoder's output remains byte-identical
    to autoregressive decoding.

    **Production caveat — why this is off by default.** The simulator
    (``benchmarks/spec_decode/benchmark_parsed_draft.py``) measured
    +17 % speedup at chunk=16 with this rule, but the e2e GTX5k
    benchmark measured a regression (1.49× → 1.03×). The two diverge
    because the simulator's "verifier" is the gt token stream, which
    slides forward independently of what the simulator feeds it. The
    real verifier is a deterministic forward pass over the prompt +
    accepted-so-far + draft chunk: re-feeding the same chunk after a
    HOLD produces the same rejection, so the cursor stays parked until
    ``max_hold`` forces an advance — wasting an entire verifier pass
    per held step. ``holdsnap_advance`` is kept for research; the
    fix is a verifier-aware cursor rule, not this one.

    Args:
        last_draft: Tokens of the draft chunk proposed last step.
        sampled: Verifier-accepted/corrected tokens from this step
            (must be non-empty; caller filters empty case).
        lcs_advance: Result of the LCS-based advance computation,
            used as the fallback if neither snap nor hold apply.
        consecutive_holds: Number of consecutive holds applied to
            this request's cursor so far.
        max_hold: Maximum consecutive holds before forcing an advance
            (typically equal to chunk_size, default 8 in benchmarks).

    Returns:
        ``(advance, new_consecutive_holds)`` — the cursor advance to
        apply (always ≥ 0; 0 means "hold") and the updated hold
        counter (0 on snap or prefix match, +1 on hold, 0 after
        forced advance).
    """
    if not sampled or not last_draft:
        # Defensive: caller should filter, but if we land here just
        # use the LCS advance with the standard floor of 1.
        return max(lcs_advance, 1), 0

    if sampled[0] == last_draft[0]:
        # Prefix-match case — the LCS advance is the right answer.
        return max(lcs_advance, 1), 0

    # No prefix match. Look for the correction in the rest of the
    # draft chunk.
    correction = sampled[0]
    for i, tok in enumerate(last_draft):
        if tok == correction:
            # SNAP: advance to the correction position so the next
            # chunk starts with it.
            return i, 0

    # Correction not in draft — HOLD, unless we've hit the cap.
    if consecutive_holds >= max_hold:
        return 1, 0
    return 0, consecutive_holds + 1


class IncrementalLCSMatcher:
    """Incremental LCS-based matcher for speculative decoding.

    The draft is fixed upfront. GT tokens arrive incrementally in
    chunks. We maintain a single DP row and extend it column-by-column
    as new GT tokens arrive, so each new GT token costs O(M) work
    where M = len(draft).

    After processing a chunk of GT tokens, we backtrack through the DP
    table to identify which GT positions are LCS-matched (accepted) vs
    unmatched (rejected).
    """

    def __init__(self, draft_tokens: list[int]):
        self.draft = draft_tokens
        self.M = len(draft_tokens)
        self.dp_row: list[int] = [0] * (self.M + 1)
        self.dp_history: list[list[int]] = [[0] * (self.M + 1)]
        self.gt_so_far: list[int] = []

    def add_chunk(
        self,
        gt_chunk_tokens: list[int],
    ) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        """Process a chunk of GT tokens, return accepted/rejected spans.

        Spans are half-open intervals in chunk-relative coordinates.
        """
        chunk_start = len(self.gt_so_far)

        for gt_tok in gt_chunk_tokens:
            prev_row = self.dp_row
            new_row = [0] * (self.M + 1)
            for i in range(1, self.M + 1):
                if self.draft[i - 1] == gt_tok:
                    new_row[i] = prev_row[i - 1] + 1
                else:
                    new_row[i] = max(new_row[i - 1], prev_row[i])
            self.dp_row = new_row
            self.dp_history.append(list(new_row))
            self.gt_so_far.append(gt_tok)

        chunk_len = len(gt_chunk_tokens)
        gt_matched: set[int] = set()

        i = self.M
        j = chunk_start + chunk_len

        while i > 0 and j > chunk_start:
            if (
                self.draft[i - 1] == self.gt_so_far[j - 1]
                and self.dp_history[j][i] == self.dp_history[j - 1][i - 1] + 1
            ):
                gt_matched.add(j - 1 - chunk_start)
                i -= 1
                j -= 1
                continue
            if self.dp_history[j - 1][i] >= self.dp_history[j][i - 1]:
                j -= 1
            else:
                i -= 1

        accepted_spans: list[tuple[int, int]] = []
        rejected_spans: list[tuple[int, int]] = []

        pos = 0
        while pos < chunk_len:
            if pos in gt_matched:
                start = pos
                while pos < chunk_len and pos in gt_matched:
                    pos += 1
                accepted_spans.append((start, pos))
            else:
                start = pos
                while pos < chunk_len and pos not in gt_matched:
                    pos += 1
                rejected_spans.append((start, pos))

        return accepted_spans, rejected_spans

    def reset_history(self):
        """Clear backtrack history to save memory."""
        self.dp_history = [list(self.dp_row)]
        self.gt_so_far = []


# ---------------------------------------------------------------------------
# Draft provider
# ---------------------------------------------------------------------------


class ParsedDraftProvider:
    """Manages parsed text as a draft token source.

    Serves chunks of draft token IDs via a simple cursor.
    After verification, the cursor is advanced using LCS
    alignment to skip past OCR insertions/deletions.

    Accepts either pre-tokenized IDs (preferred — avoids
    tokenizer overhead in the engine loop) or raw text
    (tokenized once on construction).

    Text-input path: ``normalize=True`` (default) runs
    ``normalize_parsed_text`` on the input before tokenizing.
    This typically improves end-to-end speedup by 14-46% on OCR
    workloads (see the helper's docstring for measurements).
    For the pre-tokenized path callers must apply normalization
    themselves before tokenizing — the IDs are taken as given.
    """

    def __init__(
        self,
        draft_ids: list[int] | None = None,
        draft_text: str | None = None,
        tokenizer: Any = None,
        normalize: bool = True,
    ):
        if draft_ids is not None:
            self.draft_ids = draft_ids
        elif draft_text is not None and tokenizer is not None:
            if normalize:
                draft_text = normalize_parsed_text(draft_text)
            self.draft_ids = tokenizer.encode(draft_text, add_special_tokens=False)
        else:
            raise ValueError(
                "ParsedDraftProvider requires either "
                "draft_ids or (draft_text + tokenizer)"
            )
        self.cursor: int = 0
        # Consecutive-hold counter for the holdsnap cursor rule.
        # Reset on any prefix-match or snap step; incremented on each
        # held step. Bounded by the proposer's max_hold parameter.
        self.consecutive_holds: int = 0

    def get_next_chunk(self, max_tokens: int) -> list[int]:
        """Return next chunk of draft token ids."""
        end = min(self.cursor + max_tokens, len(self.draft_ids))
        return self.draft_ids[self.cursor : end]

    def advance(self, n_consumed: int) -> None:
        """Advance cursor by n_consumed draft tokens."""
        self.cursor = min(self.cursor + n_consumed, len(self.draft_ids))

    def is_exhausted(self) -> bool:
        """Return True if all draft tokens have been consumed."""
        return self.cursor >= len(self.draft_ids)

    @property
    def remaining_tokens(self) -> int:
        return len(self.draft_ids) - self.cursor


# ---------------------------------------------------------------------------
# Proposer
# ---------------------------------------------------------------------------


class ParsedDraftProposer:
    """Proposes draft tokens from pre-parsed text for speculative
    decoding.

    Non-model proposer (like NgramProposer). Maintains per-request
    state via ParsedDraftProvider objects. Requests without draft_text
    return empty drafts and skip spec decode.

    Cursor advancement happens at the start of each ``propose()`` call:
    the ``sampled_token_ids`` from the previous verification step tell
    us which tokens were accepted, and LCS alignment determines how
    far to advance the draft cursor (skipping OCR insertions/deletions).
    """

    def __init__(self, vllm_config: VllmConfig):
        assert vllm_config.speculative_config is not None
        spec_config = vllm_config.speculative_config
        self.chunk_size = spec_config.num_speculative_tokens
        self.strategy = spec_config.parsed_draft_strategy
        self.max_reject = spec_config.parsed_draft_max_reject
        self.holdsnap = spec_config.parsed_draft_holdsnap
        self.max_hold = spec_config.parsed_draft_max_hold
        self.lcs_backend = spec_config.parsed_draft_lcs_backend
        self._providers: dict[str, ParsedDraftProvider] = {}
        # Track which draft tokens were proposed per request so
        # cursor advancement can use LCS alignment on the next call.
        self._last_proposed: dict[str, list[int]] = {}

        # Resolve device for Triton backend
        self._device: torch.device | None = None
        if self.lcs_backend == "triton":
            self._device = torch.device("cuda")

        # Load tokenizer for converting draft_text to token ids.
        # The tokenizer is loaded here (in the EngineCore process)
        # because draft_text arrives via SamplingParams.extra_args
        # and must be tokenized when the provider is first created.
        model_config = vllm_config.model_config
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_config.tokenizer if model_config.tokenizer else model_config.model,
            trust_remote_code=model_config.trust_remote_code,
        )
        logger.info(
            "ParsedDraftProposer initialized: "
            "chunk_size=%d, strategy=%s, max_reject=%d, "
            "holdsnap=%s, max_hold=%d, lcs_backend=%s",
            self.chunk_size,
            self.strategy,
            self.max_reject,
            self.holdsnap,
            self.max_hold,
            self.lcs_backend,
        )

    def propose(
        self,
        sampled_token_ids: list[list[int]],
        requests: dict[str, Any],
        input_batch: InputBatch,
    ) -> list[list[int]]:
        """Return draft token IDs per request from parsed text cursors.

        Args:
            sampled_token_ids: Per-request accepted token ids from last
                step. Empty list means partial prefill (skip).
            requests: Dict mapping req_id → CachedRequestState.
            input_batch: Current InputBatch with req_ids and metadata.

        Returns:
            list[list[int]]: Draft token ids per request. Empty list for
                requests without draft_text or exhausted drafts.
        """
        num_reqs = input_batch.num_reqs

        # ── Pass 1: Collect advance pairs ──
        # Gather (draft, gt) pairs for all requests that need
        # cursor advancement, so we can batch the LCS computation.
        advance_indices: list[int] = []
        advance_drafts: list[list[int]] = []
        advance_gts: list[list[int]] = []
        providers: list[ParsedDraftProvider | None] = []

        for i in range(num_reqs):
            req_id = input_batch.req_ids[i]
            sampled = sampled_token_ids[i]

            if not sampled:
                providers.append(None)
                continue

            provider = self._get_or_create_provider(req_id, requests)
            providers.append(provider)
            if provider is None or provider.is_exhausted():
                continue

            last_draft = self._last_proposed.pop(req_id, None)
            if last_draft is not None and sampled:
                advance_indices.append(i)
                advance_drafts.append(last_draft)
                advance_gts.append(sampled)

        # ── Batched LCS advance ──
        if advance_indices:
            advances = self._compute_lcs_advances(advance_drafts, advance_gts)
            for idx, adv, last_draft, sampled in zip(
                advance_indices, advances, advance_drafts, advance_gts
            ):
                prov = providers[idx]
                assert prov is not None
                if self.holdsnap:
                    new_adv, prov.consecutive_holds = holdsnap_advance(
                        last_draft,
                        sampled,
                        adv,
                        prov.consecutive_holds,
                        self.max_hold,
                    )
                    prov.advance(new_adv)
                else:
                    prov.advance(max(adv, 1))

        # ── Pass 2: Emit draft chunks ──
        draft_token_ids: list[list[int]] = []
        for i in range(num_reqs):
            req_id = input_batch.req_ids[i]
            provider = providers[i]
            if provider is None or provider.is_exhausted():
                draft_token_ids.append([])
                continue

            chunk = provider.get_next_chunk(self.chunk_size)
            self._last_proposed[req_id] = chunk
            draft_token_ids.append(chunk)

        # Cleanup providers for requests no longer in the batch.
        active_req_ids = set(input_batch.req_ids[j] for j in range(num_reqs))
        for rid in list(self._providers):
            if rid not in active_req_ids:
                del self._providers[rid]
                self._last_proposed.pop(rid, None)

        return draft_token_ids

    def _compute_lcs_advances(
        self,
        draft_list: list[list[int]],
        gt_list: list[list[int]],
    ) -> list[int]:
        """Dispatch LCS cursor advancement to the configured backend.

        'auto' (default): python for < 64 requests, numpy for >= 64.
        'python':  per-request incremental DP (Phase 1).
        'numpy':   batched numpy vectorisation (Phase 2).
        'triton':  GPU Triton kernel (Phase 3).
        'positional': DISABLES LCS — cursor advances by exactly
            ``len(gt)`` (prefix_len + 1 for stop_at_first, bail_pos
            for hybrid). Ablation only; OCR insertions/deletions
            will permanently desync the cursor from the verifier.
        """
        backend = self.lcs_backend
        n = len(draft_list)

        if backend == "positional":
            # No LCS: assume draft and GT are positionally aligned and
            # advance by the number of GT tokens consumed last step.
            return [max(len(g), 1) for g in gt_list]

        if backend == "triton":
            from vllm.v1.spec_decode.utils import triton_lcs_advance

            assert self._device is not None
            return triton_lcs_advance(draft_list, gt_list, self._device)

        if backend == "numpy" or (backend == "auto" and n >= 64):
            return _batched_lcs_advance(draft_list, gt_list)

        # "python" or "auto" with small batch
        return [_incremental_lcs_advance(d, g) for d, g in zip(draft_list, gt_list)]

    def _get_or_create_provider(
        self,
        req_id: str,
        requests: dict[str, Any],
    ) -> ParsedDraftProvider | None:
        """Get existing provider or create one from request's
        extra_args.

        Checks for pre-tokenized ``draft_token_ids`` first
        (list[int]), falling back to ``draft_text`` (str) +
        tokenizer. Pre-tokenized IDs avoid tokenizer overhead
        inside the engine loop.
        """
        if req_id in self._providers:
            return self._providers[req_id]

        req_state = requests.get(req_id)
        if req_state is None:
            return None

        # Extract from sampling_params.extra_args
        sp = getattr(req_state, "sampling_params", None)
        if sp is None:
            return None
        extra = getattr(sp, "extra_args", None)
        if extra is None:
            return None

        # Prefer pre-tokenized IDs (no tokenizer needed)
        draft_ids = extra.get("draft_token_ids")
        if draft_ids and isinstance(draft_ids, list):
            provider = ParsedDraftProvider(draft_ids=draft_ids)
            self._providers[req_id] = provider
            logger.debug(
                "Created ParsedDraftProvider for req %s: %d pre-tokenized draft tokens",
                req_id,
                len(provider.draft_ids),
            )
            return provider

        # Fall back to draft_text + tokenizer
        draft_text = extra.get("draft_text")
        if not draft_text:
            return None

        provider = ParsedDraftProvider(
            draft_text=draft_text,
            tokenizer=self._tokenizer,
        )
        self._providers[req_id] = provider
        logger.debug(
            "Created ParsedDraftProvider for req %s: "
            "%d draft tokens (tokenized from text)",
            req_id,
            len(provider.draft_ids),
        )
        return provider

    # ------------------------------------------------------------------
    # Hybrid acceptance (bypass standard rejection sampler)
    # ------------------------------------------------------------------

    def accept_tokens(
        self,
        metadata: SpecDecodeMetadata,
        target_logits: torch.Tensor,
        bonus_logits: torch.Tensor,
        input_batch: InputBatch,
    ) -> SamplerOutput:
        """Accept draft tokens using LCS-based hybrid strategy.

        Instead of stopping at the first mismatch (standard rejection),
        this method:
        1. Computes target argmax for each draft position
        2. Uses LCS alignment to identify matching positions
        3. Scans left-to-right, accepting matches and including
           corrections for mismatches
        4. Bails out after max_reject consecutive rejections

        Only supports greedy decoding (temperature=0).

        For requests with 0 draft tokens, the bonus token (target
        model's argmax at the request's position) is emitted so the
        request makes progress — matching the standard rejection
        sampler's behavior.

        Returns SamplerOutput with the same shape as the standard
        rejection sampler: sampled_token_ids[batch, max_spec_len+1]
        with PLACEHOLDER_TOKEN_ID for unused slots.
        """
        PLACEHOLDER = -1

        draft_token_ids = metadata.draft_token_ids
        num_draft_tokens = metadata.num_draft_tokens
        cu_num_draft = metadata.cu_num_draft_tokens
        batch_size = len(num_draft_tokens)
        max_spec_len = metadata.max_spec_len
        device = target_logits.device

        # Compute target argmax for all draft positions
        target_argmax = target_logits.argmax(dim=-1)

        # Compute bonus tokens (one per request)
        bonus_argmax = bonus_logits.argmax(dim=-1)

        # Output buffer: [batch_size, max_spec_len + 1]
        output = torch.full(
            (batch_size, max_spec_len + 1),
            PLACEHOLDER,
            dtype=torch.int32,
            device=device,
        )

        # Move to CPU for Python-level LCS + scan
        draft_ids_cpu = draft_token_ids.cpu().tolist()
        target_ids_cpu = target_argmax.cpu().tolist()
        bonus_ids_cpu = bonus_argmax.cpu().tolist()
        cu_num_cpu = cu_num_draft.cpu().tolist()

        output_cpu = output.cpu()
        for req_idx in range(batch_size):
            start = cu_num_cpu[req_idx - 1] if req_idx > 0 else 0
            end = cu_num_cpu[req_idx]
            n_draft = end - start

            if n_draft == 0:
                # No draft tokens — emit the bonus token so the
                # request makes progress (matches standard rejection
                # sampler behavior).
                output_cpu[req_idx, 0] = bonus_ids_cpu[req_idx]
                continue

            draft_chunk = draft_ids_cpu[start:end]
            target_chunk = target_ids_cpu[start:end]

            accepted = self._hybrid_accept_one(draft_chunk, target_chunk)

            # If all draft tokens matched, append bonus token
            if len(accepted) == n_draft:
                accepted.append(bonus_ids_cpu[req_idx])

            for i, tok in enumerate(accepted):
                output_cpu[req_idx, i] = tok

        output = output_cpu.to(device)

        return SamplerOutput(
            sampled_token_ids=output,
            logprobs_tensors=None,
        )

    def _hybrid_accept_one(
        self,
        draft_tokens: list[int],
        target_tokens: list[int],
    ) -> list[int]:
        """Hybrid accept for a single request.

        Uses LCS alignment + left-to-right scan with bail-out.
        Returns the list of accepted/corrected token IDs.
        """
        chunk_len = min(len(draft_tokens), len(target_tokens))
        if chunk_len == 0:
            return []

        if self.strategy == "stop_at_first":
            # Simple prefix match — accept until first mismatch,
            # then one correction (the target's argmax)
            for i in range(chunk_len):
                if draft_tokens[i] != target_tokens[i]:
                    return draft_tokens[:i] + [target_tokens[i]]
            # All matched — return all draft tokens
            # (no bonus token here; bonus is handled elsewhere
            # for stop_at_first via the standard rejection path)
            return list(draft_tokens[:chunk_len])

        # ── Hybrid strategy ───────────────────────────────
        # LCS alignment to find which positions match
        matcher = IncrementalLCSMatcher(draft_tokens[:chunk_len])
        accepted_spans, _ = matcher.add_chunk(target_tokens[:chunk_len])

        # Build match mask
        matched: set[int] = set()
        for s, e in accepted_spans:
            for pos in range(s, e):
                matched.add(pos)

        # Left-to-right scan with bail-out
        output: list[int] = []
        consec_reject = 0

        for pos in range(chunk_len):
            if pos in matched:
                # LCS-matched: accept the draft token
                output.append(draft_tokens[pos])
                consec_reject = 0
            else:
                consec_reject += 1
                if consec_reject >= self.max_reject:
                    # Bail out: discard the rejected run,
                    # keep output up to start of rejected run
                    # + 1 correction
                    bail_start = pos - self.max_reject + 1
                    output = output[:bail_start]
                    output.append(target_tokens[bail_start])
                    break
                # Include correction from target
                output.append(target_tokens[pos])

        return output

    def load_model(self, *args: Any, **kwargs: Any) -> None:
        """No model to load."""
        pass
