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

from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.worker.gpu_input_batch import InputBatch

logger = init_logger(__name__)


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

    Tokenizes once upfront, then serves chunks via a simple token
    cursor. After verification, the cursor is advanced using LCS
    alignment to skip past OCR insertions/deletions.
    """

    def __init__(self, draft_text: str, tokenizer: Any):
        self.draft_text = draft_text
        self.draft_ids: list[int] = tokenizer.encode(
            draft_text, add_special_tokens=False
        )
        self.cursor: int = 0

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
        self._providers: dict[str, ParsedDraftProvider] = {}
        # Track which draft tokens were proposed per request so
        # cursor advancement can use LCS alignment on the next call.
        self._last_proposed: dict[str, list[int]] = {}

    def propose(
        self,
        sampled_token_ids: list[list[int]],
        requests: dict[str, Any],
        input_batch: InputBatch,
        tokenizer: Any,
    ) -> list[list[int]]:
        """Return draft token IDs per request from parsed text cursors.

        Args:
            sampled_token_ids: Per-request accepted token ids from last
                step. Empty list means partial prefill (skip).
            requests: Dict mapping req_id → CachedRequestState.
            input_batch: Current InputBatch with req_ids and metadata.
            tokenizer: Tokenizer for converting draft_text to token ids.

        Returns:
            list[list[int]]: Draft token ids per request. Empty list for
                requests without draft_text or exhausted drafts.
        """
        draft_token_ids: list[list[int]] = []
        num_reqs = input_batch.num_reqs

        for i in range(num_reqs):
            req_id = input_batch.req_ids[i]
            sampled = sampled_token_ids[i]

            if not sampled:
                # Partial prefill — skip spec decode.
                draft_token_ids.append([])
                continue

            provider = self._get_or_create_provider(req_id, requests, tokenizer)
            if provider is None or provider.is_exhausted():
                draft_token_ids.append([])
                continue

            # Advance cursor from previous step's verification.
            # sampled contains the accepted tokens (prefix + bonus).
            last_draft = self._last_proposed.pop(req_id, None)
            if last_draft is not None and sampled:
                advance = _lcs_draft_advance(last_draft, sampled)
                provider.advance(max(advance, 1))

            if provider.is_exhausted():
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

    def _get_or_create_provider(
        self,
        req_id: str,
        requests: dict[str, Any],
        tokenizer: Any,
    ) -> ParsedDraftProvider | None:
        """Get existing provider or create one from request's
        draft_text."""
        if req_id in self._providers:
            return self._providers[req_id]

        req_state = requests.get(req_id)
        if req_state is None:
            return None

        # Extract draft_text from sampling_params.extra_args
        sp = getattr(req_state, "sampling_params", None)
        if sp is None:
            return None
        extra = getattr(sp, "extra_args", None)
        if extra is None:
            return None
        draft_text = extra.get("draft_text")
        if not draft_text:
            return None

        provider = ParsedDraftProvider(draft_text, tokenizer)
        self._providers[req_id] = provider
        logger.debug(
            "Created ParsedDraftProvider for req %s: %d draft tokens",
            req_id,
            len(provider.draft_ids),
        )
        return provider

    def load_model(self, *args: Any, **kwargs: Any) -> None:
        """No model to load."""
        pass
