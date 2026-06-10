# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Parsed-Draft Speculative Decoding for OCR VL Models.

Uses parsed text (e.g. extracted from PDF via mupdf) as a "draft" for
speculative decoding, with a VL (Vision-Language) model as the target
verifier. The draft is free (no neural draft model needed).

Two decoding strategies:
  - stop_at_first (default): Stop at first mismatch, take correction,
    re-draft. Guarantees identical output to autoregressive decoding.
    Uses LCS alignment to advance draft cursor past insertions/deletions.
  - whole_chunk: Verify entire chunk, accept LCS-matched tokens, correct
    rest from verifier logits. Faster but approximate — corrections at
    rejected positions have stale KV context.
"""

from __future__ import annotations

import torch


class IncrementalLCSMatcher:
    """Incremental LCS-based matcher for speculative decoding.

    The draft is fixed upfront. GT tokens arrive incrementally in chunks.
    We maintain a single DP row and extend it column-by-column as new GT
    tokens arrive, so each new GT token costs O(M) work where M = len(draft).

    After processing a chunk of GT tokens, we backtrack through the DP table
    to identify which GT positions are LCS-matched (accepted) vs unmatched
    (rejected), returning spans in the same format as scan_multi_span.

    State:
        dp_row: 1-D array of length M+1, representing dp[*][j] for the
                latest GT position j processed so far.
        dp_history: list of dp rows (kept for backtracking). Can be cleared
                    after each chunk if spans are extracted per-chunk.
    """

    def __init__(self, draft_tokens: list[int]):
        self.draft = draft_tokens
        self.M = len(draft_tokens)
        # dp_row[i] = LCS length of draft[0..i-1] vs all gt tokens seen so far
        self.dp_row = [0] * (self.M + 1)
        # History of dp rows for backtracking. Index 0 = before any gt token.
        self.dp_history: list[list[int]] = [[0] * (self.M + 1)]
        self.gt_so_far: list[int] = []

    def add_chunk(
        self,
        gt_chunk_tokens: list[int],
    ) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        """Process a chunk of GT tokens and return accepted/rejected spans.

        Spans are in coordinates relative to the GT chunk (0-based within chunk).

        Args:
            gt_chunk_tokens: New GT tokens from verifier logits (argmax).

        Returns:
            accepted_spans, rejected_spans: half-open intervals within the chunk.
        """
        chunk_start = len(self.gt_so_far)

        # Extend DP table column by column
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

        # Backtrack over the chunk portion to find which GT positions are matched
        chunk_len = len(gt_chunk_tokens)
        gt_matched = set()

        # Backtrack from dp_history[chunk_start + chunk_len][M]
        # to dp_history[chunk_start][?].
        i = self.M
        j = chunk_start + chunk_len  # index into dp_history

        while i > 0 and j > chunk_start:
            if (
                self.draft[i - 1] == self.gt_so_far[j - 1]
                and self.dp_history[j][i] == self.dp_history[j - 1][i - 1] + 1
            ):
                gt_matched.add(j - 1 - chunk_start)  # chunk-relative
                i -= 1
                j -= 1
                continue
            if self.dp_history[j - 1][i] >= self.dp_history[j][i - 1]:
                j -= 1
            else:
                i -= 1

        # Convert matched set to accepted/rejected spans
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

    def find_draft_match_position(self, gt_token: int, search_start: int = 0) -> int:
        """Find where a GT token matches in the draft, starting from search_start.

        Used by stop-at-first-mismatch to advance the draft cursor past
        insertions/deletions after a correction.

        Args:
            gt_token: The correction token from verifier.
            search_start: Start searching from this draft index.

        Returns:
            Draft index where gt_token matches, or -1 if not found.
        """
        for i in range(search_start, self.M):
            if self.draft[i] == gt_token:
                return i
        return -1

    def reset_history(self):
        """Clear backtrack history to save memory. Keeps dp_row for continuing."""
        self.dp_history = [list(self.dp_row)]
        self.gt_so_far = []


def scan_lcs(
    draft_tokens: list[int],
    logits: torch.Tensor,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """LCS-based multi-span scan — finds optimal token alignment.

    Uses LCS alignment between draft and target tokens instead of positional
    matching. This finds the maximum possible set of tokens that can be
    accepted (in order) from the draft.

    Args:
        draft_tokens: List of draft token ids.
        logits: [n, vocab_size] logits from VL model.

    Returns:
        accepted_spans, rejected_spans in gt coordinates.
    """
    n = logits.shape[0]
    if n == 0 or len(draft_tokens) == 0:
        return [], []

    predicted = torch.argmax(logits, dim=-1).cpu().tolist()
    matcher = IncrementalLCSMatcher(draft_tokens)
    return matcher.add_chunk(predicted)


def find_prefix_match(
    draft_tokens: list[int],
    logits: torch.Tensor,
) -> int:
    """Find the length of the matching prefix between draft and verifier output.

    Used by stop-at-first-mismatch strategy.

    Args:
        draft_tokens: Draft token ids for this chunk.
        logits: [chunk_len, vocab_size] logits from verifier.

    Returns:
        Number of consecutive matching tokens from the start (0 if first mismatches).
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


def _find_draft_skip(draft_tokens: list[int], correction: int, start: int) -> int:
    """Find how far to advance draft cursor when no prefix matched.

    Searches for the correction token in the draft starting from `start`.
    If found, skip past it (the correction replaces it). If not found,
    skip 1 token (the mismatched position).

    Returns:
        Number of draft tokens to skip.
    """
    for i in range(start, len(draft_tokens)):
        if draft_tokens[i] == correction:
            return i + 1  # skip past the matching position
    return 1  # fallback: skip 1


def _lcs_draft_advance(draft_tokens: list[int], gt_prefix: list[int]) -> int:
    """Find how many draft tokens are consumed by matching a gt prefix via LCS.

    Runs a small LCS between draft_tokens and gt_prefix, then finds the
    draft position of the last matched token + 1.

    Example:
        draft:     [A, X, B, C, D, ...]   (X is OCR insertion)
        gt_prefix: [A, B, C, Y]           (A,B,C matched, Y is correction)

        LCS matches A(0), B(2), C(3) in draft → last matched at draft pos 3
        Advance = 3 + 1 = 4 (skip past matched region + the mismatch)

    Returns:
        Number of draft tokens to advance.
    """
    M = len(draft_tokens)
    N = len(gt_prefix)
    if M == 0 or N == 0:
        return 1

    # Build DP table
    dp = [[0] * (N + 1) for _ in range(M + 1)]
    for i in range(1, M + 1):
        for j in range(1, N + 1):
            if draft_tokens[i - 1] == gt_prefix[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Backtrack to find last matched draft position
    last_draft_pos = 0
    i, j = M, N
    while i > 0 and j > 0:
        if draft_tokens[i - 1] == gt_prefix[j - 1] and dp[i][j] == dp[i - 1][j - 1] + 1:
            last_draft_pos = max(last_draft_pos, i - 1)
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return last_draft_pos + 1  # advance past last matched position


class ParsedDraftProvider:
    """
    Manages parsed text as a draft token source.

    Tokenizes once upfront, then serves chunks via a simple token cursor.

    Args:
        parsed_text: Text parsed/extracted from PDF (e.g. via mupdf).
        tokenizer: HuggingFace-compatible tokenizer with encode/decode methods.
    """

    def __init__(self, parsed_text: str, tokenizer):
        self.parsed_text = parsed_text
        self.tokenizer = tokenizer
        self.draft_ids: list[int] = tokenizer.encode(
            parsed_text, add_special_tokens=False
        )
        self.cursor: int = 0  # token-level cursor

    def get_next_chunk(self, max_tokens: int) -> list[int]:
        """Return next chunk of draft token ids from cursor position."""
        end = min(self.cursor + max_tokens, len(self.draft_ids))
        return self.draft_ids[self.cursor : end]

    def advance(self, n_consumed: int) -> None:
        """Advance cursor by n_consumed draft tokens."""
        self.cursor += n_consumed

    def is_exhausted(self) -> bool:
        """Return True if all draft tokens have been consumed."""
        return self.cursor >= len(self.draft_ids)

    @property
    def remaining_tokens(self) -> int:
        """Number of draft tokens not yet consumed."""
        return len(self.draft_ids) - self.cursor

    def __repr__(self) -> str:
        return (
            f"ParsedDraftProvider("
            f"cursor={self.cursor}/{len(self.draft_ids)}, "
            f"remaining={self.remaining_tokens})"
        )


class ParsedDraftSpeculativeDecoder:
    """
    Speculative decoder that uses parsed text as draft for a VL target model.

    Three strategies:
      - stop_at_first (default): Accept matching prefix + 1 correction,
        truncate KV cache, advance draft cursor via LCS alignment, repeat.
        Guarantees identical output to autoregressive decoding.
      - whole_chunk: Verify entire chunk, accept all LCS-matched tokens,
        correct rest from verifier logits. Faster but approximate — KV
        cache becomes stale at rejected positions.
      - hybrid: Like whole_chunk, but bails out after max_reject consecutive
        rejected tokens. Accepts up to that point, truncates KV cache, and
        re-drafts (like stop_at_first). Balances speed and KV quality.

    Args:
        vl_model: Vision-Language model with forward() method.
        tokenizer: HuggingFace-compatible tokenizer.
        chunk_size: Number of draft tokens to verify per forward pass.
        strategy: "stop_at_first" (default), "whole_chunk", or "hybrid".
        max_reject: Maximum consecutive rejected tokens before bailing out
            and re-drafting. Only used in "hybrid" strategy. Default 3.
    """

    def __init__(
        self,
        vl_model,
        tokenizer,
        chunk_size: int = 50,
        strategy: str = "stop_at_first",
        max_reject: int = 3,
    ):
        self.vl_model = vl_model
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.strategy = strategy
        self.max_reject = max_reject

    @property
    def device(self):
        """Infer device from VL model parameters."""
        try:
            return next(self.vl_model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @torch.no_grad()
    def decode(
        self,
        image: torch.Tensor,
        parsed_text: str,
        prompt_ids: torch.Tensor | None = None,
        max_new_tokens: int = 512,
    ) -> list[int]:
        """
        Speculative decode using parsed text as draft.

        Args:
            image: Image tensor for the VL model's vision encoder.
            parsed_text: Text parsed/extracted from PDF for this text block.
            prompt_ids: Optional prompt/instruction token ids to prepend.
            max_new_tokens: Maximum output tokens.

        Returns:
            List of generated token ids.
        """
        past_key_values = self._prefill_image(image, prompt_ids)
        return self.decode_with_cache(past_key_values, parsed_text, max_new_tokens)

    @torch.no_grad()
    def decode_with_cache(
        self,
        past_key_values,
        parsed_text: str,
        max_new_tokens: int = 512,
    ) -> list[int]:
        """Speculative decode using a pre-built KV cache (skip prefill).

        Use this when sharing a single prefill across multiple strategy runs:
            cache = decoder._prefill_image(image, prompt)
            for strategy in strategies:
                decoder.strategy = strategy
                ids = decoder.decode_with_cache(copy.deepcopy(cache), text)

        Args:
            past_key_values: KV cache from _prefill_image().
            parsed_text: Parsed text for this block.
            max_new_tokens: Maximum output tokens.

        Returns:
            List of generated token ids.
        """
        if self.strategy == "whole_chunk":
            return self._decode_whole_chunk_inner(
                past_key_values, parsed_text, max_new_tokens
            )
        elif self.strategy == "hybrid":
            return self._decode_hybrid_inner(
                past_key_values, parsed_text, max_new_tokens
            )
        else:
            return self._decode_stop_at_first_inner(
                past_key_values, parsed_text, max_new_tokens
            )

    def _decode_stop_at_first(
        self,
        image: torch.Tensor,
        parsed_text: str,
        prompt_ids: torch.Tensor | None = None,
        max_new_tokens: int = 512,
    ) -> list[int]:
        """Stop-at-first-mismatch strategy (greedy-exact). See decode()."""
        past_key_values = self._prefill_image(image, prompt_ids)
        return self._decode_stop_at_first_inner(
            past_key_values, parsed_text, max_new_tokens
        )

    def _decode_stop_at_first_inner(
        self,
        past_key_values,
        parsed_text: str,
        max_new_tokens: int = 512,
    ) -> list[int]:
        """Stop-at-first-mismatch strategy (greedy-exact).

        For each chunk:
          1. Feed draft tokens to verifier
          2. Accept matching prefix
          3. Take correction at first mismatch (+1 token)
          4. Truncate KV cache to prefix + correction
          5. Advance draft cursor using LCS alignment to skip past
             insertions/deletions in the draft
          6. Repeat

        Uses LCS to find how many draft tokens were "consumed" by the
        accepted prefix, so the cursor skips past OCR insertions that
        would otherwise break positional alignment.
        """
        draft_provider = ParsedDraftProvider(parsed_text, self.tokenizer)
        output_ids: list[int] = []

        while not draft_provider.is_exhausted() and len(output_ids) < max_new_tokens:
            draft_tokens = draft_provider.get_next_chunk(self.chunk_size)
            if not draft_tokens:
                break

            draft_tensor = torch.tensor(
                [draft_tokens], dtype=torch.long, device=self.device
            )

            # Verify
            logits = self._verify_chunk(draft_tensor, past_key_values)
            gt_tokens = torch.argmax(logits[0], dim=-1).cpu().tolist()

            # Find prefix match using LCS alignment
            # LCS tells us which draft positions correspond to which gt positions
            matcher = IncrementalLCSMatcher(draft_tokens)
            accepted_spans, rejected_spans = matcher.add_chunk(gt_tokens)

            if not accepted_spans or accepted_spans[0][0] != 0:
                # First gt token is rejected — no prefix match at all
                # Take correction for first position and advance draft by 1
                correction = gt_tokens[0]
                output_ids.append(correction)

                # Truncate KV to remove ALL draft entries (none were correct)
                n_keep = 0
                self._truncate_kv_cache(past_key_values, n_keep, len(draft_tokens))
                # Feed correction to create a clean KV entry
                self._generate_one(correction, past_key_values)

                # Find where correction token appears in draft to skip past
                # the misaligned region
                skip = _find_draft_skip(draft_tokens, correction, 0)
                draft_provider.advance(skip)
            else:
                # We have a prefix match starting at gt pos 0
                first_acc_end = accepted_spans[0][1]  # end of first accepted span

                # The accepted prefix in gt coordinates = draft tokens that matched
                # We need to find how many draft tokens were consumed for this prefix
                # via LCS backtracking
                prefix_len = first_acc_end  # gt positions accepted

                if prefix_len == len(gt_tokens):
                    # Entire chunk matched
                    output_ids.extend(gt_tokens)
                    draft_provider.advance(len(draft_tokens))
                else:
                    # Accept prefix + take correction at first mismatch
                    output_ids.extend(gt_tokens[:prefix_len])
                    correction = gt_tokens[prefix_len]
                    output_ids.append(correction)

                    # Truncate KV to only the accepted prefix (not +1),
                    # because the KV at prefix_len has the WRONG draft token.
                    n_keep = prefix_len
                    self._truncate_kv_cache(past_key_values, n_keep, len(draft_tokens))
                    # Feed correction to create a clean KV entry
                    self._generate_one(correction, past_key_values)

                    # Find how many draft tokens were consumed by the LCS-matched
                    # prefix. The LCS backtrack tells us which draft positions matched;
                    # we need to advance past the last matched draft position + 1
                    # (to skip the mismatched draft token too).
                    draft_advance = _lcs_draft_advance(
                        draft_tokens, gt_tokens[: prefix_len + 1]
                    )
                    draft_provider.advance(draft_advance)

            # Check EOS
            if self.tokenizer.eos_token_id in output_ids:
                eos_pos = output_ids.index(self.tokenizer.eos_token_id)
                return output_ids[: eos_pos + 1]

        # Autoregressive fallback
        while len(output_ids) < max_new_tokens:
            if not output_ids:
                break
            logits = self._generate_one(output_ids[-1], past_key_values)
            next_token = torch.argmax(logits).item()
            if next_token == self.tokenizer.eos_token_id:
                output_ids.append(next_token)
                break
            output_ids.append(next_token)

        return output_ids

    def _decode_whole_chunk(
        self,
        image: torch.Tensor,
        parsed_text: str,
        prompt_ids: torch.Tensor | None = None,
        max_new_tokens: int = 512,
    ) -> list[int]:
        """Verify-whole-chunk strategy (fast, approximate). See decode()."""
        past_key_values = self._prefill_image(image, prompt_ids)
        return self._decode_whole_chunk_inner(
            past_key_values, parsed_text, max_new_tokens
        )

    def _decode_whole_chunk_inner(
        self,
        past_key_values,
        parsed_text: str,
        max_new_tokens: int = 512,
    ) -> list[int]:
        """Verify-whole-chunk strategy (fast, approximate).

        Verifies entire chunk, accepts LCS-matched tokens, uses verifier's
        argmax for corrections at rejected positions. All tokens produced
        per verify pass.

        WARNING: Corrections at rejected positions have stale KV context
        (the verifier saw draft tokens, not the correct ones). Output may
        differ from autoregressive. Only use when image anchor hypothesis
        holds (VL model driven by image, not text prefix).
        """
        draft_provider = ParsedDraftProvider(parsed_text, self.tokenizer)
        output_ids: list[int] = []

        lcs_matcher = IncrementalLCSMatcher(draft_provider.draft_ids)

        while not draft_provider.is_exhausted() and len(output_ids) < max_new_tokens:
            draft_tokens = draft_provider.get_next_chunk(self.chunk_size)
            if not draft_tokens:
                break

            draft_tensor = torch.tensor(
                [draft_tokens], dtype=torch.long, device=self.device
            )

            logits = self._verify_chunk(draft_tensor, past_key_values)

            # LCS alignment
            gt_tokens = torch.argmax(logits[0], dim=-1).cpu().tolist()
            accepted_spans, rejected_spans = lcs_matcher.add_chunk(gt_tokens)
            lcs_matcher.reset_history()

            # Build output
            output_tokens: list[int] = []
            n_draft_consumed = 0

            all_events = [(s, e, "accept") for s, e in accepted_spans] + [
                (s, e, "reject") for s, e in rejected_spans
            ]
            all_events.sort(key=lambda x: x[0])

            for start, end, kind in all_events:
                if kind == "accept":
                    output_tokens.extend(draft_tokens[start:end])
                else:
                    for j in range(start, end):
                        correction = torch.argmax(logits[0, j]).item()
                        output_tokens.append(correction)
                n_draft_consumed = max(n_draft_consumed, end)

            output_ids.extend(output_tokens)
            self._truncate_kv_cache(
                past_key_values, n_draft_consumed, len(draft_tokens)
            )
            draft_provider.advance(n_draft_consumed)

            if self.tokenizer.eos_token_id in output_ids:
                eos_pos = output_ids.index(self.tokenizer.eos_token_id)
                return output_ids[: eos_pos + 1]

        # Autoregressive fallback
        while len(output_ids) < max_new_tokens:
            if not output_ids:
                break
            logits = self._generate_one(output_ids[-1], past_key_values)
            next_token = torch.argmax(logits).item()
            if next_token == self.tokenizer.eos_token_id:
                output_ids.append(next_token)
                break
            output_ids.append(next_token)

        return output_ids

    def _decode_hybrid(
        self,
        image: torch.Tensor,
        parsed_text: str,
        prompt_ids: torch.Tensor | None = None,
        max_new_tokens: int = 512,
    ) -> list[int]:
        """Hybrid strategy. See decode()."""
        past_key_values = self._prefill_image(image, prompt_ids)
        return self._decode_hybrid_inner(past_key_values, parsed_text, max_new_tokens)

    def _decode_hybrid_inner(
        self,
        past_key_values,
        parsed_text: str,
        max_new_tokens: int = 512,
    ) -> list[int]:
        """Hybrid strategy: whole-chunk with bail-out on excessive rejections.

        Verifies entire chunk with LCS alignment, but scans the accepted/rejected
        spans left-to-right and bails out when max_reject consecutive rejected
        tokens are encountered. At that point:
          - Output accepted + corrected tokens up to the bail-out point
          - Truncate KV cache to that point
          - Re-draft (like stop_at_first)

        This limits KV cache staleness: after max_reject consecutive mismatches,
        the context is too polluted to trust further corrections. Re-drafting
        gives the verifier fresh context.

        The key insight: a long rejected span likely means a multi-token
        insertion (gt has tokens the draft doesn't), so the verifier can't
        produce the right tokens anyway — it only has 1 logit per draft position.

        Args:
            max_reject: set via self.max_reject (constructor param). Default 3.
        """
        draft_provider = ParsedDraftProvider(parsed_text, self.tokenizer)
        output_ids: list[int] = []

        while not draft_provider.is_exhausted() and len(output_ids) < max_new_tokens:
            draft_tokens = draft_provider.get_next_chunk(self.chunk_size)
            if not draft_tokens:
                break

            draft_tensor = torch.tensor(
                [draft_tokens], dtype=torch.long, device=self.device
            )

            logits = self._verify_chunk(draft_tensor, past_key_values)
            gt_tokens = torch.argmax(logits[0], dim=-1).cpu().tolist()

            # LCS alignment
            matcher = IncrementalLCSMatcher(draft_tokens)
            accepted_spans, rejected_spans = matcher.add_chunk(gt_tokens)

            # Build match mask for left-to-right scan
            chunk_len = len(gt_tokens)
            matched = set()
            for s, e in accepted_spans:
                for pos in range(s, e):
                    matched.add(pos)

            # Scan left-to-right, bail out on max_reject consecutive rejections
            output_tokens: list[int] = []
            consec_reject = 0
            bail_pos = chunk_len  # default: no bail-out

            for pos in range(chunk_len):
                if pos in matched:
                    output_tokens.append(draft_tokens[pos])
                    consec_reject = 0
                else:
                    consec_reject += 1
                    if consec_reject >= self.max_reject:
                        # Bail out: keep everything before the rejected run,
                        # plus 1 correction token (like stop-at-first)
                        bail_pos = pos - self.max_reject + 1 + 1  # +1 for correction
                        output_tokens = output_tokens[: bail_pos - 1]
                        output_tokens.append(gt_tokens[pos - self.max_reject + 1])
                        break
                    else:
                        # Haven't hit limit yet, include as correction
                        output_tokens.append(gt_tokens[pos])

            output_ids.extend(output_tokens)

            # Truncate KV cache and advance draft
            n_keep = len(output_tokens)
            self._truncate_kv_cache(past_key_values, n_keep, len(draft_tokens))

            if bail_pos < chunk_len:
                # Bailed out — advance draft using LCS alignment up to bail point
                draft_advance = (
                    _lcs_draft_advance(draft_tokens, gt_tokens[:bail_pos])
                    if bail_pos > 0
                    else 1
                )
                draft_provider.advance(max(draft_advance, 1))
            else:
                # Full chunk consumed
                draft_provider.advance(len(draft_tokens))

            # Check EOS
            if self.tokenizer.eos_token_id in output_ids:
                eos_pos = output_ids.index(self.tokenizer.eos_token_id)
                return output_ids[: eos_pos + 1]

        # Autoregressive fallback
        while len(output_ids) < max_new_tokens:
            if not output_ids:
                break
            logits = self._generate_one(output_ids[-1], past_key_values)
            next_token = torch.argmax(logits).item()
            if next_token == self.tokenizer.eos_token_id:
                output_ids.append(next_token)
                break
            output_ids.append(next_token)

        return output_ids

    # ------------------------------------------------------------------
    # VL model interface stubs — override for specific model integration
    # ------------------------------------------------------------------

    def _prefill_image(
        self,
        image: torch.Tensor,
        prompt_ids: torch.Tensor | None = None,
    ):
        """Process image through VL encoder, return initial KV cache.

        Override this method for your specific VL model. Should:
        1. Run the vision encoder on the image
        2. Run the language model on prompt tokens (if any)
        3. Return past_key_values for subsequent decoding

        Returns:
            past_key_values: Model-specific KV cache object.
        """
        raise NotImplementedError(
            "Subclass ParsedDraftSpeculativeDecoder and implement _prefill_image "
            "for your VL model."
        )

    def _verify_chunk(
        self,
        draft_tensor: torch.Tensor,
        past_key_values,
    ) -> torch.Tensor:
        """Run target VL model forward pass over draft tokens.

        Override this method for your specific VL model. Should:
        1. Feed draft_tensor as decoder input with existing KV cache
        2. Return logits for each position
        3. Update past_key_values in-place with new KV entries

        Args:
            draft_tensor: [1, chunk_len] token ids.
            past_key_values: KV cache from previous steps.

        Returns:
            logits: [1, chunk_len, vocab_size] tensor.
        """
        raise NotImplementedError(
            "Subclass ParsedDraftSpeculativeDecoder and implement _verify_chunk "
            "for your VL model."
        )

    def _truncate_kv_cache(
        self,
        past_key_values,
        n_keep: int,
        n_total: int,
    ) -> None:
        """Truncate KV cache to keep only accepted entries.

        After verification, the KV cache has entries for all draft tokens.
        We need to discard entries beyond n_keep.

        Args:
            past_key_values: KV cache to modify in-place.
            n_keep: Number of new entries to keep (accepted prefix + correction).
            n_total: Total number of new entries added during verification.
        """
        raise NotImplementedError(
            "Subclass ParsedDraftSpeculativeDecoder and implement "
            "_truncate_kv_cache for your VL model."
        )

    def _generate_one(
        self,
        last_token_id: int,
        past_key_values,
    ) -> torch.Tensor:
        """Generate logits for next token autoregressively.

        Fallback for when parsed draft is exhausted but generation
        should continue.

        Args:
            last_token_id: The most recently generated token.
            past_key_values: Current KV cache.

        Returns:
            logits: [vocab_size] tensor.
        """
        raise NotImplementedError(
            "Subclass ParsedDraftSpeculativeDecoder and implement _generate_one "
            "for your VL model."
        )
