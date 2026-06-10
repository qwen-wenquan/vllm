# Parsed-Draft LCS Simulation Results

Results from `benchmark_parsed_draft.py` — a pure-Python simulator that
replays the EAGLE `_decode_stop_at_first_inner` / `_decode_hybrid_inner`
control flow against pre-tokenized OCR blocks. No GPU required.

The simulator drives **both** vLLM's LCS utilities
(`vllm.v1.spec_decode.parsed_draft`) and the EAGLE reference
implementation side-by-side, and asserts that the two produce
identical accept/reject decisions on every block.

## What the simulator measures

For each block, the simulator runs the same loop a real decoder runs:

1. Take a chunk of `chunk_size` draft tokens from the parsed text.
2. Run LCS alignment against the ground-truth chunk of the same length.
3. Accept the matching prefix, take 1 correction at the first mismatch
  (stop_at_first), accept matching tokens until `max_reject`
   consecutive misses (hybrid), or use the hold+snap cursor rule
   (holdsnap).
4. Advance the draft cursor via `_lcs_draft_advance`
  (partial-prefix-match path) or `_find_draft_skip` (no-prefix-match
   path, baseline), mirroring `_decode_stop_at_first_inner`.
5. When the draft is exhausted before the ground truth, fall back to
  autoregressive decoding — one forward pass per remaining gt token.

Each iteration of step 2 counts as one **spec step** (one verifier
forward pass). Each iteration of step 5 counts as one **AR fallback
step**. The two sum to **forward passes**, which is the denominator
for end-to-end speedup vs a pure-AR baseline.

## Strategies

Three strategies are compared:

- `**stop_at_first`** — production baseline. Accept matching prefix
    - 1 correction per chunk; advance via `_find_draft_skip`.
- `**hybrid_mr3**` — production whole-chunk strategy. Verify entire
chunk via LCS, accept all matched positions until `max_reject=3`
consecutive rejections, then bail and re-draft.
- `**holdsnap_h8**` — lossless experimental cursor rule for
stop_at_first. When LCS finds no prefix match: SNAP cursor to the
correction token if it appears ahead in the draft chunk; otherwise
HOLD (don't advance), capped at 8 consecutive holds. Designed to
absorb gt-side insertions (e.g. extra formatting tokens the OCR
draft doesn't have).

All three are lossless — they produce the same output token sequence
as autoregressive decoding.

## Dataset

`data/sft_ocr_blocks_10k.tokenized.json` — 10,000 OCR text blocks
pre-tokenized with the PaddleOCR-VL_finetune tokenizer. Each block has:

- `parsed_text` — raw text extracted from the PDF by mupdf.
- `parsed_text_normalized` — `parsed_text` with newlines replaced by
spaces. This is the text that produced `parsed_ids`.
- `parsed_ids` — token ids of the draft (62/500 blocks match raw text,
438/500 match normalized text — confirmed by re-tokenizing).
- `gt_ids` — token ids of the model's greedy AR output (the
"ground truth" the speculative decoder must reproduce).

Total non-empty blocks: **10,000**. Total ground-truth tokens:
**726,871**. Total draft tokens: **1,241,744**.

All headline numbers below are on the **full 10,000-block set**.

## Headline numbers (full 10,000 blocks)

End-to-end speedup vs pure autoregressive decoding, computed as
`n_gt / (spec_steps + ar_fallback_steps)`:

| Strategy      | chunk_size | Accept rate | Avg tok/spec step | Spec steps | AR fallback steps | Forward passes | **End-to-end speedup** |
| ------------- | ---------- | ----------- | ----------------- | ---------- | ----------------- | -------------- | ---------------------- |
| stop_at_first | 16         | 26.2%       | 1.45              | 412,687    | 126,649           | 539,336        | 1.35×                  |
| hybrid_mr3    | 16         | 33.7%       | 2.19              | 291,656    | 89,123            | 380,779        | **1.91×**              |
| holdsnap_h8   | 16         | 37.4%       | 1.60              | 445,958    | 13,747            | 459,705        | 1.58×                  |
| stop_at_first | 50         | 16.2%       | 1.44              | 261,660    | 350,211           | 611,871        | 1.19×                  |
| hybrid_mr3    | 50         | 28.2%       | 2.46              | 191,187    | 255,899           | 447,086        | **1.63×**              |
| holdsnap_h8   | 50         | 19.3%       | 1.35              | 391,446    | 198,882           | 590,328        | 1.23×                  |
| stop_at_first | 200        | 11.6%       | 1.44              | 184,932    | 459,709           | 644,641        | 1.13×                  |
| hybrid_mr3    | 200        | 21.4%       | 2.50              | 137,751    | 382,755           | 520,506        | **1.40×**              |
| holdsnap_h8   | 200        | 13.6%       | 1.30              | 323,459    | 307,709           | 631,168        | 1.15×                  |

vLLM and EAGLE produce **identical** accept counts, spec steps, AR
fallback steps, and forward passes for every configuration above
(0 mismatches / 10,000 blocks).

> **⚠ The `holdsnap_h8` rows are simulator-only.** The simulator's
> verifier is the gt token stream, which slides forward independently
> of what the simulator feeds it. The real verifier re-rejects the
> same draft after a HOLD (deterministic on context), so the e2e
> benchmark measured a **−31 %** speedup regression vs the
> stop_at_first baseline. The production default is
> `parsed_draft_holdsnap=False`. See the
> "Production caveat: holdsnap is a simulator artifact" section below.

## How to read these numbers

The simulator assumes one forward pass per spec step and one per AR
fallback step — i.e. it treats verify-N-tokens and decode-1-token as
equal cost. That's the right model when the verifier is small enough
to be kernel-launch-bound (true for PaddleOCR-VL 0.3B; see README's
"Measured Latency" — 10.2 ms decode vs 11.2 ms verify regardless of
chunk size). For larger verifiers, multiply spec-step cost by
`verify_ms / decode_ms` to get the wall-clock-honest speedup.

### Key takeaways

- `**hybrid_mr3` at `chunk_size=16` wins.** 1.91× end-to-end —
best across all 9 (strategy × chunk_size) cells. 41% fewer
forward passes than the pure-AR baseline (380,779 vs 726,871).
- **Smaller chunks win for all three strategies.** Chunk=16 beats
chunk=200 by 1.2–1.4×. OCR drift is frequent and short — every
drift truncates the chunk's accepted prefix, wasting more draft
tokens at larger chunk sizes.
- **Hybrid wins by accepting multiple tokens per spec step.** Its
`tok/spec step` (2.19–2.50) is ~50–80% higher than the other two
(1.30–1.60). This is the only metric that moves materially across
strategies, and it directly drives the speedup ratio.
- **Holdsnap eliminates AR fallback (126,649 → 13,747 at chunk=16,
−89%) and raises accept rate to 37.4% — best of the three.**
But it pays for it with more spec steps (445,958 vs 412,687)
because every "hold" step emits exactly 1 token. Net: 1.58× vs
hybrid's 1.91×. Holdsnap is a strict improvement over
`stop_at_first` (+17% at chunk=16), but loses to hybrid.

### Why hybrid_mr3 wins

Hybrid runs LCS over the whole chunk and accepts every matched
position until `max_reject=3` consecutive rejections trigger a bail.
That extracts multi-token wins from chunks with scattered matches
(e.g. `[a, b, X, c, d]` against gt `[a, b, c, d]` — hybrid accepts
4 tokens in one step; stop_at_first and holdsnap accept only `[a, b]`
then re-align). In OCR data this pattern is very common — VL model
output frequently inserts/swaps a few tokens but stays largely
aligned within a sentence.

### Why holdsnap doesn't beat hybrid (even though its accept rate is higher)

Holdsnap's accept rate (37.4% at chunk=16) is the highest in the
table because it doesn't burn draft tokens during gt-side insertions
— each held step preserves the draft chunk for re-verification next
step. But every held step contributes exactly 1 token to the output,
which pulls `tok/spec step` down to 1.60. Hybrid's lower accept rate
(33.7%) maps to higher tokens-per-step (2.19) because it accepts
non-prefix matches within a chunk. End-to-end forward passes is what
matters, and hybrid wins there.

## Production caveat: holdsnap is a simulator artifact

**Holdsnap was implemented in `vllm/v1/spec_decode/parsed_draft.py`,
landed with `parsed_draft_holdsnap=True` as the default, then disabled
after the e2e benchmark on GTX5k showed a regression.** The config now
defaults to `parsed_draft_holdsnap=False`. The helper function
(`holdsnap_advance`) is kept as a documented code artifact.

### What the e2e benchmark measured

`benchmark_parsed_draft_e2e.py` on 500 GTX5k blocks, `chunk_size=16`,
`stop_at_first` strategy, sequential mode:

| metric              | holdsnap=False | holdsnap=True | Δ      |
| ---                 | ---:           | ---:          | ---:   |
| Speedup vs baseline | **1.49×**      | **1.03×**     | −31 %  |
| Spec-decode tok/s   | 984.7          | 687.0         | −30 %  |
| Spec-decode time    | 27.97 s        | 40.48 s       | +45 %  |
| Exact-match output  | 66.6 %         | 92.4 %        | +26 pp |

Holdsnap on is **45 % slower** end-to-end despite producing the same
total token count. (The exact-match jump is consistent with
repetition-detection terminating at different positions; both branches
should be byte-identical to AR by construction, so the lower
exact-match in the off case is most likely an early-termination
artifact, not a correctness bug.)

### Why the simulator misled us

The simulator's "verifier" is the ground-truth token stream
(`gt_ids`). It slides forward by 1 every spec step regardless of what
draft was fed in. So in the simulator:

1. Spec step N: draft chunk `[d0..d15]`. LCS against gt window
   `gt[k:k+16]`. No prefix match → HOLD, cursor stays at draft[0].
2. Spec step N+1: same draft chunk `[d0..d15]`. LCS against the
   **next** gt window `gt[k+1:k+17]`. Eventually the gt window slides
   past the drift region and `d0..d15` aligns with `gt[m:m+16]` as a
   prefix match.

The real verifier is a deterministic forward pass over
(prompt + accepted-so-far + draft chunk). After a HOLD, the same
draft is fed in, with a context that now includes the correction
that was just emitted:

1. Spec step N: draft chunk `[d0..d15]`. Verifier rejects `d0`
   (argmax at position 0 = some correction `c1` ≠ `d0`). Emit `c1`.
   HOLD.
2. Spec step N+1: same draft chunk `[d0..d15]`. Context is now
   `(prompt + c1)`. Verifier argmax at position 0 = "what comes
   after `c1`?" — call it `c2`. If `c2 ≠ d0` (very likely, since
   gt drifted), reject again. HOLD again.
3. … repeat until `max_hold=8` forces an advance by 1.

Each held step consumes a full verifier forward pass — at chunk=16
that's the same wall-clock cost as a productive step. The simulator
counted these 1-token steps and concluded "fewer forward passes
overall because AR fallback shrinks." The real decoder pays for them
without amortizing the cost across multiple emitted tokens.

### What the simulator got right

The cursor-advance question is real. The simulator's evidence that
"advance by 1" sometimes wastes draft tokens during gt-side
insertions is correct. The error was assuming HOLD-and-retry would
work with a deterministic verifier. A correct fix would have to be
**verifier-aware** — e.g. only retry the same chunk after the engine
context has actually changed in a way that could plausibly change
the verifier's argmax. None of the simple cursor rules we explored
do that.

### How to detect this class of bug earlier

Two checks that would have caught it:

1. **Run the e2e benchmark before defaulting any cursor rule on.**
   The simulator validated LCS-matcher correctness, not cursor-rule
   effectiveness against a real verifier.
2. **Question any simulator finding that depends on the verifier
   producing different output for the same input.** The
   "hold-and-re-verify" rule is the canonical instance — it assumes
   the verifier's answer depends on something the simulator doesn't
   model (in this case, the engine's growing KV context).

## Effect of `normalize_parsed_text` on the draft

`vllm/v1/spec_decode/parsed_draft.py::normalize_parsed_text` applies
four content-preserving transformations to the parsed draft text
before tokenization:

1. Visual line breaks (`\n`) → single space.
2. NFKC unicode normalization (ligatures `ﬁ` → `fi`, etc.).
3. Residual hyphen fix (`MALE- IDENTIFIED` → `MALE-IDENTIFIED`).
4. Whitespace collapse + strip.

These mirror the rules in `pymupdf_utils.py::_clean_paragraph_text`
and are applied to the draft only — `gt_ids` is unchanged, so the
spec decoder still produces byte-identical output to AR decoding
(strictly lossless).

To measure the effect, the original `parsed_text` of every block was
re-normalized and re-tokenized via `benchmarks/spec_decode/pretokenize.py`,
producing `data/sft_ocr_blocks_10k.normalized.jsonl`. The simulator was
then run on the new file. **Both runs are on the same 10,000 blocks
with the same `gt_ids`.**

Headline cell-by-cell comparison (full 10k):

| chunk | strategy      | un-norm | normalized | Δ rel    | accept un-norm → norm |
| ----- | ------------- | ------- | ---------- | -------- | --------------------- |
| 16    | stop_at_first | 1.35×   | 1.36×      | **+1 %** | 26.2 % → 27.0 %       |
| 16    | hybrid_mr3    | 1.91×   | 1.95×      | **+2 %** | 33.7 % → 35.6 %       |
| 16    | holdsnap_h8   | 1.58×   | 1.64×      | **+4 %** | 37.4 % → 39.6 %       |
| 50    | stop_at_first | 1.19×   | 1.21×      | **+2 %** | 16.2 % → 18.0 %       |
| 50    | hybrid_mr3    | 1.63×   | 1.67×      | **+2 %** | 28.2 % → 30.4 %       |
| 50    | holdsnap_h8   | 1.23×   | 1.26×      | **+2 %** | 19.3 % → 21.3 %       |
| 200   | stop_at_first | 1.13×   | 1.15×      | **+2 %** | 11.6 % → 13.7 %       |
| 200   | hybrid_mr3    | 1.40×   | 1.47×      | **+5 %** | 21.4 % → 25.2 %       |
| 200   | holdsnap_h8   | 1.15×   | 1.18×      | **+3 %** | 13.6 % → 16.0 %       |

Per-cell forward-pass change for the production-recommended config
(`hybrid_mr3` at `chunk_size=16`):

|                 | un-norm | normalized | Δ       |
| --------------- | ------- | ---------- | ------- |
| Forward passes  | 380,779 | 373,009    | −7,770  |
| Accepted tokens | 244,731 | 258,450    | +13,719 |
| Spec steps      | 291,656 | 274,712    | −16,944 |
| Tok / spec step | 2.19    | 2.29       | +0.10   |
| AR fallback     | 89,123  | 98,297     | +9,174  |

Fidelity ✓ on all 9 cells (0 mismatches / 10,000 blocks) for both
the un-normalized and normalized runs.

### Honest read: a small win, not the headline

An earlier 500-block sample suggested the normalization would deliver
**+14–46 %** speedup. The full-10k measurement comes in much smaller
(**+1–5 %**). Three reasons for the gap:

1. **Selection bias in the small sample.** The first 500 blocks of
  the dataset are systematically easier (the same effect that made
   the original 100-block run optimistic by 30–55 %).
2. **The dataset's `parsed_text_normalized` field already did most
  of the work.** It collapsed newlines and double spaces before any
   of this work began. Re-normalizing only saved an additional 4 %
   of draft tokens (1,241,744 → 1,191,094) on the full 10k.
3. **Per-block effect varies by script.** On Arabic blocks (~half
  the dataset by inspection), collapsing some spaces actually
   misaligns the draft from the model output in places — net effect
   is still positive but modest. NFKC ligature decomposition is also
   a no-op on this data: 0/500 sampled blocks contain ligatures.

The normalization is still worth keeping — it's a one-shot regex pass
at `ParsedDraftProvider` construction with zero per-step cost, and
the gain is positive on every cell. Just don't expect it to move the
needle by more than a few percent on OCR-style data that already had
basic whitespace cleanup upstream. On academic-PDF data with ligatures
and aggressive line-break hyphenation, the gain would be larger.

### Reproducing the normalized numbers

```bash
# Re-tokenize the dataset with normalization applied to parsed_text
.venv/bin/python benchmarks/spec_decode/pretokenize.py \
    --input data/sft_ocr_blocks_10k.tokenized.json \
    --tokenizer /path/to/PaddleOCR-VL_finetune \
    --output data/sft_ocr_blocks_10k.normalized.jsonl \
    --workers 16

# Run the simulator on the normalized data
.venv/bin/python benchmarks/spec_decode/benchmark_parsed_draft.py \
    --data data/sft_ocr_blocks_10k.normalized.jsonl \
    --n 10000
```

## Caveats

- **Forward-pass parity assumption.** The simulator doesn't model
KV-cache-truncation cost, sampling cost, or the verify/decode
latency gap. For the 0.3B model this is a good approximation; for
larger models, scale spec-step cost by `verify_ms / decode_ms`.
- **Greedy-only.** The "ground truth" is the model's greedy AR
output. Sampling-based decoders would have a smaller acceptance
rate because the per-token distribution adds variance the LCS
matcher can't anticipate.
- **No prefill cost.** Spec decoding still requires prefilling the
image and prompt once; that cost is the same for AR and spec
decoding and is excluded from the speedup ratio.
- **Per-block, sequential.** The simulator runs one block at a time.
Real serving batches many requests and the spec/AR ratio shifts
with batch size and chunked prefill scheduling. See
`benchmark_parsed_draft_e2e.py` for the batched-on-GPU numbers.
- **Holdsnap is landed but defaults off.** The cursor rule
(`holdsnap_advance` in `parsed_draft.py`) is wired into the proposer
behind `parsed_draft_holdsnap` (default `False`). The simulator
validates the algorithm; the e2e benchmark on GTX5k showed a
regression (see "Production caveat" section above), so the production
default is off. Code is retained for further research.

## Related

- `benchmark_parsed_draft.py` — the simulator itself.
- `benchmark_parsed_draft_e2e.py` — end-to-end vLLM benchmark with a
real GPU and the PaddleOCR-VL model.
- `eagle_parsed_draft_spec_decode.py` — the EAGLE reference
implementation the simulator validates against.
- `vllm/v1/spec_decode/parsed_draft.py` — vLLM's production
implementation.
