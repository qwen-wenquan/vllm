# Agent Instructions for `benchmarks/`

> Applies to every script under `benchmarks/`. The project-wide
> [`AGENTS.md`](../AGENTS.md) still applies (env setup, duplicate-work checks,
> commit conventions, no-busywork rule).

## What lives where

| Path | Contents | Add new file here when… |
| ---- | -------- | ----------------------- |
| `benchmarks/benchmark_*.py` | Single-file scripts that exercise one vLLM subsystem (proposer, scheduler, sampler, etc.). | The bench is self-contained in one file and shares no extra assets. |
| `benchmarks/<area>/` | Multi-file suites with their own `README.md` (e.g. `attention_benchmarks/`, `multi_turn/`, `auto_tune/`). | You need ≥ 2 scripts, configs, fixtures, or area-specific setup docs. |
| `benchmarks/kernels/` | Micro-benches for CUDA/Triton kernels. Files here may use the `bench_*.py` prefix. | The bench targets a kernel (matmul, attention op, reshape, …) and uses `triton.testing.do_bench*`. |
| `benchmarks/cutlass_benchmarks/`, `benchmarks/fused_kernels/` | Kernel benches for those specific stacks. | The bench is specific to CUTLASS or a fused-kernel implementation. |

**Do not modify** `benchmark_latency.py`, `benchmark_serving.py`,
`benchmark_throughput.py`. They are deprecation stubs pointing at
`vllm bench …`. Edit the CLI in `vllm/entrypoints/cli/` instead.

## Naming

- Default to **`benchmark_<feature>.py`** for top-level scripts. This is the
  dominant convention and what `vllm bench`-adjacent tooling expects.
- `bench_<feature>.py` is acceptable only inside `kernels/` and `multi_turn/`,
  matching the local convention.
- For multi-file suites, name the entrypoint `benchmark.py` or
  `benchmark_<feature>.py`; helpers may use any name.

## File skeleton

Every script starts with these two header lines (and an optional shebang above
them if you mark it executable):

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
```

Use this entrypoint pattern (mirrors `benchmark_ngram_proposer.py`,
`benchmark_block_pool.py`):

```python
def invoke_main() -> None:
    parser = FlexibleArgumentParser(description="…")
    ...
    args = parser.parse_args()
    benchmark_fn(args)


if __name__ == "__main__":
    invoke_main()  # pragma: no cover
```

## Argparse choice

- **`FlexibleArgumentParser`** (from `vllm.utils.argparse_utils`): use whenever
  the script constructs a `VllmConfig`, calls `LLM(...)`, or otherwise wants
  users to pass full vLLM CLI flags. This is the default.
- Plain `argparse.ArgumentParser`: only for pure micro-benches that never load
  a vLLM engine (e.g. `benchmark_hash.py`, `benchmark_topk_topp.py`).

## Shared utilities

- `benchmarks/benchmark_utils.py` exports **only** `TimeCollector`. Import it
  as a bare top-level module (no package path):

  ```python
  from benchmark_utils import TimeCollector  # NOT: from benchmarks.benchmark_utils
  ```

  Scripts are run with `benchmarks/` on `sys.path` (the canonical invocation is
  `python benchmarks/benchmark_foo.py` from the repo root, which adds the
  script's directory to `sys.path`).

- For tabular output of micro-benches, use `tabulate`:

  ```python
  print(tabulate(rows, headers=[...], tablefmt="grid", floatfmt=".3f"))
  ```

- For CUDA kernel timing, prefer `triton.testing.do_bench_cudagraph` over
  hand-rolled `torch.cuda.Event` loops (see `kernels/bench_concat_mla_q.py`).

## Data and fixtures

- Small text/JSON fixtures used by exactly one bench are fine to check in
  (see `sonnet.txt`, `structured_schemas/`).
- Anything > 1 MB or model-specific (PDFs, OCR dumps, model weights) must be
  downloaded by the script or documented in the suite's `README.md` — never
  committed.
- Do **not** hardcode absolute paths like `/home/<user>/...` or
  sibling-repo paths like `../EAGLE/`. Use CLI flags with sensible defaults
  that point to public/HF datasets, and gate optional external comparisons
  behind a flag plus a `try/except ImportError`.

## Speculative decoding benches

Before adding a new spec-decode bench, read
`vllm/v1/spec_decode/CLAUDE.md` for the proposer taxonomy (model-based vs.
non-model). Use `benchmark_ngram_proposer.py` as the template for non-model
proposers — same flat layout, `TimeCollector` + `tabulate`, two-mode CLI
(`--batched` for the runner-integrated path).

End-to-end spec-decode benches that load a real model belong in their own
subdirectory under `benchmarks/spec_decode/` with a `README.md` that
documents required models, datasets, and expected runtime.

## Before opening a PR

- Run `pre-commit run --files benchmarks/<your_file>.py`.
- Execute the bench at least once and paste the headline numbers into the
  PR description.
- For spec-decode work, also run the relevant correctness tests under
  `tests/v1/spec_decode/` or `tests/v1/e2e/spec_decode/` — perf gains are
  meaningless without an output-fidelity check elsewhere in the tree.
