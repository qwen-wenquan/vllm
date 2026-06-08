# Parsed-Draft Benchmark — Installation Guide

Step-by-step setup for running the parsed-draft speculative decoding
benchmarks on a machine with an NVIDIA GPU (tested on H100 NVL, CUDA 12.9).

## Prerequisites

- NVIDIA GPU (Hopper or later recommended)
- Python 3.10+
- `uv` package manager
- Git LFS (for downloading model weights)

## 1. Environment Setup

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and activate
uv venv --python 3.12
source .venv/bin/activate

# Install lint tools and pre-commit hooks
uv pip install -r requirements/lint.txt
pre-commit install
```

## 2. Install vLLM (editable, precompiled)

Use precompiled binaries to skip C++/CUDA compilation — this is the
recommended path if you are only making Python changes:

```bash
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
```

If you are also modifying C/C++/CUDA code, compile from source instead
(requires `nvcc` and the CUDA toolkit):

```bash
uv pip install -e . --torch-backend=auto
```

## 3. Install Python Dev Headers

FlashInfer and other components JIT-compile small C extensions at runtime.
This requires `Python.h`:

```bash
# For Python 3.10
sudo apt-get install python3.10-dev

# Or generically
sudo apt-get install python3-dev
```

Without this you will see:

```text
fatal error: Python.h: No such file or directory
```

## 4. Install Benchmark Dependencies

```bash
uv pip install pybase64 pymupdf Pillow
```

- **pybase64** — fast base64 encoding for images
- **pymupdf** (`fitz`) — PDF rendering and page extraction
- **Pillow** — image processing

## 5. Environment Variables

These environment variables address common runtime errors when `nvcc` /
the full CUDA toolkit is not installed on the host:

```bash
# Disable FlashInfer JIT sampler (avoids "Could not find nvcc" error)
export VLLM_USE_FLASHINFER_SAMPLER=0

# Disable DeepGEMM warmup (avoids "DeepGEMM backend is not available"
# error). Safe to disable — PaddleOCR-VL does not use FP8 quantization.
export VLLM_USE_DEEP_GEMM=0
```

### Why are these needed?

| Variable | Default | What it controls |
| --- | --- | --- |
| `VLLM_USE_FLASHINFER_SAMPLER` | `1` (on) | FlashInfer's fused top-k/top-p GPU sampling kernels. Requires `nvcc` for JIT compilation. Setting to `0` falls back to PyTorch-based sampling — no quality difference for greedy (`temperature=0`). |
| `VLLM_USE_DEEP_GEMM` | `1` (on) | DeepGEMM FP8 GEMM kernels for Hopper/Blackwell. The warmup step crashes if the package is missing. Safe to disable for non-FP8 models. |

If you **do** have the CUDA toolkit installed, you can point to it instead:

```bash
export CUDA_HOME=/usr/local/cuda   # or wherever nvcc lives
export PATH="$CUDA_HOME/bin:$PATH"
# Then you can leave VLLM_USE_FLASHINFER_SAMPLER=1 (default)
```

## 6. Model Setup

### Download PaddleOCR-VL

```bash
# Make sure Git LFS is installed
sudo apt-get install git-lfs
git lfs install

# Clone the model (adjust path as needed)
git clone https://huggingface.co/PaddlePaddle/PaddleOCR-VL \
    /home/dlisuser/EAGLE/models/PaddleOCR-VL_finetune
```

**Important**: Verify that large files were actually downloaded by LFS
(not left as pointer files):

```bash
# Should be ~11 MB, NOT 133 bytes
wc -c /home/dlisuser/EAGLE/models/PaddleOCR-VL_finetune/tokenizer.json

# If it shows ~133 bytes, the file is a Git LFS pointer. Fix with:
cd /home/dlisuser/EAGLE/models/PaddleOCR-VL_finetune
git lfs pull
```

A 133-byte `tokenizer.json` causes:

```text
json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```

### Sample Data (GTX5k)

The E2E benchmark expects document samples in GTX5k format:

```text
samples/GTX5k/
  <doc_id>/
    document.pdf
    det_ocr_results.json
```

## 7. Run the Benchmarks

### E2E Benchmark (GPU required)

```bash
# Full command with all env vars
VLLM_USE_FLASHINFER_SAMPLER=0 \
VLLM_USE_DEEP_GEMM=0 \
.venv/bin/python benchmarks/spec_decode/benchmark_parsed_draft_e2e.py \
    --model /home/dlisuser/EAGLE/models/PaddleOCR-VL_finetune \
    --samples-dir /home/dlisuser/EAGLE/models/samples/GTX5k \
    --max-docs 1 --max-blocks 50
```

### Fidelity Benchmark (no GPU required)

```bash
.venv/bin/python benchmarks/spec_decode/benchmark_parsed_draft.py
```

## Troubleshooting

| Error | Cause | Fix |
| --- | --- | --- |
| `JSONDecodeError: Expecting value: line 1 column 1` | `tokenizer.json` is a Git LFS pointer (133 bytes) | Run `git lfs pull` in the model directory |
| `fatal error: Python.h: No such file or directory` | Missing Python development headers | `sudo apt-get install python3-dev` |
| `Could not find nvcc` | FlashInfer JIT needs CUDA toolkit | `export VLLM_USE_FLASHINFER_SAMPLER=0` |
| `DeepGEMM backend is not available` | `deep_gemm` package not installed | `export VLLM_USE_DEEP_GEMM=0` |
| `Unrecognized keys in rope_parameters: {'mrope_section'}` | Transformers doesn't recognize M-RoPE config | Benign warning, no action needed |
