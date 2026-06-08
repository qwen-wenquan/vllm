#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for PyMuPDF text extraction utilities.

Runs against real GTX5k sample data to verify extraction quality.
Falls back to synthetic tests when sample data is unavailable.

Usage:
    .venv/bin/python -m pytest benchmarks/spec_decode/test_pymupdf_utils.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.spec_decode.pymupdf_utils import (
    _build_page_mapping,
    _clean_paragraph_text,
    _clean_table_text,
    extract_block_text,
    reparse_blocks,
)

# Path to real GTX5k samples (skip if unavailable)
SAMPLES_DIR = Path("/home/dlisuser/EAGLE/models/samples/GTX5k")
HAS_SAMPLES = SAMPLES_DIR.exists() and any(
    (d / "det_ocr_results.json").exists() for d in SAMPLES_DIR.iterdir() if d.is_dir()
)


# ── Unit tests for _clean_paragraph_text ──────────────────────────


class TestCleanParagraphText:
    def test_join_newlines(self):
        raw = "first line\nsecond line\nthird line\n"
        assert _clean_paragraph_text(raw) == ("first line second line third line")

    def test_skip_empty_lines(self):
        raw = "line one\n\n\nline two\n"
        assert _clean_paragraph_text(raw) == ("line one line two")

    def test_fix_residual_hyphen(self):
        """After joining, 'MALE-\\nIDENTIFIED' becomes
        'MALE- IDENTIFIED', should fix to 'MALE-IDENTIFIED'."""
        raw = "MALE-\nIDENTIFIED"
        assert _clean_paragraph_text(raw) == "MALE-IDENTIFIED"

    def test_preserve_sentence_hyphen(self):
        """Hyphens between words in same line should stay."""
        raw = "state-of-the-art technology"
        assert _clean_paragraph_text(raw) == ("state-of-the-art technology")

    def test_collapse_double_spaces(self):
        raw = "word1  word2   word3"
        assert _clean_paragraph_text(raw) == ("word1 word2 word3")

    def test_strip_whitespace(self):
        raw = "  text with padding  \n"
        assert _clean_paragraph_text(raw) == ("text with padding")

    def test_unicode_normalization(self):
        """NFC normalization: decomposed → composed."""
        # e + combining acute = é
        raw = "café"
        result = _clean_paragraph_text(raw)
        assert "é" in result

    def test_empty_input(self):
        assert _clean_paragraph_text("") == ""
        assert _clean_paragraph_text("\n\n\n") == ""

    def test_multiline_academic_paragraph(self):
        """Simulate a typical academic PDF paragraph."""
        raw = (
            "Silicon photonics has developed into a\n"
            "mainstream technology driven by\n"
            "advances in optical communications.\n"
        )
        expected = (
            "Silicon photonics has developed into a "
            "mainstream technology driven by "
            "advances in optical communications."
        )
        assert _clean_paragraph_text(raw) == expected

    def test_hyphenated_word_across_lines(self):
        """Word hyphenated at line break:
        'pho-\\ntonic' → TEXT_DEHYPHENATE handles this,
        but if it leaks through: 'pho- tonic' → 'pho-tonic'."""
        raw = "pho-\ntonic integrated circuits"
        assert _clean_paragraph_text(raw) == ("pho-tonic integrated circuits")

    def test_leading_garbage(self):
        """Leading single-char lines from cross-line bleed."""
        raw = "g\np\nFigure 1 maps the evolution"
        result = _clean_paragraph_text(raw)
        assert result == "g p Figure 1 maps the evolution"


# ── Unit tests for _clean_table_text ──────────────────────────────


class TestCleanTableText:
    def test_basic_table(self):
        raw = "Name  Age\nAlice  30\nBob  25\n"
        result = _clean_table_text(raw)
        assert result == "Name Age Alice 30 Bob 25"

    def test_skip_empty_lines(self):
        raw = "Row1\n\nRow2\n"
        assert _clean_table_text(raw) == "Row1 Row2"

    def test_collapse_internal_spaces(self):
        raw = "Col1    Col2    Col3\n"
        assert _clean_table_text(raw) == "Col1 Col2 Col3"


# ── Unit tests for _build_page_mapping ────────────────────────────


class TestBuildPageMapping:
    def test_basic_mapping(self):
        det_data = {
            "page_spans": [[0, 3], [5, 6]],
            "packed_page_spans": [[0, 3], [4, 5]],
        }
        mapping = _build_page_mapping(det_data)
        assert mapping == {0: 0, 1: 1, 2: 2, 3: 3, 5: 4, 6: 5}

    def test_empty_mapping(self):
        det_data = {}
        mapping = _build_page_mapping(det_data)
        assert mapping == {}

    def test_single_page(self):
        det_data = {
            "page_spans": [[0, 0]],
            "packed_page_spans": [[0, 0]],
        }
        mapping = _build_page_mapping(det_data)
        assert mapping == {0: 0}


# ── Integration tests with real PDF ──────────────────────────────


@pytest.mark.skipif(
    not HAS_SAMPLES,
    reason="GTX5k samples not available",
)
class TestExtractBlockTextReal:
    """Tests using real GTX5k PDF + det_ocr_results.json."""

    @pytest.fixture(autouse=True)
    def setup(self):
        import fitz

        self.doc_dir = next(
            d
            for d in sorted(SAMPLES_DIR.iterdir())
            if d.is_dir()
            and (d / "det_ocr_results.json").exists()
            and (d / "document.pdf").exists()
        )
        with open(self.doc_dir / "det_ocr_results.json") as f:
            self.det_data = json.load(f)
        self.doc = fitz.open(str(self.doc_dir / "document.pdf"))
        self.page_mapping = _build_page_mapping(self.det_data)
        yield
        self.doc.close()

    def _get_page(self, blk, page_key):
        pdf_page = self.page_mapping.get(
            blk.get("pdf_page", int(page_key)),
            blk.get("pdf_page", int(page_key)),
        )
        if pdf_page < len(self.doc):
            return self.doc[pdf_page]
        return None

    def test_extraction_returns_string(self):
        """Basic: extract_block_text returns a non-empty str."""
        for pk, pd in self.det_data["pages"].items():
            for blk in pd["blocks"]:
                gt = blk.get("gt_text", "").strip()
                if not gt or blk.get("category_id") == 7:
                    continue
                page = self._get_page(blk, pk)
                if page is None:
                    continue
                text = extract_block_text(page, blk["bbox"], blk.get("category_id", 0))
                if text:
                    assert isinstance(text, str)
                    return  # one success is enough
        pytest.skip("No extractable blocks found")

    def test_no_newlines_in_paragraph(self):
        """Paragraphs should have no newlines after extraction."""
        for pk, pd in self.det_data["pages"].items():
            for blk in pd["blocks"]:
                if blk.get("category_id") != 1:
                    continue
                gt = blk.get("gt_text", "").strip()
                if not gt:
                    continue
                page = self._get_page(blk, pk)
                if page is None:
                    continue
                text = extract_block_text(page, blk["bbox"], category_id=1)
                if text:
                    assert "\n" not in text, f"Newline in paragraph: {repr(text[:80])}"
                    return
        pytest.skip("No paragraph blocks found")

    def test_no_double_spaces(self):
        """Output should not have double spaces."""
        for pk, pd in self.det_data["pages"].items():
            for blk in pd["blocks"]:
                gt = blk.get("gt_text", "").strip()
                if not gt or blk.get("category_id") == 7:
                    continue
                page = self._get_page(blk, pk)
                if page is None:
                    continue
                text = extract_block_text(page, blk["bbox"], blk.get("category_id", 0))
                if text:
                    assert "  " not in text, f"Double space in: {repr(text[:80])}"

    def test_lcs_acceptance_above_threshold(self):
        """Extracted text should have >= 80% LCS overlap
        with gt_text on average."""
        import sys

        sys.path.insert(0, ".")
        from transformers import AutoTokenizer

        from vllm.v1.spec_decode.parsed_draft import (
            IncrementalLCSMatcher,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            "/home/dlisuser/EAGLE/models/PaddleOCR-VL_finetune",
            trust_remote_code=True,
        )

        accepted_total = 0
        gt_total = 0
        n_blocks = 0

        for pk, pd in self.det_data["pages"].items():
            for blk in pd["blocks"]:
                gt = blk.get("gt_text", "").strip()
                if not gt or blk.get("category_id") in (7, 9, 3):
                    continue
                page = self._get_page(blk, pk)
                if page is None:
                    continue
                text = extract_block_text(
                    page,
                    blk["bbox"],
                    blk.get("category_id", 0),
                )
                if not text:
                    continue

                gt_ids = tokenizer.encode(gt, add_special_tokens=False)
                text_ids = tokenizer.encode(text, add_special_tokens=False)
                if not gt_ids or not text_ids:
                    continue

                matcher = IncrementalLCSMatcher(text_ids)
                acc, _ = matcher.add_chunk(gt_ids)
                n_acc = sum(e - s for s, e in acc)

                accepted_total += n_acc
                gt_total += len(gt_ids)
                n_blocks += 1

        if gt_total == 0:
            pytest.skip("No blocks to test")

        rate = accepted_total / gt_total
        assert rate >= 0.80, f"LCS acceptance {rate:.1%} < 80% ({n_blocks} blocks)"


@pytest.mark.skipif(
    not HAS_SAMPLES,
    reason="GTX5k samples not available",
)
class TestReparseBlocks:
    """Test the full reparse_blocks function."""

    def test_reparse_returns_stats(self):
        doc_dir = next(
            d
            for d in sorted(SAMPLES_DIR.iterdir())
            if d.is_dir()
            and (d / "det_ocr_results.json").exists()
            and (d / "document.pdf").exists()
        )
        with open(doc_dir / "det_ocr_results.json") as f:
            det_data = json.load(f)

        stats = reparse_blocks(det_data, str(doc_dir / "document.pdf"))
        assert "total" in stats
        assert "reparsed" in stats
        assert stats["total"] > 0

    def test_reparse_removes_newlines(self):
        """After reparse, no paragraph block should have
        newlines in parsed_text."""
        doc_dir = next(
            d
            for d in sorted(SAMPLES_DIR.iterdir())
            if d.is_dir()
            and (d / "det_ocr_results.json").exists()
            and (d / "document.pdf").exists()
        )
        with open(doc_dir / "det_ocr_results.json") as f:
            det_data = json.load(f)

        reparse_blocks(det_data, str(doc_dir / "document.pdf"))

        for pk, pd in det_data["pages"].items():
            for blk in pd["blocks"]:
                if blk.get("category_id") in (7, 9):
                    continue
                parsed = blk.get("parsed_text", "")
                if parsed:
                    assert "\n" not in parsed, (
                        f"Newline after reparse: {repr(parsed[:80])}"
                    )
