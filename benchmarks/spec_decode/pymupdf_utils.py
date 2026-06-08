#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
PyMuPDF text extraction utilities for parsed-draft speculative decoding.

Provides a robust ``extract_block_text()`` function that extracts
clean text from a PDF region, suitable for use as a speculative
draft. The output is designed to maximize token-level overlap with
what a VL model produces from the same image crop.

Guidelines for PyMuPDF text extraction
======================================

1. **Join visual line breaks with spaces.**
   PDF text extraction returns ``\\n`` at every visual line boundary,
   even within a single paragraph. These are layout artifacts, not
   semantic breaks. Always join lines with a space.

2. **Enable ``TEXT_DEHYPHENATE``.**
   Academic PDFs frequently hyphenate words at line breaks
   (``pho-\\ntonic`` → ``photonic``). PyMuPDF's ``TEXT_DEHYPHENATE``
   flag (``0x10``) merges these automatically.

3. **Handle residual hyphens.**
   ``TEXT_DEHYPHENATE`` doesn't catch compound-word hyphens at line
   boundaries (``MALE-\\nIDENTIFIED``). After joining lines, fix
   patterns like ``word- next`` → ``word-next``.

4. **Preserve ``TEXT_PRESERVE_WHITESPACE``.**
   Without this flag (``0x2``), PyMuPDF collapses internal
   whitespace, which can merge tokens that should be separate.

5. **Collapse double spaces.**
   After joining lines and fixing hyphens, collapse runs of
   multiple spaces to a single space.

6. **Use ``clip=rect`` for block-level extraction.**
   Extract text only from the bounding box region. Full-page
   extraction mixes text from adjacent blocks.

7. **Use ``get_text("text", ...)`` not ``get_textbox()``.**
   ``get_textbox()`` returns the same content but doesn't support
   the ``flags`` parameter, so ``TEXT_DEHYPHENATE`` won't work.

8. **Tables (category_id=9): preserve structure.**
   For table blocks, newlines between rows are semantically
   meaningful. Use a tab separator between cells instead of
   collapsing everything to a single line.

9. **Formulas (category_id=3): extract as-is.**
   Formula text from PDF is plain-text (``Z = b1 X1 + ...``)
   while ground truth is LaTeX. The VL model generates LaTeX,
   so the plain-text draft still has partial token overlap via
   variable names and operators.

10. **Normalize unicode.**
    Apply NFC normalization to handle composed vs decomposed
    unicode differences (e.g. accented characters).

Known limitations
=================

- **Column ordering**: PyMuPDF extracts text in reading order
  within a block bbox, but if the bbox spans two columns, text
  from both columns gets interleaved. This is a layout detection
  issue (upstream), not a text extraction issue.

- **Superscript/subscript numbers**: PyMuPDF extracts superscript
  numbers as regular text (``photonics1,2``). The VL model may
  produce them with special formatting. LCS handles the overlap.

- **Ligatures**: With ``TEXT_PRESERVE_LIGATURES``, PyMuPDF
  keeps ligature characters (``ﬀ``, ``ﬁ``, ``ﬂ``). Without it,
  they get decomposed. The VL model may produce either form.
  We decompose by default for better token matching.
"""

from __future__ import annotations

import unicodedata

import regex as re


def extract_block_text(
    page,
    bbox_norm: list[float],
    category_id: int = 0,
    dehyphenate: bool = True,
) -> str:
    """Extract clean text from a PDF page region.

    Args:
        page: A PyMuPDF ``fitz.Page`` object.
        bbox_norm: Normalized bounding box ``[x, y, w, h]``
            where each value is a fraction of page dimensions.
        category_id: Block category ID from layout detection.
            9 = Table (preserves row structure),
            3 = Formula (minimal processing).
        dehyphenate: If True, enable ``TEXT_DEHYPHENATE`` to
            merge hyphenated words across line breaks.

    Returns:
        Extracted text with visual line breaks joined, hyphens
        fixed, and whitespace normalized. Returns empty string
        if no text is found in the region.

    Example::

        import fitz

        doc = fitz.open("document.pdf")
        page = doc[0]
        text = extract_block_text(
            page,
            bbox_norm=[0.08, 0.15, 0.81, 0.12],
            category_id=1,  # Paragraph
        )
    """
    import fitz

    pw, ph = page.rect.width, page.rect.height
    x, y, w, h = bbox_norm
    rect = fitz.Rect(x * pw, y * ph, (x + w) * pw, (y + h) * ph)

    # Build extraction flags
    flags = fitz.TEXT_PRESERVE_WHITESPACE
    if dehyphenate:
        flags |= fitz.TEXT_DEHYPHENATE
    # Decompose ligatures for better token matching
    # (don't set TEXT_PRESERVE_LIGATURES)

    raw = page.get_text("text", clip=rect, flags=flags)
    if not raw or not raw.strip():
        return ""

    if category_id == 9:
        # Table: preserve row-level structure with tabs
        return _clean_table_text(raw)

    return _clean_paragraph_text(raw)


def _clean_paragraph_text(raw: str) -> str:
    """Clean extracted text for paragraphs, titles, captions, etc.

    Joins visual line breaks with spaces, fixes residual hyphens,
    and normalizes whitespace and unicode.
    """
    # Split into lines, strip each, filter empties
    lines = [line.strip() for line in raw.split("\n") if line.strip()]
    text = " ".join(lines)

    # Fix residual hyphenation that TEXT_DEHYPHENATE missed.
    # Pattern: "word- continuation" → "word-continuation"
    # This catches compound words split at line boundaries
    # (e.g. "MALE-\nIDENTIFIED" → after join "MALE- IDENTIFIED")
    text = re.sub(r"(\w)- (\w)", r"\1-\2", text)

    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)

    # Unicode NFC normalization
    text = unicodedata.normalize("NFC", text)

    return text.strip()


def _clean_table_text(raw: str) -> str:
    """Clean extracted text for table blocks.

    Tables have semantically meaningful row breaks, so we
    preserve line structure but clean up within each line.
    """
    lines = []
    for line in raw.split("\n"):
        stripped = line.strip()
        if stripped:
            # Collapse internal whitespace runs to single space
            stripped = re.sub(r" {2,}", " ", stripped)
            lines.append(stripped)

    text = " ".join(lines)
    text = unicodedata.normalize("NFC", text)
    return text.strip()


def reparse_blocks(
    det_data: dict,
    pdf_path: str,
    skip_categories: set[int] | None = None,
) -> dict:
    """Re-extract parsed_text for all blocks in a document.

    Modifies ``det_data`` in-place, updating each block's
    ``parsed_text`` field with clean text extracted from the PDF.

    Args:
        det_data: Parsed ``det_ocr_results.json`` dict.
        pdf_path: Path to ``document.pdf``.
        skip_categories: Category IDs to skip (default: {7}
            for Picture blocks).

    Returns:
        Stats dict with counts of changes made.
    """
    import fitz

    if skip_categories is None:
        skip_categories = {7}

    doc = fitz.open(pdf_path)
    page_mapping = _build_page_mapping(det_data)

    stats = {
        "total": 0,
        "reparsed": 0,
        "unchanged": 0,
        "empty": 0,
        "skipped": 0,
    }

    for page_key, page_data in det_data["pages"].items():
        for blk in page_data["blocks"]:
            stats["total"] += 1
            cat_id = blk.get("category_id", 0)

            if cat_id in skip_categories:
                stats["skipped"] += 1
                continue

            gt_text = blk.get("gt_text", "").strip()
            if not gt_text:
                stats["skipped"] += 1
                continue

            pdf_page_orig = blk.get("pdf_page", int(page_key))
            pdf_page = page_mapping.get(pdf_page_orig, pdf_page_orig)
            if pdf_page >= len(doc):
                stats["skipped"] += 1
                continue

            page = doc[pdf_page]
            new_text = extract_block_text(page, blk["bbox"], category_id=cat_id)

            if not new_text:
                stats["empty"] += 1
                continue

            old_clean = _clean_paragraph_text(blk.get("parsed_text", ""))
            if new_text == old_clean:
                stats["unchanged"] += 1
            else:
                stats["reparsed"] += 1

            blk["parsed_text"] = new_text

    doc.close()
    return stats


def _build_page_mapping(det_data: dict) -> dict[int, int]:
    """Build original-to-packed page index mapping."""
    mapping: dict[int, int] = {}
    for orig, packed in zip(
        det_data.get("page_spans", []),
        det_data.get("packed_page_spans", []),
    ):
        for o, p in zip(
            range(orig[0], orig[1] + 1),
            range(packed[0], packed[1] + 1),
        ):
            mapping[o] = p
    return mapping
