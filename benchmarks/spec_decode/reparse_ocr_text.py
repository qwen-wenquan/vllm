#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Re-extract parsed_text from document.pdf using PyMuPDF.

Reads det_ocr_results.json for block bounding boxes, then uses
PyMuPDF's get_text() with TEXT_DEHYPHENATE to extract clean text
from the corresponding PDF region. Line breaks within a block are
joined with spaces (since they are visual line-wraps, not semantic
paragraph breaks).

This fixes several issues with the original mupdf extraction:
  - Newlines from visual line-wrapping (e.g. column layout)
  - Hyphenated line breaks (pho-\\ntonic -> photonic)
  - Leading garbage characters from cross-line bleeding

Usage:
    # Re-parse a single document
    .venv/bin/python benchmarks/spec_decode/reparse_ocr_text.py \\
        --doc-dir /path/to/samples/GTX5k/<doc_id>_document

    # Re-parse all documents under a samples directory
    .venv/bin/python benchmarks/spec_decode/reparse_ocr_text.py \\
        --samples-dir /path/to/samples/GTX5k

    # Dry run (print changes without writing)
    .venv/bin/python benchmarks/spec_decode/reparse_ocr_text.py \\
        --samples-dir /path/to/samples/GTX5k --dry-run

    # Write to a different output file
    .venv/bin/python benchmarks/spec_decode/reparse_ocr_text.py \\
        --samples-dir /path/to/samples/GTX5k \\
        --output-name det_ocr_results_reparsed.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import regex as re


def build_page_mapping(det_data: dict) -> dict[int, int]:
    """Build original-to-packed page index mapping."""
    mapping: dict[int, int] = {}
    page_spans = det_data.get("page_spans", [])
    packed_spans = det_data.get("packed_page_spans", [])
    if page_spans and packed_spans:
        for orig, packed in zip(page_spans, packed_spans):
            for o, p in zip(
                range(orig[0], orig[1] + 1),
                range(packed[0], packed[1] + 1),
            ):
                mapping[o] = p
    return mapping


def extract_text_from_rect(page, rect, dehyphenate: bool = True) -> str:
    """Extract clean text from a PDF page region.

    Uses PyMuPDF's get_text with TEXT_DEHYPHENATE to handle
    hyphenated line breaks, then joins lines with spaces.
    """
    import fitz

    flags = fitz.TEXT_PRESERVE_WHITESPACE
    if dehyphenate:
        flags |= fitz.TEXT_DEHYPHENATE

    raw = page.get_text("text", clip=rect, flags=flags)

    # Split into lines, strip each, filter empties
    lines = [line.strip() for line in raw.split("\n") if line.strip()]

    # Join lines with space — these are visual line-wraps
    text = " ".join(lines)

    # Fix residual hyphenation that TEXT_DEHYPHENATE missed
    # Pattern: "word- continuation" -> "word-continuation"
    # Only when the hyphen is at what was a line boundary
    text = re.sub(r"(\w)- (\w)", r"\1-\2", text)

    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)

    return text.strip()


def reparse_document(
    doc_dir: Path,
    dpi: int = 200,
    output_name: str | None = None,
    dry_run: bool = False,
) -> dict:
    """Re-extract parsed_text for all blocks in a document.

    Returns stats dict with counts of changes.
    """
    import fitz

    det_path = doc_dir / "det_ocr_results.json"
    pdf_path = doc_dir / "document.pdf"

    if not det_path.exists():
        return {"error": f"No det_ocr_results.json in {doc_dir}"}
    if not pdf_path.exists():
        return {"error": f"No document.pdf in {doc_dir}"}

    with open(det_path) as f:
        det_data = json.load(f)

    doc = fitz.open(str(pdf_path))
    page_mapping = build_page_mapping(det_data)

    stats = {
        "total_blocks": 0,
        "reparsed": 0,
        "unchanged": 0,
        "empty_region": 0,
        "skipped_no_gt": 0,
        "skipped_category": 0,
        "newlines_removed": 0,
        "hyphens_fixed": 0,
    }

    skip_categories = {7}  # Picture

    for page_key, page_data in det_data["pages"].items():
        for blk in page_data["blocks"]:
            stats["total_blocks"] += 1
            cat_id = blk.get("category_id", 0)

            if cat_id in skip_categories:
                stats["skipped_category"] += 1
                continue

            gt_text = blk.get("gt_text", "").strip()
            if not gt_text:
                stats["skipped_no_gt"] += 1
                continue

            # Resolve the PDF page
            pdf_page_orig = blk.get("pdf_page", int(page_key))
            pdf_page = page_mapping.get(pdf_page_orig, pdf_page_orig)
            if pdf_page >= len(doc):
                continue

            page = doc[pdf_page]
            pw, ph = page.rect.width, page.rect.height

            # Convert normalized bbox [x, y, w, h] to fitz.Rect
            x, y, w, h = blk["bbox"]
            rect = fitz.Rect(x * pw, y * ph, (x + w) * pw, (y + h) * ph)

            new_text = extract_text_from_rect(page, rect)

            old_text = blk.get("parsed_text", "")
            old_stripped = old_text.replace("\n", " ").strip()

            if not new_text:
                stats["empty_region"] += 1
                continue

            if new_text == old_stripped:
                stats["unchanged"] += 1
            else:
                stats["reparsed"] += 1

            # Count specific fixes
            if "\n" in old_text:
                stats["newlines_removed"] += 1
            if "-\n" in old_text:
                stats["hyphens_fixed"] += 1

            blk["parsed_text"] = new_text

    doc.close()

    # Write output
    if not dry_run:
        out_name = output_name or "det_ocr_results.json"
        out_path = doc_dir / out_name
        with open(out_path, "w") as f:
            json.dump(det_data, f, indent=2, ensure_ascii=False)
        stats["output_path"] = str(out_path)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description=("Re-extract parsed_text from PDF using PyMuPDF"),
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--doc-dir",
        type=str,
        help="Single document directory to process",
    )
    group.add_argument(
        "--samples-dir",
        type=str,
        help=(
            "Root samples directory — processes all "
            "subdirectories with det_ocr_results.json"
        ),
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help=("Output filename (default: overwrite det_ocr_results.json in-place)"),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print stats without writing files",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Max documents to process",
    )
    args = parser.parse_args()

    if args.doc_dir:
        doc_dirs = [Path(args.doc_dir)]
    else:
        samples_path = Path(args.samples_dir)
        doc_dirs = sorted(
            d
            for d in samples_path.iterdir()
            if d.is_dir() and (d / "det_ocr_results.json").exists()
        )
        if args.max_docs:
            doc_dirs = doc_dirs[: args.max_docs]

    print(f"Processing {len(doc_dirs)} document(s)...")
    if args.dry_run:
        print("(dry run — no files will be written)")
    print()

    totals = {
        "total_blocks": 0,
        "reparsed": 0,
        "unchanged": 0,
        "empty_region": 0,
        "skipped_no_gt": 0,
        "skipped_category": 0,
        "newlines_removed": 0,
        "hyphens_fixed": 0,
    }

    for doc_dir in doc_dirs:
        stats = reparse_document(
            doc_dir,
            output_name=args.output_name,
            dry_run=args.dry_run,
        )
        if "error" in stats:
            print(f"  SKIP {doc_dir.name}: {stats['error']}")
            continue

        for k in totals:
            totals[k] += stats.get(k, 0)

        n_fixed = stats["newlines_removed"]
        n_total = stats["reparsed"] + stats["unchanged"]
        print(f"  {doc_dir.name}: {n_total} blocks, {n_fixed} had newlines")

    print(f"\n{'=' * 50}")
    print("SUMMARY")
    print(f"{'=' * 50}")
    print(f"  Total blocks:       {totals['total_blocks']}")
    print(f"  Reparsed:           {totals['reparsed']}")
    print(f"  Unchanged:          {totals['unchanged']}")
    print(f"  Empty region:       {totals['empty_region']}")
    print(f"  Skipped (no gt):    {totals['skipped_no_gt']}")
    print(f"  Skipped (picture):  {totals['skipped_category']}")
    print(f"  Newlines removed:   {totals['newlines_removed']}")
    print(f"  Hyphens fixed:      {totals['hyphens_fixed']}")

    if args.dry_run:
        print("\n  (dry run — no files written)")
    else:
        out = args.output_name or "det_ocr_results.json"
        print(f"\n  Output written to: {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
