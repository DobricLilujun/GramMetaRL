from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz

from .example_agent import ExampleAgent
from .llm_extract import ExtractionChunk, MBGExtractor
from .ocr import OCRProvider
from .pdf_ingest import PDFIngestor
from .schema import MBGCard
from .workspace import get_workspace_paths


@dataclass(slots=True)
class OCRSection:
    section_id: str
    title: str
    page_start: int
    page_end: int
    text: str


def _stable_id(seed: str) -> str:
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]


def _chunk_text(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    parts: list[str] = []
    current: list[str] = []
    cur_len = 0
    for para in text.split("\n\n"):
        p = para.strip()
        if not p:
            continue
        if cur_len + len(p) + 2 > max_chars and current:
            parts.append("\n\n".join(current))
            current = [p]
            cur_len = len(p)
        else:
            current.append(p)
            cur_len += len(p) + 2
    if current:
        parts.append("\n\n".join(current))
    return parts


def _block_to_tagged_text(block: dict[str, Any]) -> str:
    label = str(block.get("label") or block.get("block_type") or "unknown")
    bbox = block.get("bbox", [])
    text = str(block.get("text", "")).strip()
    if not text:
        return ""
    return f"[OCR_BLOCK type={label} bbox={bbox}] {text}"


def _page_record_to_text(page_row: dict[str, Any]) -> str:
    blocks = page_row.get("blocks", [])
    if isinstance(blocks, list) and blocks:
        lines = [_block_to_tagged_text(b) for b in blocks if isinstance(b, dict)]
        return "\n".join(line for line in lines if line)

    # Fallback for records without blocks payload.
    markdown_text = str(page_row.get("result_markdown", "")).strip()
    if markdown_text:
        return markdown_text
    return str(page_row.get("ocr_text", "")).strip()


def _load_precomputed_ocr_page_rows(path: Path) -> dict[int, dict[str, Any]]:
    page_rows: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            page_number = row.get("page_number")
            if not isinstance(page_number, int):
                continue
            # Skip error rows written by batch OCR.
            if row.get("error"):
                continue
            page_rows[page_number] = row
    return page_rows


def _build_sections_from_ocr_rows(page_rows: dict[int, dict[str, Any]], max_page: int) -> list[OCRSection]:
    sections: list[OCRSection] = []
    sec_idx = 1
    current_title = "Untitled"
    current_lines: list[str] = []
    current_start: int | None = None
    current_end: int | None = None

    def flush() -> None:
        nonlocal sec_idx, current_lines, current_start, current_end
        if not current_lines or current_start is None or current_end is None:
            return
        sections.append(
            OCRSection(
                section_id=f"ocr_sec_{sec_idx:04d}",
                title=current_title,
                page_start=current_start,
                page_end=current_end,
                text="\n".join(current_lines).strip(),
            )
        )
        sec_idx += 1
        current_lines = []
        current_start = None
        current_end = None

    for page in sorted(page_rows.keys()):
        if page < 1 or page > max_page:
            continue
        row = page_rows[page]
        raw_blocks = row.get("blocks", [])
        if isinstance(raw_blocks, list) and raw_blocks:
            blocks = [b for b in raw_blocks if isinstance(b, dict)]
        else:
            fallback_text = _page_record_to_text(row)
            blocks = [{"label": "text", "bbox": [], "text": fallback_text}] if fallback_text else []

        for block in blocks:
            label = str(block.get("label") or block.get("block_type") or "unknown").strip().lower()
            text = str(block.get("text", "")).strip()
            if not text:
                continue

            if label == "sub_title":
                flush()
                current_title = text.replace("\n", " ").strip() or "Untitled"
                continue

            line = _block_to_tagged_text(block)
            if not line:
                continue
            if current_start is None:
                current_start = page
            current_end = page
            current_lines.append(line)

    flush()
    return sections


def _build_page_batch_sections_from_ocr_rows(page_rows: dict[int, dict[str, Any]], max_page: int, batch_pages: int) -> list[OCRSection]:
    sections: list[OCRSection] = []
    if batch_pages < 1:
        raise ValueError("batch_pages must be >= 1")

    start_page = min(page_rows.keys()) if page_rows else 1
    end_page = max(page_rows.keys()) if page_rows else 0
    sec_idx = 1

    for batch_start in range(start_page, end_page + 1, batch_pages):
        batch_end = min(batch_start + batch_pages - 1, max_page)
        lines: list[str] = []
        for page in range(batch_start, batch_end + 1):
            row = page_rows.get(page)
            if not row:
                continue
            raw_blocks = row.get("blocks", [])
            blocks = [b for b in raw_blocks if isinstance(b, dict)] if isinstance(raw_blocks, list) else []
            if not blocks:
                fallback_text = _page_record_to_text(row)
                if fallback_text:
                    lines.append(f"[OCR_PAGE page={page}] {fallback_text}")
                continue
            for block in blocks:
                line = _block_to_tagged_text(block)
                if line:
                    lines.append(line)

        if not lines:
            continue

        sections.append(
            OCRSection(
                section_id=f"ocr_batch_{sec_idx:04d}",
                title=f"Pages {batch_start}-{batch_end}",
                page_start=batch_start,
                page_end=batch_end,
                text="\n".join(lines).strip(),
            )
        )
        sec_idx += 1

    return sections


def _build_block_sections_from_ocr_rows(page_rows: dict[int, dict[str, Any]], max_page: int) -> list[OCRSection]:
    sections: list[OCRSection] = []
    sec_idx = 1
    current_title = "Untitled"

    for page in sorted(page_rows.keys()):
        if page < 1 or page > max_page:
            continue
        row = page_rows[page]
        raw_blocks = row.get("blocks", [])
        if isinstance(raw_blocks, list) and raw_blocks:
            blocks = [b for b in raw_blocks if isinstance(b, dict)]
        else:
            fallback_text = _page_record_to_text(row)
            blocks = [{"label": "text", "bbox": [], "text": fallback_text}] if fallback_text else []

        for block_idx, block in enumerate(blocks, start=1):
            label = str(block.get("label") or block.get("block_type") or "unknown").strip().lower()
            text = str(block.get("text", "")).strip()
            if not text:
                continue

            if label == "sub_title":
                current_title = text.replace("\n", " ").strip() or "Untitled"
                continue

            line = _block_to_tagged_text(block)
            if not line:
                continue

            sections.append(
                OCRSection(
                    section_id=f"ocr_blk_{sec_idx:04d}",
                    title=current_title,
                    page_start=page,
                    page_end=page,
                    text=line,
                )
            )
            sec_idx += 1

    return sections


def _build_extraction_sections_from_ocr_rows(
    page_rows: dict[int, dict[str, Any]],
    max_page: int,
    extract_mode: str,
    batch_pages: int,
) -> list[OCRSection]:
    mode = (extract_mode or "chapter").strip().lower()
    if mode == "chapter":
        return _build_sections_from_ocr_rows(page_rows, max_page=max_page)
    if mode == "page_batch":
        return _build_page_batch_sections_from_ocr_rows(page_rows, max_page=max_page, batch_pages=batch_pages)
    if mode == "block":
        return _build_block_sections_from_ocr_rows(page_rows, max_page=max_page)
    raise ValueError(f"Unsupported extract_mode: {extract_mode}")


def build_mbg_from_pdf(
    pdf_path: Path | None,
    output_jsonl: Path,
    language: str,
    source_id: str,
    extractor: MBGExtractor,
    ocr_provider: OCRProvider | None,
    example_agent: ExampleAgent | None = None,
    example_output_jsonl: Path | None = None,
    extract_mode: str = "chapter",
    batch_pages: int = 3,
    max_chunk_chars: int = 4200,
    render_dpi: int = 220,
    work_dir: Path | None = None,
    ocr_pages_jsonl: Path | None = None,
    page_start: int | None = None,
    page_end: int | None = None,
) -> list[MBGCard]:
    def _append_card(handle, card: MBGCard) -> None:
        handle.write(json.dumps(card.as_dict(), ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())

    def _append_example(handle, record) -> None:
        handle.write(json.dumps(record.as_dict(), ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())

    use_precomputed_ocr = ocr_pages_jsonl is not None

    if use_precomputed_ocr:
        if not ocr_pages_jsonl.exists():
            raise FileNotFoundError(f"Precomputed OCR JSONL not found: {ocr_pages_jsonl}")
        page_rows = _load_precomputed_ocr_page_rows(ocr_pages_jsonl)

        if not page_rows:
            raise ValueError("No usable rows found in precomputed OCR JSONL")

        if pdf_path is not None:
            with fitz.open(pdf_path) as doc:
                max_page = doc.page_count
        else:
            max_page = max(page_rows.keys())

        start = page_start if page_start is not None else 1
        end = page_end if page_end is not None else max_page
        if start < 1:
            raise ValueError("page_start must be >= 1")
        if end < start:
            raise ValueError("page_end must be >= page_start")

        page_rows = {p: row for p, row in page_rows.items() if start <= p <= end}
        if not page_rows:
            raise ValueError("No OCR pages left after applying page range filter")

        ocr_sections = _build_extraction_sections_from_ocr_rows(
            page_rows,
            max_page=max_page,
            extract_mode=extract_mode,
            batch_pages=batch_pages,
        )
        if not ocr_sections:
            raise ValueError("No usable OCR sections were built from ocr_pages_jsonl")

        cards: list[MBGCard] = []
        seen_ids: set[str] = set()
        total = len(ocr_sections)
        example_handle = None
        if example_agent and example_output_jsonl is not None:
            example_output_jsonl.parent.mkdir(parents=True, exist_ok=True)
            example_handle = example_output_jsonl.open("w", encoding="utf-8")

        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        try:
            with output_jsonl.open("w", encoding="utf-8") as f:
                for idx, sec in enumerate(ocr_sections, start=1):
                    print(
                        f"[MBG] section {idx}/{total} title={sec.title} pages={sec.page_start}-{sec.page_end}",
                        flush=True,
                    )

                    chunk = ExtractionChunk(
                        language=language,
                        source_id=source_id,
                        section_id=sec.section_id,
                        section_title=sec.title,
                        page_start=sec.page_start,
                        page_end=sec.page_end,
                        text=sec.text,
                    )

                    try:
                        extracted = extractor.extract_cards(chunk)
                    except Exception as exc:  # noqa: BLE001
                        print(f"[MBG][WARN] section {idx}/{total} failed: {exc}", flush=True)
                        continue
                    added = 0
                    for card in extracted:
                        if not card.id or card.id in seen_ids:
                            seed = (
                                f"{language}|{card.scope}|{','.join(card.trigger_conditions)}|"
                                f"{sec.section_id}|{card.source.page_start}-{card.source.page_end}"
                            )
                            card.id = f"{language.upper()}_{_stable_id(seed)}"
                        if card.id in seen_ids:
                            continue
                        seen_ids.add(card.id)
                        cards.append(card)
                        _append_card(f, card)
                        if example_agent and example_handle is not None:
                            try:
                                example_record = example_agent.generate_record(card, source_text=chunk.text)
                            except Exception as exc:  # noqa: BLE001
                                print(f"[MBG][WARN] example agent failed for {card.id}: {exc}", flush=True)
                            else:
                                if example_record is not None:
                                    _append_example(example_handle, example_record)
                        added += 1

                    print(f"[MBG] section {idx}/{total} extracted={len(extracted)} added={added}", flush=True)
        finally:
            if example_handle is not None:
                example_handle.close()

        print(f"[MBG] completed sections={total} cards={len(cards)} output={output_jsonl}", flush=True)
        return cards

    else:
        if pdf_path is None:
            raise ValueError("pdf_path is required when using live OCR mode")
        work = work_dir or get_workspace_paths().default_mbg_work_dir
        images_dir = work / "images"
        ingestor = PDFIngestor(render_dpi=render_dpi)
        sections, page_images = ingestor.parse(
            pdf_path=pdf_path,
            image_dir=images_dir,
            render_images=True,
        )

        if ocr_provider is None:
            raise ValueError("ocr_provider is required when ocr_pages_jsonl is not provided")

        ocr_cache: dict[int, str] = {}
        for page_no, image_path in page_images.items():
            ocr_text = ocr_provider.extract_page_text(image_path=image_path, page_number=page_no)
            ocr_cache[page_no] = ocr_text

    cards: list[MBGCard] = []
    seen_ids: set[str] = set()

    total_sections = len(sections)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    example_handle = None
    if example_agent and example_output_jsonl is not None:
        example_output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        example_handle = example_output_jsonl.open("w", encoding="utf-8")
    try:
        with output_jsonl.open("w", encoding="utf-8") as f:
            for sec_idx, sec in enumerate(sections, start=1):
                print(
                    f"[MBG] section {sec_idx}/{total_sections} title={sec.title} pages={sec.page_start}-{sec.page_end}",
                    flush=True,
                )
                page_ocr = "\n".join(ocr_cache.get(p, "") for p in range(sec.page_start, sec.page_end + 1))
                fused_text = (sec.text + "\n\n" + page_ocr).strip()
                if not fused_text:
                    continue

                subchunks = _chunk_text(fused_text, max_chars=max_chunk_chars)
                for i, text_chunk in enumerate(subchunks, start=1):
                    chunk = ExtractionChunk(
                        language=language,
                        source_id=source_id,
                        section_id=f"{sec.section_id}_c{i}",
                        section_title=sec.title,
                        page_start=sec.page_start,
                        page_end=sec.page_end,
                        text=text_chunk,
                    )
                    try:
                        extracted = extractor.extract_cards(chunk)
                    except Exception as exc:  # noqa: BLE001
                        print(
                            f"[MBG][WARN] section {sec_idx}/{total_sections} chunk={i}/{len(subchunks)} failed: {exc}",
                            flush=True,
                        )
                        continue
                    added = 0
                    for card in extracted:
                        if not card.id or card.id in seen_ids:
                            seed = (
                                f"{language}|{card.scope}|{','.join(card.trigger_conditions)}|"
                                f"{sec.section_id}|{card.source.page_start}-{card.source.page_end}"
                            )
                            card.id = f"{language.upper()}_{_stable_id(seed)}"
                        if card.id in seen_ids:
                            continue
                        seen_ids.add(card.id)
                        cards.append(card)
                        _append_card(f, card)
                        if example_agent and example_handle is not None:
                            try:
                                example_record = example_agent.generate_record(card, source_text=chunk.text)
                            except Exception as exc:  # noqa: BLE001
                                print(f"[MBG][WARN] example agent failed for {card.id}: {exc}", flush=True)
                            else:
                                if example_record is not None:
                                    _append_example(example_handle, example_record)
                        added += 1

                    print(
                        f"[MBG] section {sec_idx}/{total_sections} chunk={i}/{len(subchunks)} "
                        f"extracted={len(extracted)} added={added}",
                        flush=True,
                    )
    finally:
        if example_handle is not None:
            example_handle.close()

    print(f"[MBG] completed sections={total_sections} cards={len(cards)} output={output_jsonl}", flush=True)

    return cards


def load_cards(path: Path) -> list[MBGCard]:
    cards: list[MBGCard] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cards.append(MBGCard.model_validate_json(line))
    return cards
