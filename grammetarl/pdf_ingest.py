from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path

import fitz


@dataclass(slots=True)
class TextBlock:
    block_id: str
    page: int
    text: str
    bbox: tuple[float, float, float, float]
    avg_font_size: float
    is_heading: bool


@dataclass(slots=True)
class Section:
    section_id: str
    title: str
    page_start: int
    page_end: int
    blocks: list[TextBlock]

    @property
    def text(self) -> str:
        return "\n".join(block.text for block in self.blocks if block.text.strip())


class PDFIngestor:
    def __init__(self, render_dpi: int = 220) -> None:
        self.render_dpi = render_dpi

    def parse(
        self,
        pdf_path: Path,
        image_dir: Path,
        render_images: bool = True,
    ) -> tuple[list[Section], dict[int, Path]]:
        if render_images:
            image_dir.mkdir(parents=True, exist_ok=True)
        doc = fitz.open(pdf_path)
        page_images: dict[int, Path] = {}
        all_blocks: list[TextBlock] = []

        for i, page in enumerate(doc):
            page_no = i + 1
            if render_images:
                img_path = image_dir / f"page_{page_no:04d}.png"
                if not img_path.exists():
                    zoom = self.render_dpi / 72.0
                    matrix = fitz.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=matrix, alpha=False)
                    pix.save(img_path)
                page_images[page_no] = img_path

            text_dict = page.get_text("dict")
            font_sizes = []
            for block in text_dict.get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        size = span.get("size", 0.0)
                        if size > 0:
                            font_sizes.append(size)
            base_font = statistics.median(font_sizes) if font_sizes else 10.0

            block_idx = 0
            for block in text_dict.get("blocks", []):
                lines = block.get("lines", [])
                if not lines:
                    continue

                chunk_lines = []
                block_font_sizes = []
                for line in lines:
                    line_text = "".join(span.get("text", "") for span in line.get("spans", []))
                    if line_text.strip():
                        chunk_lines.append(line_text)
                    for span in line.get("spans", []):
                        size = span.get("size", 0.0)
                        if size > 0:
                            block_font_sizes.append(size)

                block_text = "\n".join(chunk_lines).strip()
                if not block_text:
                    continue

                avg_font = statistics.mean(block_font_sizes) if block_font_sizes else base_font
                heading_like = avg_font >= base_font * 1.15 and len(block_text.split()) <= 14

                block_idx += 1
                all_blocks.append(
                    TextBlock(
                        block_id=f"p{page_no}_b{block_idx}",
                        page=page_no,
                        text=block_text,
                        bbox=tuple(block.get("bbox", (0.0, 0.0, 0.0, 0.0))),
                        avg_font_size=avg_font,
                        is_heading=heading_like,
                    )
                )

        sections = self._split_sections(all_blocks)
        return sections, page_images

    def _split_sections(self, blocks: list[TextBlock]) -> list[Section]:
        if not blocks:
            return []

        sections: list[Section] = []
        current_title = "Untitled"
        current_blocks: list[TextBlock] = []
        sec_idx = 1

        def flush() -> None:
            nonlocal sec_idx, current_blocks
            if not current_blocks:
                return
            sections.append(
                Section(
                    section_id=f"sec_{sec_idx:04d}",
                    title=current_title,
                    page_start=current_blocks[0].page,
                    page_end=current_blocks[-1].page,
                    blocks=current_blocks,
                )
            )
            sec_idx += 1
            current_blocks = []

        for block in blocks:
            if block.is_heading and current_blocks:
                flush()
                current_title = block.text.replace("\n", " ").strip()
                continue
            if block.is_heading and not current_blocks:
                current_title = block.text.replace("\n", " ").strip()
                continue
            current_blocks.append(block)

        flush()
        return sections
