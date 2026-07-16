
from __future__ import annotations

import argparse
import json
import os
import re
import traceback
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import fitz

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
from transformers import AutoModel, AutoTokenizer

from grammetarl.prompt_manager import render_prompt
from grammetarl.workspace import get_workspace_paths

try:
    from tqdm.auto import tqdm
except Exception:  # noqa: BLE001
    tqdm = None


MARKER_PATTERN = re.compile(
    r'^<\|ref\|>(?P<label>.*?)<\|/ref\|><\|det\|>(?P<bbox>\[\[.*?\]\])<\|/det\|>$'
)
DEFAULT_OCR_INFER_PROMPT = render_prompt("ocr_infer_markdown.j2")


def render_pdf_page(pdf_path: Path, page_number: int, image_path: Path, dpi: int) -> Path:
    pdf_doc = fitz.open(pdf_path)
    try:
        if page_number < 1 or page_number > pdf_doc.page_count:
            raise ValueError(f"page_number must be between 1 and {pdf_doc.page_count}")
        page = pdf_doc.load_page(page_number - 1)
        zoom = dpi / 72.0
        pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        image_path.parent.mkdir(parents=True, exist_ok=True)
        pixmap.save(image_path)
        return image_path
    finally:
        pdf_doc.close()


def parse_ocr_stdout(raw_stdout: str) -> list[dict[str, object]]:
    blocks: list[dict[str, object]] = []
    current_block: dict[str, object] | None = None

    for line in raw_stdout.splitlines():
        marker_match = MARKER_PATTERN.match(line.strip())
        if marker_match:
            if current_block is not None:
                current_block["text"] = "\n".join(current_block.pop("text_lines", [])).strip()
                blocks.append(current_block)
            current_block = {
                "label": marker_match.group("label"),
                "bbox": json.loads(marker_match.group("bbox"))[0],
                "text_lines": [],
            }
            continue

        if current_block is None:
            continue

        if line.startswith("===============save results:="):
            current_block["text"] = "\n".join(current_block.pop("text_lines", [])).strip()
            blocks.append(current_block)
            current_block = None
            break

        current_block["text_lines"].append(line)

    if current_block is not None:
        current_block["text"] = "\n".join(current_block.pop("text_lines", [])).strip()
        blocks.append(current_block)

    return blocks


def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2",
    ).eval().cuda()
    return tokenizer, model


def run_page_ocr(
    pdf_path: Path,
    page_number: int,
    output_dir: Path,
    tokenizer,
    model,
    prompt: str,
    dpi: int,
    base_size: int,
    image_size: int,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    page_image_path = output_dir / f"page_{page_number:04d}.png"
    render_pdf_page(pdf_path, page_number, page_image_path, dpi)

    stdout_buffer = StringIO()
    with redirect_stdout(stdout_buffer):
        model.infer(
            tokenizer,
            prompt=prompt,
            image_file=str(page_image_path),
            output_path=str(output_dir),
            base_size=base_size,
            image_size=image_size,
            crop_mode=True,
            save_results=True,
        )

    raw_stdout = stdout_buffer.getvalue()
    result_md_path = output_dir / "result.mmd"
    markdown_text = result_md_path.read_text(encoding="utf-8") if result_md_path.exists() else ""
    blocks = parse_ocr_stdout(raw_stdout)

    structured_result = {
        "pdf_path": str(pdf_path),
        "page_number": page_number,
        "page_image_path": str(page_image_path),
        "output_dir": str(output_dir),
        "result_markdown_path": str(result_md_path),
        "result_markdown": markdown_text,
        "stdout_blocks": blocks,
        "raw_stdout": raw_stdout,
    }

    structured_json_path = output_dir / f"page_{page_number:04d}_ocr_structured.json"
    raw_stdout_path = output_dir / f"page_{page_number:04d}_ocr_stdout.txt"
    structured_json_path.write_text(json.dumps(structured_result, ensure_ascii=False, indent=2), encoding="utf-8")
    raw_stdout_path.write_text(raw_stdout, encoding="utf-8")

    return {
        **structured_result,
        "structured_json_path": str(structured_json_path),
        "raw_stdout_path": str(raw_stdout_path),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch OCR PDF pages with DeepSeek and export structured JSONL.")
    parser.add_argument("--workspace-root", type=Path, default=None, help="Project root path")
    parser.add_argument("--namespace", default=None, help="Experiment namespace")
    parser.add_argument("--pdf", required=True, type=Path, help="Path to the source PDF")
    parser.add_argument("--page", type=int, default=None, help="1-based page number to OCR (single-page mode)")
    parser.add_argument("--start-page", type=int, default=1, help="1-based start page for batch mode")
    parser.add_argument("--end-page", type=int, default=None, help="1-based end page for batch mode; defaults to last page")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for per-page artifacts and summary JSONL outputs",
    )
    parser.add_argument(
        "--page-jsonl",
        type=Path,
        default=None,
        help="Path for page-level JSONL; defaults to <output-dir>/pages_ocr.jsonl",
    )
    parser.add_argument(
        "--block-jsonl",
        type=Path,
        default=None,
        help="Path for block-level JSONL; defaults to <output-dir>/blocks_ocr.jsonl",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing remaining pages if one page fails",
    )
    parser.add_argument(
        "--model-name",
        default="deepseek-ai/DeepSeek-OCR-2",
        help="HF model name for DeepSeek OCR",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_OCR_INFER_PROMPT,
        help="Prompt passed to the OCR model",
    )
    parser.add_argument("--dpi", type=int, default=220, help="Render DPI for the PDF page image")
    parser.add_argument("--base-size", type=int, default=1024, help="DeepSeek OCR base_size argument")
    parser.add_argument("--image-size", type=int, default=768, help="DeepSeek OCR image_size argument")
    parser.add_argument(
        "--cuda-visible-devices",
        default="0",
        help="CUDA_VISIBLE_DEVICES value to use before loading the model",
    )
    parser.add_argument(
        "--overwrite-jsonl",
        action="store_true",
        help="Overwrite existing JSONL files instead of appending",
    )
    return parser


def _count_block_types(blocks: list[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for block in blocks:
        label = str(block.get("label", "unknown"))
        counts[label] = counts.get(label, 0) + 1
    return counts


def _write_jsonl_line(path: Path, row: dict[str, object], append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _iter_target_pages(pdf_path: Path, page: int | None, start_page: int, end_page: int | None) -> tuple[list[int], int]:
    with fitz.open(pdf_path) as doc:
        page_count = doc.page_count

    if page is not None:
        if page < 1 or page > page_count:
            raise ValueError(f"--page must be between 1 and {page_count}")
        return [page], page_count

    resolved_end_page = end_page if end_page is not None else page_count
    if start_page < 1:
        raise ValueError("--start-page must be >= 1")
    if resolved_end_page < start_page:
        raise ValueError("--end-page must be >= --start-page")
    if resolved_end_page > page_count:
        raise ValueError(f"--end-page must be <= {page_count}")

    return list(range(start_page, resolved_end_page + 1)), page_count


def _page_progress_iter(target_pages: list[int]):
    if tqdm is not None:
        return tqdm(target_pages, desc="OCR pages", unit="page")
    return target_pages


def _print_fallback_progress(index: int, total: int, page_number: int) -> None:
    pct = (index / total) * 100 if total else 100.0
    print(f"[PROGRESS] {index}/{total} ({pct:.1f}%) page={page_number}")


def main() -> None:
    args = build_arg_parser().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    ws = get_workspace_paths(args.workspace_root, namespace=args.namespace)
    ws.ensure_layout()

    if not torch.cuda.is_available():
        raise RuntimeError("DeepSeek OCR requires a CUDA-capable GPU")

    target_pages, page_count = _iter_target_pages(args.pdf, args.page, args.start_page, args.end_page)
    output_dir = args.output_dir or ws.default_ocr_run_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    page_jsonl_path = args.page_jsonl or (output_dir / "pages_ocr.jsonl")
    block_jsonl_path = args.block_jsonl or (output_dir / "blocks_ocr.jsonl")

    if args.overwrite_jsonl:
        if page_jsonl_path.exists():
            page_jsonl_path.unlink()
        if block_jsonl_path.exists():
            block_jsonl_path.unlink()

    tokenizer, model = load_model(args.model_name)

    success_pages = 0
    failed_pages = 0
    wrote_page_header = False
    wrote_block_header = False

    page_iter = _page_progress_iter(target_pages)
    total_pages = len(target_pages)

    for idx, page_number in enumerate(page_iter, start=1):
        if tqdm is None:
            _print_fallback_progress(idx, total_pages, page_number)

        page_dir = output_dir / f"page_{page_number:04d}"
        try:
            result = run_page_ocr(
                pdf_path=args.pdf,
                page_number=page_number,
                output_dir=page_dir,
                tokenizer=tokenizer,
                model=model,
                prompt=args.prompt,
                dpi=args.dpi,
                base_size=args.base_size,
                image_size=args.image_size,
            )

            blocks = result["stdout_blocks"]
            block_type_counts = _count_block_types(blocks)

            page_row = {
                "pdf_path": str(args.pdf),
                "pdf_total_pages": page_count,
                "page_number": page_number,
                "page_image_path": result["page_image_path"],
                "result_markdown_path": result["result_markdown_path"],
                "structured_json_path": result["structured_json_path"],
                "raw_stdout_path": result["raw_stdout_path"],
                "block_count": len(blocks),
                "block_type_counts": block_type_counts,
                "blocks": blocks,
            }
            _write_jsonl_line(page_jsonl_path, page_row, append=wrote_page_header or (not args.overwrite_jsonl))
            wrote_page_header = True

            for block_index, block in enumerate(blocks, start=1):
                block_row = {
                    "pdf_path": str(args.pdf),
                    "page_number": page_number,
                    "block_index": block_index,
                    "block_type": block.get("label", "unknown"),
                    "bbox": block.get("bbox", []),
                    "text": block.get("text", ""),
                }
                _write_jsonl_line(block_jsonl_path, block_row, append=wrote_block_header or (not args.overwrite_jsonl))
                wrote_block_header = True

            success_pages += 1
            if tqdm is not None:
                page_iter.set_postfix({"ok": success_pages, "fail": failed_pages, "page": page_number}, refresh=False)
            print(
                f"[OK] page={page_number} blocks={len(blocks)} "
                f"types={block_type_counts} page_dir={page_dir}"
            )

        except Exception as exc:  # noqa: BLE001
            failed_pages += 1
            if tqdm is not None:
                page_iter.set_postfix({"ok": success_pages, "fail": failed_pages, "page": page_number}, refresh=False)
            err_row = {
                "pdf_path": str(args.pdf),
                "page_number": page_number,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            _write_jsonl_line(page_jsonl_path, err_row, append=wrote_page_header or (not args.overwrite_jsonl))
            wrote_page_header = True
            print(f"[ERROR] page={page_number} error={exc}")
            if not args.continue_on_error:
                raise

    print("\n=== OCR Batch Summary ===")
    print(f"PDF: {args.pdf}")
    print(f"Target pages: {len(target_pages)}")
    print(f"Success pages: {success_pages}")
    print(f"Failed pages: {failed_pages}")
    print(f"Page JSONL: {page_jsonl_path}")
    print(f"Block JSONL: {block_jsonl_path}")


if __name__ == "__main__":
    main()
