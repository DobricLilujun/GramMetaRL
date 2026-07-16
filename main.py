from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from grammetarl import RuleIndex, TranslationVerifier, build_mbg_from_pdf, load_cards
from grammetarl.llm_extract import MBGExtractor
from grammetarl.migration import migrate_legacy_layout
from grammetarl.ocr import DeepSeekOCRLocalProvider, NoOCRProvider, OpenAICompatibleVisionOCR
from grammetarl.workspace import get_workspace_paths


def _make_ocr_provider(args: argparse.Namespace):
    if args.ocr_provider == "none":
        return NoOCRProvider()
    if args.ocr_provider == "openai_vision":
        api_key = os.getenv(args.ocr_api_key_env) if args.ocr_api_key_env else None
        return OpenAICompatibleVisionOCR(
            endpoint=args.ocr_endpoint,
            model=args.ocr_model,
            api_key=api_key,
        )
    if args.ocr_provider == "deepseek_local":
        return DeepSeekOCRLocalProvider(model_name=args.deepseek_ocr_model)
    raise ValueError(f"Unsupported OCR provider: {args.ocr_provider}")


def cmd_build(args: argparse.Namespace) -> None:
    if not args.pdf and not args.ocr_pages_jsonl and not args.ocr_pages_dir:
        raise ValueError("Provide --pdf, or provide precomputed OCR via --ocr-pages-jsonl/--ocr-pages-dir")

    ws = get_workspace_paths(args.workspace_root, namespace=args.namespace)
    ws.ensure_layout()
    output_jsonl = Path(args.output) if args.output else ws.default_mbg_output_file

    api_key = os.getenv(args.llm_api_key_env) if args.llm_api_key_env else None
    extractor = MBGExtractor(
        endpoint=args.llm_endpoint,
        model=args.llm_model,
        api_key=api_key,
        timeout=args.llm_timeout,
    )
    ocr_pages_jsonl: Path | None = None
    if args.ocr_pages_jsonl:
        ocr_pages_jsonl = Path(args.ocr_pages_jsonl)
    elif args.ocr_pages_dir:
        ocr_pages_jsonl = Path(args.ocr_pages_dir) / "pages_ocr.jsonl"

    # If precomputed OCR is provided, skip building OCR provider entirely.
    ocr_provider = None if ocr_pages_jsonl else _make_ocr_provider(args)

    cards = build_mbg_from_pdf(
        pdf_path=Path(args.pdf) if args.pdf else None,
        output_jsonl=output_jsonl,
        language=args.language,
        source_id=args.source_id,
        extractor=extractor,
        ocr_provider=ocr_provider,
        max_chunk_chars=args.max_chunk_chars,
        render_dpi=args.render_dpi,
        work_dir=Path(args.work_dir) if args.work_dir else ws.default_mbg_work_dir,
        ocr_pages_jsonl=ocr_pages_jsonl,
        page_start=args.page_start,
        page_end=args.page_end,
    )
    print(f"Built {len(cards)} MBG cards -> {output_jsonl}")


def cmd_retrieve(args: argparse.Namespace) -> None:
    ws = get_workspace_paths(args.workspace_root, namespace=args.namespace)
    cards = load_cards(Path(args.cards) if args.cards else ws.rules_cards_file)
    idx = RuleIndex(cards)
    result = idx.retrieve(sentence=args.sentence, top_k=args.top_k)
    for card, score in result:
        print(f"{score:.3f}\t{card.id}\t{','.join(card.phenomenon_tags)}\t{card.scope}")


def cmd_verify(args: argparse.Namespace) -> None:
    ws = get_workspace_paths(args.workspace_root, namespace=args.namespace)
    cards = load_cards(Path(args.cards) if args.cards else ws.rules_cards_file)
    selected_ids = set(args.rule_ids.split(",")) if args.rule_ids else {c.id for c in cards}
    selected = [c for c in cards if c.id in selected_ids]

    verifier = TranslationVerifier(
        llm_endpoint=args.llm_endpoint,
        llm_model=args.llm_model,
        api_key=os.getenv(args.llm_api_key_env) if args.llm_api_key_env else None,
    )
    report = verifier.verify(
        sentence=args.sentence,
        translation=args.translation,
        rules=selected,
    )
    print(json.dumps(report.__dict__, default=lambda o: o.__dict__, ensure_ascii=False, indent=2))


def cmd_init_workspace(args: argparse.Namespace) -> None:
    ws = get_workspace_paths(args.workspace_root, namespace=args.namespace)
    ws.ensure_layout()
    print("Workspace layout ready:")
    print(f"- namespace: {ws.namespace}")
    print(f"- root: {ws.root}")
    print(f"- data/raw/pdf: {ws.input_pdf_dir}")
    print(f"- data/raw/ocr: {ws.input_ocr_dir}")
    print(f"- data/rules: {ws.rules_dir}")
    print(f"- data/lexicon: {ws.lexicon_dir}")
    print(f"- data/processed: {ws.processed_dir}")
    print(f"- artifacts/ocr: {ws.ocr_artifacts_dir}")
    print(f"- artifacts/extraction: {ws.extraction_artifacts_dir}")
    print(f"- artifacts/rl: {ws.rl_artifacts_dir}")
    print(f"- artifacts/eval: {ws.eval_artifacts_dir}")
    print(f"- artifacts/logs: {ws.logs_dir}")


def cmd_migrate_workspace(args: argparse.Namespace) -> None:
    report = migrate_legacy_layout(
        workspace_root=args.workspace_root,
        namespace=args.namespace,
        mode=args.mode,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Meta Bullet Grammar toolkit")
    parser.add_argument(
        "--workspace-root",
        default=None,
        help="Optional project root. Defaults to GRAMMETARL_ROOT or repository root.",
    )
    parser.add_argument(
        "--namespace",
        default=None,
        help="Experiment namespace. Defaults to GRAMMETARL_NAMESPACE or 'default'.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build-mbg", help="Extract MBG cards from grammar PDF")
    p_build.add_argument("--pdf", required=False, default=None, help="Path to grammar PDF")
    p_build.add_argument(
        "--output",
        required=False,
        default=None,
        help="Output JSONL path",
    )
    p_build.add_argument("--language", required=True, help="Language code, e.g. lb")
    p_build.add_argument("--source-id", required=True, help="Source book identifier")
    p_build.add_argument("--max-chunk-chars", type=int, default=4200)
    p_build.add_argument("--render-dpi", type=int, default=220)
    p_build.add_argument("--work-dir", default=None)
    p_build.add_argument("--page-start", type=int, default=None, help="Start page for extraction (1-based)")
    p_build.add_argument("--page-end", type=int, default=None, help="End page for extraction (1-based)")

    p_build.add_argument(
        "--ocr-provider",
        choices=["none", "openai_vision", "deepseek_local"],
        default="none",
    )
    p_build.add_argument("--ocr-endpoint", default="http://127.0.0.1:8000/v1")
    p_build.add_argument("--ocr-model", default="deepseek-ai/DeepSeek-OCR-2")
    p_build.add_argument("--ocr-api-key-env", default="OPENAI_API_KEY")
    p_build.add_argument("--deepseek-ocr-model", default="deepseek-ai/DeepSeek-OCR-2")
    p_build.add_argument(
        "--ocr-pages-jsonl",
        default=None,
        help="Optional page-level OCR JSONL generated by scripts/extract_pdf_page_ocr.py",
    )
    p_build.add_argument(
        "--ocr-pages-dir",
        default=None,
        help="Directory containing precomputed OCR outputs (expects pages_ocr.jsonl inside)",
    )

    p_build.add_argument("--llm-endpoint", default="http://10.6.32.16:8000/v1")
    p_build.add_argument("--llm-model", default="nvidia/Qwen3.6-35B-A3B-NVFP4")
    p_build.add_argument("--llm-api-key-env", default="OPENAI_API_KEY")
    p_build.add_argument("--llm-timeout", type=int, default=240)
    p_build.set_defaults(func=cmd_build)

    p_init = sub.add_parser("init-workspace", help="Create standardized workspace folders")
    p_init.set_defaults(func=cmd_init_workspace)

    p_migrate = sub.add_parser("migrate-workspace", help="Migrate legacy outputs into standardized layout")
    p_migrate.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    p_migrate.add_argument("--dry-run", action="store_true")
    p_migrate.set_defaults(func=cmd_migrate_workspace)

    p_retrieve = sub.add_parser("retrieve", help="Retrieve matching MBG rules")
    p_retrieve.add_argument(
        "--cards",
        required=False,
        default=None,
        help="MBG JSONL path (defaults to workspace data/rules/mbg_cards.jsonl)",
    )
    p_retrieve.add_argument("--sentence", required=True)
    p_retrieve.add_argument("--top-k", type=int, default=8)
    p_retrieve.set_defaults(func=cmd_retrieve)

    p_verify = sub.add_parser("verify", help="Verify translation with MBG cards")
    p_verify.add_argument(
        "--cards",
        required=False,
        default=None,
        help="MBG JSONL path (defaults to workspace data/rules/mbg_cards.jsonl)",
    )
    p_verify.add_argument("--sentence", required=True)
    p_verify.add_argument("--translation", required=True)
    p_verify.add_argument("--rule-ids", default="", help="Comma-separated card IDs")
    p_verify.add_argument("--llm-endpoint", default=None)
    p_verify.add_argument("--llm-model", default=None)
    p_verify.add_argument("--llm-api-key-env", default="OPENAI_API_KEY")
    p_verify.set_defaults(func=cmd_verify)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
