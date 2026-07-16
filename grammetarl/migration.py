from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from .workspace import get_workspace_paths


@dataclass(slots=True)
class MigrationAction:
    source: str
    target: str
    action: str
    status: str
    note: str = ""


def _transfer_path(src: Path, dst: Path, mode: str) -> tuple[str, str]:
    if not src.exists():
        return "skip", "source_missing"
    if dst.exists():
        return "skip", "target_exists"

    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        dst.symlink_to(src, target_is_directory=src.is_dir())
        return "symlink", "ok"

    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
    return "copy", "ok"


def migrate_legacy_layout(
    workspace_root: str | Path | None = None,
    namespace: str | None = None,
    mode: str = "symlink",
    dry_run: bool = False,
) -> dict[str, object]:
    if mode not in {"symlink", "copy"}:
        raise ValueError("mode must be one of: symlink, copy")

    ws = get_workspace_paths(workspace_root, namespace=namespace)
    ws.ensure_layout()
    legacy_root = ws.root

    mapping: list[tuple[Path, Path]] = [
        (legacy_root / "artifacts" / "deepseek_ocr_fullbook", ws.default_ocr_run_dir),
        (legacy_root / "artifacts" / "deepseek_ocr_fullbook_smoke", ws.ocr_artifacts_dir / "deepseek_fullbook_smoke"),
        (legacy_root / "artifacts" / "deepseek_ocr_page1", ws.ocr_artifacts_dir / "deepseek_page1"),
        (legacy_root / "artifacts" / "mbg_5page_qwen_test", ws.extraction_artifacts_dir / "mbg_5page_qwen_test"),
        (legacy_root / "artifacts" / "lb_mbg_cards_from_ocr.jsonl", ws.rules_cards_file),
        (legacy_root / "data" / "lb_mbg_cards.jsonl", ws.rules_cards_file),
    ]

    actions: list[MigrationAction] = []
    for src, dst in mapping:
        if dry_run:
            status = "planned" if src.exists() and not dst.exists() else "skip"
            note = "" if status == "planned" else ("source_missing" if not src.exists() else "target_exists")
            actions.append(MigrationAction(source=str(src), target=str(dst), action=mode, status=status, note=note))
            continue

        action, status = _transfer_path(src, dst, mode=mode)
        actions.append(MigrationAction(source=str(src), target=str(dst), action=action, status=status))

    report = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "namespace": ws.namespace,
        "mode": mode,
        "dry_run": dry_run,
        "actions": [asdict(a) for a in actions],
    }

    report_name = f"migration_{ws.namespace}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    report_path = ws.logs_dir / report_name
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report
