from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from grammetarl.workspace import get_workspace_paths


def dictionary_lookup(
    token: str,
    lemma: str | None = None,
    pos: str | None = None,
    context: str | None = None,
    lexicon_path: str | None = None,
    max_results: int = 5,
    namespace: str | None = None,
) -> list[dict[str, Any]]:
    if not token.strip():
        raise ValueError("token cannot be empty")
    rows: list[dict[str, Any]] = []
    p = Path(lexicon_path) if lexicon_path else get_workspace_paths(namespace=namespace).lexicon_file
    if not p.exists():
        return rows

    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            keys = [str(obj.get("lemma", "")), str(obj.get("surface", ""))]
            hay = " ".join(keys).lower()
            if token.lower() in hay or (lemma and lemma.lower() in hay):
                if pos and str(obj.get("pos", "")).lower() != pos.lower():
                    continue
                rows.append(obj)
                if len(rows) >= max_results:
                    break
    return rows
