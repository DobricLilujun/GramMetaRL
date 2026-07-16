from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from grammetarl import RuleIndex, load_cards
from grammetarl.workspace import get_workspace_paths


@dataclass(slots=True)
class GrammarSearchResult:
    rule_id: str
    score: float
    tags: list[str]
    scope: str


def grammar_rule_search(
    query: str,
    src_sentence: str,
    draft: str | None = None,
    top_k: int = 5,
    cards_path: str | None = None,
    namespace: str | None = None,
) -> list[dict[str, Any]]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not src_sentence.strip():
        raise ValueError("src_sentence cannot be empty")

    resolved_cards_path = (
        Path(cards_path) if cards_path else get_workspace_paths(namespace=namespace).rules_cards_file
    )
    cards = load_cards(resolved_cards_path)
    idx = RuleIndex(cards)
    packed = " ".join(p for p in [query, src_sentence, draft or ""] if p)
    found = idx.retrieve(sentence=packed, top_k=top_k)

    return [
        {
            "rule_id": c.id,
            "score": float(score),
            "tags": c.phenomenon_tags,
            "scope": c.scope,
        }
        for c, score in found
    ]
