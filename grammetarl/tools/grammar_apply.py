from __future__ import annotations

from pathlib import Path

from grammetarl import load_cards
from grammetarl.workspace import get_workspace_paths


def grammar_rule_apply(
    rule_ids: list[str],
    src_sentence: str,
    lexical_hints: dict[str, str] | None = None,
    cards_path: str | None = None,
    namespace: str | None = None,
) -> dict[str, object]:
    if not src_sentence.strip():
        raise ValueError("src_sentence cannot be empty")
    if not rule_ids:
        return {"draft": src_sentence, "applied": [], "notes": ["no rules requested"]}

    resolved_cards_path = (
        Path(cards_path) if cards_path else get_workspace_paths(namespace=namespace).rules_cards_file
    )
    cards = {c.id: c for c in load_cards(resolved_cards_path)}
    applied = []
    notes = []
    for rid in rule_ids:
        card = cards.get(rid)
        if not card:
            notes.append(f"missing rule: {rid}")
            continue
        applied.append(rid)
        notes.extend(card.operation_steps[:2])

    # Placeholder deterministic draft. Real generation should be done by model.
    draft = src_sentence
    if lexical_hints:
        notes.append(f"lexical_hints={len(lexical_hints)}")
    return {"draft": draft, "applied": applied, "notes": notes}
