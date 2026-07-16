from __future__ import annotations

from pathlib import Path
from typing import Any

from grammetarl import TranslationVerifier, load_cards
from grammetarl.workspace import get_workspace_paths


def grammar_verify(
    src_sentence: str,
    hypothesis: str,
    applied_rule_ids: list[str],
    cards_path: str | None = None,
    namespace: str | None = None,
) -> dict[str, Any]:
    if not src_sentence.strip() or not hypothesis.strip():
        raise ValueError("src_sentence and hypothesis must be non-empty")

    resolved_cards_path = (
        Path(cards_path) if cards_path else get_workspace_paths(namespace=namespace).rules_cards_file
    )
    cards = load_cards(resolved_cards_path)
    selected = [c for c in cards if c.id in set(applied_rule_ids)]
    verifier = TranslationVerifier()
    report = verifier.verify(sentence=src_sentence, translation=hypothesis, rules=selected)
    return {
        "overall_passed": report.overall_passed,
        "checked_rules": len(report.results),
        "failed_rules": [r.rule_id for r in report.results if not r.passed],
    }
