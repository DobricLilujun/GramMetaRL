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
    verdicts = [r.verdict for r in report.results]
    return {
        "overall_passed": report.overall_passed,
        "checked_rules": len(report.results),
        "decided_rules": report.decided,
        "verdicts": verdicts,
        # aligned 1:1 with `verdicts`, so a reliability table can weight each check by its rule
        "rule_ids": [r.rule_id for r in report.results],
        "applicable_rules": sum(1 for r in report.results if r.applicability == "applicable"),
        "not_applicable_rules": sum(1 for r in report.results if r.verdict == "not_applicable"),
        "unknown_applicability_rules": sum(1 for r in report.results if r.applicability == "unknown"),
        "failed_rules": [r.rule_id for r in report.results if r.verdict == "violated"],
        "abstained_rules": [r.rule_id for r in report.results if r.verdict == "not_applicable"],
    }
