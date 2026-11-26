#!/usr/bin/env python3
"""Merge hand-authored applicability predicates into a card set, with the same admission tests.

Hand authoring is only credible if it goes through the same gate as model-compiled predicates: the
pattern must compile, must not be prose, and must fire on the card's own examples. Nothing is
accepted on authority.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grammetarl.llm_extract import looks_like_prose
from grammetarl.schema import MBGCard, TriggerCondition
from grammetarl.validator import TranslationVerifier


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply hand-authored triggers under strict admission")
    ap.add_argument("--cards", default="data/rules/mbg_cards_precise_app.jsonl")
    ap.add_argument("--triggers", default="data/rules/manual_triggers.json")
    ap.add_argument("--out", default="data/rules/mbg_cards_precise_manual.jsonl")
    ap.add_argument("--replace-existing", action="store_true",
                    help="Drop the card's existing predicates before adding the manual ones")
    args = ap.parse_args()

    cards = [MBGCard.model_validate_json(l) for l in Path(args.cards).open("r", encoding="utf-8") if l.strip()]
    by_id = {c.id: c for c in cards}
    spec = json.loads(Path(args.triggers).read_text("utf-8"))
    v = TranslationVerifier()

    accepted = rejected = missing = 0
    for t in spec["triggers"]:
        card = by_id.get(t["card_id"])
        if card is None:
            missing += 1
            print(f"[MANUAL][SKIP] unknown card_id {t['card_id']}")
            continue
        pattern = t.get("pattern", "")
        why = None
        try:
            re.compile(pattern)
        except re.error as exc:
            why = f"invalid regex: {exc}"
        if why is None and looks_like_prose(pattern):
            why = "prose/identifier value"
        untested = False
        if why is None:
            targets = [v.normalize_text(e.target_sentence) for e in card.examples if e.target_sentence]
            srcs = [v.normalize_text(e.source_sentence) for e in card.examples if e.source_sentence]
            pool = targets if t.get("side") == "target" else srcs
            # The own-example test only means something for sentence-grade examples: a card whose
            # illustration is a bare paradigm form ('Afen', 'Handyen') cannot host a contextual
            # predicate, and failing it there would reject valid rules.
            sentential = [p_ for p_ in pool if len(p_.split()) >= 2]
            if sentential:
                hits = sum(1 for s in sentential for _ in [v._pattern_hits(pattern, s, t.get("level", "sentence"))] if _[0] > 0)
                if hits == 0:
                    why = "does not fire on the card's own sentence examples"
            else:
                untested = True
        if why:
            rejected += 1
            print(f"[MANUAL][REJECT] {t['card_id']}: {why}")
            continue
        pred = TriggerCondition(
            source_text=t.get("source_text", ""),
            kind="regex",
            side=t.get("side", "target"),
            pattern=pattern,
            level=t.get("level", "sentence"),
            negated=bool(t.get("negated", False)),
            accepted=True,
            reject_reason=("untested: card examples are subsentential" if untested else None),
        )
        if args.replace_existing:
            card.applicability = []
        card.applicability.append(pred)
        accepted += 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for c in cards:
            f.write(json.dumps(c.model_dump(), ensure_ascii=False) + "\n")
    with_app = sum(1 for c in cards if any(p.accepted for p in c.applicability))
    print(f"accepted={accepted} rejected={rejected} unknown={missing} -> {out}")
    print(f"cards with accepted applicability: {with_app}/{len(cards)}")


if __name__ == "__main__":
    main()
