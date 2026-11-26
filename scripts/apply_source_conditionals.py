#!/usr/bin/env python3
"""Inject hand-authored source-conditional constraints, under admission tests.

Admission (all local, no model):
  1. both regexes compile;
  2. the SOURCE pattern fires on at least one of the card's own example sources - otherwise the card's
     own illustration is not an instance of the condition, and the pairing is guesswork;
  3. for those examples, the TARGET pattern fires on the matching example target - the rule must hold
     on its own illustration before we trust it on 9,000 unseen pairs.
A card that fails (2) or (3) is skipped with the reason printed; nothing ships on authority.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grammetarl.llm_extract import looks_like_prose
from grammetarl.schema import MBGCard, RuleConstraint
from grammetarl.validator import TranslationVerifier


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply hand-authored source-conditional constraints")
    ap.add_argument("--cards", default="data/rules/mbg_cards_recompiled_manual.jsonl")
    ap.add_argument("--spec", default="data/rules/source_conditionals.json")
    ap.add_argument("--out", default="data/rules/mbg_cards_conditional.jsonl")
    ap.add_argument("--replace", action="store_true",
                    help="Drop the card's existing output constraints and keep only the conditional one")
    args = ap.parse_args()

    cards = [MBGCard.model_validate_json(l) for l in Path(args.cards).open("r", encoding="utf-8") if l.strip()]
    by_id = {c.id: c for c in cards}
    spec = json.loads(Path(args.spec).read_text("utf-8"))
    v = TranslationVerifier()

    accepted = rejected = 0
    for t in spec["constraints"]:
        card = by_id.get(t["card_id"])
        if card is None:
            rejected += 1
            print(f"[COND][SKIP] unknown card_id {t['card_id']}")
            continue
        sp = t.get("source_pattern", "")
        alts = [x for x in (t.get("values") or []) if x]
        why = None
        try:
            re.compile(sp)
            for a in alts:
                re.compile(a)
        except re.error as exc:
            why = f"invalid regex: {exc}"
        if why is None and (looks_like_prose(sp) or any(looks_like_prose(a) for a in alts)):
            why = "prose/identifier pattern"
        if why is None and not alts:
            why = "no target pattern"
        if why is None:
            # the card's own illustration must instantiate the condition, and obey the rule on it
            pairs = [(v.normalize_text(e.source_sentence), v.normalize_text(e.target_sentence))
                     for e in card.examples
                     if e.source_sentence and e.target_sentence and len(e.target_sentence.split()) >= 2]
            if not pairs:
                why = "no sentence-grade example pair to test on"
            else:
                inst = []
                for s, tg in pairs:
                    n, _ = v._pattern_hits(sp, s, "sentence")
                    if n > 0:
                        inst.append((s, tg))
                if not inst:
                    why = f"source pattern never fires on the card's own examples {sp!r}"
                else:
                    holds = 0
                    for s, tg in inst:
                        hits = 0
                        for a in alts:
                            m, _ = v._pattern_hits(a, tg, t.get("level", "phrase"))
                            hits += m
                        if hits:
                            holds += 1
                    if holds == 0:
                        why = "target pattern fails on every example that satisfies the source condition"
                    elif holds < len(inst):
                        print(f"[COND][WARN] {card['id'] if isinstance(card, dict) else card.id}: "
                              f"rule holds on {holds}/{len(inst)} of its own examples")
        if why:
            rejected += 1
            print(f"[COND][REJECT] {t['card_id']}: {why}")
            continue
        con = RuleConstraint(
            kind="source_conditional",
            description=f"{t.get('description', '')} [hand-authored conditional: {t.get('why', '')}]",
            source_pattern=sp,
            values=alts,
            value=alts[0] if len(alts) == 1 else None,
            level=t.get("level", "phrase"),
        )
        if args.replace:
            card.output_constraints = [con]
        else:
            card.output_constraints = list(card.output_constraints) + [con]
        accepted += 1
        print(f"[COND][OK] {t['card_id']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for c in cards:
            f.write(json.dumps(c.model_dump(), ensure_ascii=False) + "\n")
    n_cond = sum(1 for c in cards for x in c.output_constraints if x.kind == "source_conditional")
    print(f"accepted={accepted} rejected={rejected} source_conditional_constraints={n_cond} -> {out}")


if __name__ == "__main__":
    main()
