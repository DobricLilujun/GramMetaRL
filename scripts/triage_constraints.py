#!/usr/bin/env python3
"""Triage card constraints by MEASURED precision on professional human translations.

The audit dossiers show one dominant defect: constraints compiled from the card's own example
morpheme rather than from the rule. "The definite article has a full and a reduced form" was
compiled into `^(deen|déin|d'|den|dem|der)$`; "no strong/weak adjective distinction" became
`^dënn-en$` (the example adjective); "possessive articles agree with the possessor" became
`(mäin|däin|säin|hir|eis|är)`. Such a rule can genuinely govern a sentence and still demand the
illustration word, so it fails on correct human translations at a measured rate.

Rather than guessing which constraints are broken, this pass uses the audit itself: a machine
constraint whose violation rate on judge-applicable professional pairs is >= --max-violation (with
>= --min-support applicable pairs) is refuted by evidence and is demoted to an abstaining `note`.
The measured numbers are written into the description, so every demotion is auditable and reversible.

What this buys the RL path: the surviving machine checks are the ones with measured >= 70%
agreement with human translations. What it costs: the demoted cards stop producing labels - the
yield/precision trade-off is printed, not hidden. The demoted majority needs phrase-level
applicability (POS/morphology), which is the P3 agenda.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REFUTED_MESSAGE = "TRIAGED REFUTED by measured audit: applicable={applicable} violated={violated} ({rate:.1%}) on human translations"


def main() -> None:
    ap = argparse.ArgumentParser(description="Evidence-based constraint triage")
    ap.add_argument("--cards", default="data/rules/mbg_cards_recompiled.jsonl")
    ap.add_argument("--audit", default="artifacts/audit/violations_recompiled.json")
    ap.add_argument("--out", default="data/rules/mbg_cards_triaged.jsonl")
    ap.add_argument("--min-support", type=int, default=10, help="Applicable pairs needed before judging")
    ap.add_argument(
        "--max-violation",
        type=float,
        default=0.30,
        help="Violation rate above which the compiled constraint is considered refuted",
    )
    args = ap.parse_args()

    audit = json.loads(Path(args.audit).read_text("utf-8"))
    per_card = audit["per_card"]
    cards = [json.loads(l) for l in Path(args.cards).open("r", encoding="utf-8") if l.strip()]
    machine_kinds = ("must_include_regex", "must_exclude_regex", "must_include_any_regex", "token_count_range")

    demoted_cards = 0
    demoted_constraints = 0
    kept_constraints = 0
    demotion_table = []
    for c in cards:
        stats = per_card.get(c["id"], {})
        applicable = int(stats.get("applicable", 0))
        violated = int(stats.get("violated", 0))
        rate = violated / applicable if applicable else 0.0
        refuted = applicable >= args.min_support and rate >= args.max_violation
        new_constraints = []
        for con in c.get("output_constraints", []):
            if refuted and con.get("kind") in machine_kinds:
                con = dict(con)
                con["kind"] = "note"
                con["level"] = None
                con["description"] = (
                    f"{con.get('description', '')} [{REFUTED_MESSAGE.format(applicable=applicable, violated=violated, rate=rate)}]"
                ).strip()
                new_constraints.append(con)
                demoted_constraints += 1
            else:
                new_constraints.append(con)
                if con.get("kind") in machine_kinds:
                    kept_constraints += 1
        if refuted:
            demoted_cards += 1
            demotion_table.append(
                {"card_id": c["id"], "applicable": applicable, "violated": violated, "rate": round(rate, 3),
                 "statement": c.get("statement", "")[:90]}
            )
        c["output_constraints"] = new_constraints

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for c in cards:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(
        f"cards={len(cards)} demoted_cards={demoted_cards} demoted_constraints={demoted_constraints} "
        f"kept_machine_constraints={kept_constraints}"
    )
    print(f"-> {out}")
    for row in sorted(demotion_table, key=lambda r: -r["applicable"])[:14]:
        print(f"  {row['card_id']} applicable={row['applicable']:4d} violated={row['violated']:4d} rate={row['rate']:.1%}  {row['statement']}")


if __name__ == "__main__":
    main()
