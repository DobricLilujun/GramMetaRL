#!/usr/bin/env python3
"""Repair card constraints using audit evidence.

Two passes:
  1. mechanical: any regex-shaped constraint whose value is prose or a bare identifier
     (`same_suffix_for_definite_and_indefinite`) is downgraded to llm_check. Such a value can only
     ever fail, and on professional human translations it does - that class is a large share of the
     measured 63.8% constraint-violation rate.
  2. evidence-driven: cards that violate on a majority of judged-applicable human translations are
     flagged; with --downgrade-flagged they are also downgraded, because in RL their penalty is
     indistinguishable from a wrong translation. Nothing is deleted: the prose check survives as an
     llm_check, and every change is recorded in the description so it is auditable and reversible.

Run scripts/audit_constraint_precision.py first to produce the evidence JSON.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grammetarl.llm_extract import looks_like_prose
from grammetarl.schema import MBGCard

REGEX_KINDS = ("must_include_regex", "must_exclude_regex")


def main() -> None:
    ap = argparse.ArgumentParser(description="Downgrade bad machine constraints, keep the intent")
    ap.add_argument("--cards", default="data/rules/mbg_cards_app.jsonl")
    ap.add_argument("--audit", default="artifacts/audit/constraint_precision.json")
    ap.add_argument("--out", default="data/rules/mbg_cards_app_repaired.jsonl")
    ap.add_argument("--min-support", type=int, default=5, help="Min judged-applicable checks to trust a rate")
    ap.add_argument("--max-violation", type=float, default=0.8)
    ap.add_argument(
        "--downgrade-flagged",
        action="store_true",
        help="Also downgrade cards that violate on most human translations",
    )
    args = ap.parse_args()

    audit_path = Path(args.audit)
    audit = json.loads(audit_path.read_text("utf-8")) if audit_path.exists() else {"per_card": {}}
    per_card = audit.get("per_card", {})

    cards = [MBGCard.model_validate_json(l) for l in Path(args.cards).open("r", encoding="utf-8") if l.strip()]
    n_downgraded = n_flagged = 0
    changes: list[str] = []

    for card in cards:
        evidence = per_card.get(card.id, {})
        applicable = int(evidence.get("applicable", 0))
        violated = int(evidence.get("violated", 0))
        rate = violated / applicable if applicable else 0.0
        flagged = applicable >= args.min_support and rate >= args.max_violation
        if flagged:
            n_flagged += 1

        new_constraints = []
        for c in card.output_constraints:
            if c.kind in REGEX_KINDS and looks_like_prose(c.value or ""):
                new_constraints.append(
                    c.model_copy(
                        update={
                            "kind": "llm_check",
                            "description": f"{c.description} [downgraded: value was not a matcher: {c.value!r}]",
                        }
                    )
                )
                n_downgraded += 1
                changes.append(f"{card.id}: {c.kind} -> llm_check (value={c.value!r})")
            elif flagged and args.downgrade_flagged and c.kind in REGEX_KINDS:
                new_constraints.append(
                    c.model_copy(
                        update={
                            "kind": "llm_check",
                            "description": f"{c.description} [downgraded: violated on {violated}/{applicable} "
                            f"judged-applicable human translations]",
                        }
                    )
                )
                n_downgraded += 1
                changes.append(f"{card.id}: {c.kind} -> llm_check (violation rate {rate:.0%})")
            elif flagged and c.kind in REGEX_KINDS:
                new_constraints.append(
                    c.model_copy(
                        update={
                            "description": f"{c.description} [FLAGGED: violated on {violated}/{applicable} "
                            f"judged-applicable human translations]"
                        }
                    )
                )
            else:
                new_constraints.append(c)
        card.output_constraints = new_constraints

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for c in cards:
            f.write(json.dumps(c.model_dump(), ensure_ascii=False) + "\n")

    still_exec = sum(1 for c in cards if any(x.kind in REGEX_KINDS for x in c.output_constraints))
    print(f"cards={len(cards)} flagged={n_flagged} constraints_downgraded={n_downgraded}")
    print(f"cards still carrying a machine check: {still_exec}")
    for ch in changes[:25]:
        print(f"  {ch}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
