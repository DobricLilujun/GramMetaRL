#!/usr/bin/env python3
"""Which TOKEN actually satisfies a `source_conditional` check? (false-satisfy audit)

Agreement-vs-professionals and discrimination-vs-perturbations both miss one failure mode, and policy
rollouts exposed it: a garbled translation can earn +1 because a REGEX HIT EXISTS THAT IS NOT THE
GRAMMAR AT ALL. Example, English "How much does AI already meddle with our everyday life?" rendered
as gibberish containing "mengen AI" - `\bmeng(en)?\b` matched the verb *mengen* ("to mean/think"), not
a possessive article, and the reward went up.

Luxembourgish orthography capitalizes every noun, so a possessive article is followed by an uppercase
token. Classifying the token AFTER each match therefore separates noun-governing matches from
verb/homograph matches without a parser, and the reference side tells us whether the requirement is
orthographically true in professional text (it is) rather than a guess.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grammetarl.schema import MBGCard
from grammetarl.validator import has_source_conditional


def classify(text: str, m: re.Match[str]) -> tuple[str, str]:
    nxt = text[m.end():m.end() + 24]
    nm = re.match(r"[\s\"'()\u2018\u2019\[\]]*", nxt)
    rest = nxt[nm.end():] if nm else nxt
    tok = re.match(r"[^\s,.;:!?\u2019\"')\]]+", rest)
    if not tok:
        return ("no_next_token", "")
    t = tok.group(0)
    if re.match(r"[A-ZÀ-ÞÄÖ]", t):
        return ("next_starts_upper", t)
    return ("next_starts_lower", t)


def main() -> None:
    ap = argparse.ArgumentParser(description="audit what actually satisfies a source_conditional check")
    ap.add_argument("--cards", default="data/rules/mbg_cards_condonly.jsonl")
    ap.add_argument("--rule", required=True)
    ap.add_argument("--pairs", default="", help="reference pairs JSONL (professional targets)")
    ap.add_argument("--rollouts", default="", help="rollout JSONL from scripts/p1_reward_smoke.py")
    ap.add_argument("--source-filter", default="techgov_lu_en")
    ap.add_argument("--rows", type=int, default=0)
    ap.add_argument("--examples", type=int, default=3)
    args = ap.parse_args()

    cards = {c.id: c for c in (MBGCard.model_validate_json(l)
                               for l in Path(args.cards).open("r", encoding="utf-8") if l.strip())}
    card = cards.get(args.rule)
    if card is None or not has_source_conditional(card):
        raise SystemExit(f"{args.rule} missing or not source_conditional")
    cons = [k for k in card.output_constraints if k.kind == "source_conditional"]
    src = re.compile(cons[0].source_pattern, re.I)
    vals = [re.compile(v, re.I) for v in cons[0].values]

    def audit(label: str, rows) -> None:
        cls: collections.Counter = collections.Counter()
        forms: collections.Counter = collections.Counter()
        suspicious: list[tuple[str, str, str]] = []
        n_app = n_sat = 0
        for a, b in rows:
            if not src.search(a):
                continue
            n_app += 1
            hits = [(rx, m) for rx in vals for m in rx.finditer(b)]
            if hits:
                n_sat += 1
            for rx, m in hits:
                form = b[m.start():m.end()]
                kinds, tok = classify(b, m)
                cls[f"{form} :: {kinds}"] += 1
                forms[form] += 1
                if kinds == "next_starts_lower" and len(suspicious) < args.examples:
                    suspicious.append((a[:70], b[:90], f"{form} -> {tok}"))
        print(f"\n== {label}: applicable={n_app} satisfied={n_sat} "
              f"({(n_sat / n_app if n_app else 0):.1%})")
        for k, c in cls.most_common(10):
            print(f"   {k:46} {c:6}")
        for a, b, why in suspicious:
            print(f"   [suspicious {why}]\n     EN {a}\n     LB {b}")

    if args.pairs:
        rows = []
        with Path(args.pairs).open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if args.rows and i >= args.rows:
                    break
                rec = json.loads(line)
                if args.source_filter and rec.get("source") != args.source_filter:
                    continue
                rows.append((rec["src"], rec["tgt_ref"]))
        audit(f"references ({Path(args.pairs).name})", rows)
    if args.rollouts:
        rows = []
        with Path(args.rollouts).open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                rows.append((rec["src"], rec["hyp"]))
        audit(f"policy rollouts ({Path(args.rollouts).name})", rows)


if __name__ == "__main__":
    main()
