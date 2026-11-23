#!/usr/bin/env python3
"""Audit the *precision* of compiled card constraints on professional human translations.

The applicability story is only half of P1. After the offline judge says "this rule governs this
pair", the label comes from the compiled output constraint. On a 23-pair probe, 9 of the 11
judge-applicable pairs came back as violated - on professional human translations, which should
satisfy the grammar of their own language. So either the constraints are wrong/too narrow, or the
judge over-applies rules. This script separates the two without any LLM:

  - it replays the cached applicability decisions (reproducible, no model calls),
  - re-runs only the machine constraints,
  - reports per-card applicable / satisfied / violated rates, and
  - dumps the concrete counter-example (pair + failing constraint + message) for human triage.

Cards with a high violation rate on human translations are candidate bad programs: they must be
re-compiled or retired, because in RL their penalty is indistinguishable from a wrong translation.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grammetarl.index import RuleIndex
from grammetarl.schema import MBGCard
from grammetarl.validator import TranslationVerifier
from scripts.label_with_judge import key  # same key derivation, so the cache is reusable


def main() -> None:
    ap = argparse.ArgumentParser(description="Constraint-precision audit from the judge cache")
    ap.add_argument("--pairs", default="data/external/pairs_en_lb.jsonl")
    ap.add_argument("--cards", default="data/rules/mbg_cards_app.jsonl")
    ap.add_argument("--cache", default="data/external/judge_cache.jsonl")
    ap.add_argument("--rows", type=int, default=1000)
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--sample", type=int, default=12)
    ap.add_argument("--out", default="artifacts/audit/constraint_precision.json")
    args = ap.parse_args()

    cache: dict[str, bool] = {}
    cp = Path(args.cache)
    if not cp.exists():
        raise SystemExit(f"no judge cache at {cp}; run scripts/label_with_judge.py first")
    for line in cp.open("r", encoding="utf-8"):
        if line.strip():
            rec = json.loads(line)
            cache[rec["key"]] = bool(rec["applicable"])

    cards = [MBGCard.model_validate_json(l) for l in Path(args.cards).open("r", encoding="utf-8") if l.strip()]
    cards_by_id = {c.id: c for c in cards}
    index = RuleIndex(cards)
    verifier = TranslationVerifier(require_applicability=False)

    per_card: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    examples: list[dict] = []
    stats = collections.Counter()

    with Path(args.pairs).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            src, tgt = rec.get("src", ""), rec.get("tgt_ref", "")
            if not src or not tgt:
                continue
            stats["pairs"] += 1
            if stats["pairs"] > args.rows:
                break
            for h in index.retrieve(src, top_k=args.top_k):
                card = None
                if isinstance(h, tuple):
                    for item in h:
                        if isinstance(item, MBGCard):
                            card = item
                            break
                else:
                    card = getattr(h, "card", h)
                if card is None:
                    continue
                c = cards_by_id[card.id]
                if not any("regex" in (x.kind or "") for x in c.output_constraints):
                    continue
                if any(p.accepted for p in c.applicability):
                    continue  # regex-decidable already; not a judge case
                k = key(src, tgt, c.id)
                if k not in cache:
                    continue
                stats["cached_checks"] += 1
                if not cache[k]:
                    per_card[c.id]["not_applicable"] += 1
                    continue
                per_card[c.id]["applicable"] += 1
                report = verifier.verify(src, tgt, [c])
                verdict = report.results[0].verdict
                per_card[c.id][verdict] += 1
                stats[verdict] += 1
                if verdict == "violated" and len(examples) < args.sample:
                    examples.append(
                        {
                            "card_id": c.id,
                            "statement": c.statement[:180],
                            "constraints": [
                                {"kind": x.kind, "level": x.level, "value": x.value}
                                for x in c.output_constraints
                                if "regex" in (x.kind or "")
                            ],
                            "en": src[:160],
                            "lb": tgt[:160],
                            "messages": report.results[0].messages,
                        }
                    )

    ranked = sorted(
        per_card.items(),
        key=lambda kv: kv[1]["violated"] / max(1, kv[1]["applicable"]),
        reverse=True,
    )
    print(f"pairs scanned: {stats['pairs']}  cached applicability checks: {stats['cached_checks']}")
    print(f"applicable={stats['satisfied'] + stats['violated']} "
          f"satisfied={stats['satisfied']} violated={stats['violated']}")
    overall = stats["violated"] / max(1, stats["satisfied"] + stats["violated"])
    print(f"constraint violation rate on human translations: {overall:.1%} "
          f"(a high rate means the compiled programs are wrong, not the translators)")
    print("\nper-card (applicable -> violated), worst first:")
    for cid, cnt in ranked[:15]:
        rate = cnt["violated"] / max(1, cnt["applicable"])
        print(f"  {cid} applicable={cnt['applicable']:3d} satisfied={cnt['satisfied']:3d} "
              f"violated={cnt['violated']:3d} rate={rate:5.1%}  {cards_by_id[cid].statement[:60]}")
    print("\ncounter-examples for human triage:")
    for ex in examples:
        print(f"  [{ex['card_id']}] {ex['statement'][:90]}")
        print(f"     en: {ex['en']}")
        print(f"     lb: {ex['lb']}")
        for cst in ex["constraints"]:
            print(f"     constraint {cst['kind']} level={cst['level']} = {str(cst['value'])[:70]!r}")
        for m in ex["messages"][:3]:
            print(f"     msg: {m[:120]}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "stats": dict(stats),
                "violation_rate_on_human_translations": overall,
                "per_card": {cid: dict(cnt) for cid, cnt in ranked},
                "counter_examples": examples,
            },
            ensure_ascii=False,
            indent=2,
        ),
        "utf-8",
    )
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
