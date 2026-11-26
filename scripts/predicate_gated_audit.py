#!/usr/bin/env python3
"""Constraint agreement under PREDICATE-gated applicability (no judge, no retrieval).

Why a second audit: `audit_constraint_precision.py` measures constraint agreement on pairs the
offline judge marked applicable. But the verifier at training time gates on the card's own compiled
predicate, not on the judge. Two different denominators mean the reliability weights and the gating
disagree about what "this rule applies here" means - and a weight built one way used under gating the
other way is a silent inconsistency.

This script produces the same artifact shape as the judge-based audit, but every decision comes from
the card's own accepted predicates (TranslationVerifier(require_applicability=True)), so the weights
it feeds are measured under exactly the gating the pipeline uses. It also needs no judge cache, so it
can run over the full pair pool.

Caveat it does NOT paper over: false-penalty counting only makes sense for pairs whose reference is a
professional human translation, so violation rates are reported per source (Tech-in-GOV is CC0
professional; LuxAlign pairs are only semantically similar by its own card).
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grammetarl.schema import MBGCard
from grammetarl.validator import TranslationVerifier


def main() -> None:
    ap = argparse.ArgumentParser(description="Constraint agreement under predicate-gated applicability")
    ap.add_argument("--cards", default="data/rules/mbg_cards_recompiled_manual.jsonl")
    ap.add_argument("--pairs", default="data/external/pairs_en_lb.jsonl")
    ap.add_argument("--skip", type=int, default=0, help="Skip the first N candidate pairs (held-out measurement: author on the fit half, measure on the rest)")
    ap.add_argument("--rows", type=int, default=0, help="0 = every pair in the pool")
    ap.add_argument("--source-filter", default=None,
                    help="Restrict to one source id. Weights feeding an RL reward should be measured "
                    "on professional human translations (techgov_lu_en), not on semantically-similar "
                    "news alignments, or the agreement estimate is biased by reference noise.")
    ap.add_argument("--sample", type=int, default=10, help="counter-examples to keep per run")
    ap.add_argument("--out", default="artifacts/audit/predicate_gated_precision.json")
    args = ap.parse_args()

    cards = [MBGCard.model_validate_json(l) for l in Path(args.cards).open("r", encoding="utf-8") if l.strip()]
    # Cards are measurable when applicability is decidable at all: either an accepted predicate, or the
    # source condition of a source_conditional constraint (the validator derives it from the source).
    gated = [c for c in cards
             if any(p.accepted and p.kind == "regex" and p.pattern for p in c.applicability)
             or any(x.kind == "source_conditional" and x.source_pattern for x in c.output_constraints)]
    v = TranslationVerifier(require_applicability=True)

    skipped = 0
    pairs = []
    with Path(args.pairs).open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if args.source_filter and rec.get("source") != args.source_filter:
                continue
            if skipped < args.skip:  # held-out measurement: the fit half stays out of this run
                skipped += 1
                continue
            pairs.append(rec)
            if args.rows and len(pairs) >= args.rows:
                break

    per_card: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    by_source: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    stats = collections.Counter()
    examples: list[dict] = []

    for rec in pairs:
        src, tgt = rec["src"], rec["tgt_ref"]
        rep = v.verify(src, tgt, gated)
        for r in rep.results:
            if r.verdict == "not_applicable":
                per_card[r.rule_id]["not_applicable"] += 1
                continue
            if r.verdict not in ("satisfied", "violated"):
                continue
            per_card[r.rule_id]["applicable"] += 1
            per_card[r.rule_id][r.verdict] += 1
            stats[r.verdict] += 1
            src_name = rec.get("source", "external")
            by_source[src_name]["applicable"] += 1
            by_source[src_name][r.verdict] += 1
            if r.verdict == "violated" and len(examples) < args.sample:
                card = next(c for c in gated if c.id == r.rule_id)
                examples.append(
                    {
                        "card_id": r.rule_id,
                        "statement": card.statement[:180],
                        "constraints": [
                            {"kind": x.kind, "level": x.level, "value": x.value, "values": x.values}
                            for x in card.output_constraints
                            if "regex" in (x.kind or "")
                        ],
                        "en": src[:180],
                        "lb": tgt[:180],
                        "messages": r.messages[-3:],
                    }
                )

    stats["pairs_scanned"] = len(pairs)
    stats["cards_with_predicates"] = len(gated)
    stats["decided_checks"] = stats["satisfied"] + stats["violated"]
    overall = stats["violated"] / max(1, stats["decided_checks"])

    out = {
        "mode": "predicate_gated",
        "stats": dict(stats),
        "violation_rate_on_human_translations": overall,
        "violation_rate_by_source": {
            s: {
                "applicable": c["applicable"],
                "satisfied": c["satisfied"],
                "violated": c["violated"],
                "violation_rate": c["violated"] / max(1, c["applicable"]),
            }
            for s, c in by_source.items()
        },
        "per_card": {cid: dict(c) for cid, c in per_card.items()},
        "counter_examples": examples,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), "utf-8")

    print(f"pairs_scanned={len(pairs)} cards_with_predicates={len(gated)}")
    print(f"decided_checks={stats['decided_checks']} satisfied={stats['satisfied']} violated={stats['violated']}")
    print(f"violation rate (predicate-gated): {overall:.1%}")
    for s, c in by_source.items():
        print(f"  source={s} applicable={c['applicable']} satisfied={c['satisfied']} violated={c['violated']} "
              f"rate={c['violated']/max(1,c['applicable']):.1%}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
