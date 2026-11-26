#!/usr/bin/env python3
"""Measure compiled applicability predicates against the offline judge decisions.

Applicability is the gate that decides whether a machine check runs at all, so a wrong predicate is
worse than a wrong constraint: a false positive re-introduces the false penalty we removed, and a
false negative silently deletes real signal. The judge cache (sha1(pair, rule) -> applicable, 4k+
decisions made once by a strong model) gives labelled ground truth for exactly this, so predicates
can be scored with zero further model calls.

Reports per card: predicate fires, judge-applicable, precision, recall, F1.
Also reports the end-to-end point of having predicates: how many pairs get a verified label when the
verifier requires applicability instead of consulting the offline judge.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grammetarl.schema import MBGCard
from grammetarl.validator import TranslationVerifier
from scripts.label_with_judge import key


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Score applicability predicates against judge decisions")
    ap.add_argument("--cards", default="data/rules/mbg_cards_precise_app.jsonl")
    ap.add_argument("--pairs", default="data/external/pairs_en_lb.jsonl")
    ap.add_argument("--cache", default="data/external/judge_cache.jsonl")
    ap.add_argument("--rows", type=int, default=1200)
    ap.add_argument(
        "--split-rows",
        type=int,
        default=0,
        help="Split the pair list here and report metrics separately for each half (hand-authored "
        "predicates were inspired by examples inside the pool, so a held-out half is the honest number)",
    )
    ap.add_argument("--out", default="artifacts/audit/trigger_quality.json")
    args = ap.parse_args()

    cards = [MBGCard.model_validate_json(l) for l in Path(args.cards).open("r", encoding="utf-8") if l.strip()]
    cache: dict[str, bool] = {}
    for line in Path(args.cache).open("r", encoding="utf-8"):
        rec = json.loads(line)
        cache.setdefault(rec["key"], bool(rec["applicable"]))

    pairs = []
    with Path(args.pairs).open("r", encoding="utf-8") as f:
        for line in f:
            pairs.append(json.loads(line))
            if len(pairs) >= args.rows:
                break

    v = TranslationVerifier()

    def fires(card: MBGCard, src: str, tgt: str) -> bool | None:
        """True/False when the card has executable predicates; None when it cannot decide."""
        preds = [p for p in card.applicability if p.accepted and p.kind == "regex" and p.pattern]
        if not preds:
            return None
        for p in preds:
            text = tgt if p.side == "target" else src
            n, _ = v._pattern_hits(p.pattern, v.normalize_text(text), p.level or "phrase")
            hit = n > 0
            if p.negated:
                hit = not hit
            if hit:
                return True
        return False

    def score(card: MBGCard, rows: list[dict]) -> dict:
        judged = applicable = fired = both = 0
        for rec in rows:
            k = key(rec["src"], rec["tgt_ref"], card.id)
            if k not in cache:
                continue
            judged += 1
            ja = cache[k]
            f = fires(card, rec["src"], rec["tgt_ref"])
            if f is None:
                continue
            applicable += int(ja)
            fired += int(f)
            both += int(f and ja)
        prec = both / fired if fired else 0.0
        rec_ = both / applicable if applicable else 0.0
        return {
            "judged_pairs": judged,
            "judge_applicable": applicable,
            "predicate_fires": fired,
            "both": both,
            "precision": round(prec, 3),
            "recall": round(rec_, 3),
            "f1": round(2 * prec * rec_ / (prec + rec_), 3) if prec + rec_ else 0.0,
        }

    split = args.split_rows if 0 < args.split_rows < len(pairs) else None
    per_card = {}
    micro = Counter()
    micro_ho = Counter()
    for c in cards:
        decidable = any(p.accepted and p.kind == "regex" and p.pattern for p in c.applicability)
        if not decidable:
            continue
        st = score(c, pairs)
        st["statement"] = c.statement[:70]
        if split:
            st_ho = score(c, pairs[split:])
            st["heldout"] = st_ho
            micro_ho["applicable"] += st_ho["judge_applicable"]
            micro_ho["fires"] += st_ho["predicate_fires"]
            micro_ho["both"] += st_ho["both"]
        per_card[c.id] = st
        micro["judged"] += st["judged_pairs"]
        micro["applicable"] += st["judge_applicable"]
        micro["fires"] += st["predicate_fires"]
        micro["both"] += st["both"]

    # End-to-end: labels available when the verifier requires applicability (no offline judge).
    vreq = TranslationVerifier(require_applicability=True)
    labeled_pairs = 0
    decided_checks = 0
    sat_checks = 0
    abstain_pairs = 0
    by_card = Counter()
    for rec in pairs:
        subset = [c for c in cards]
        rep = vreq.verify(rec["src"], rec["tgt_ref"], subset)
        if not rep.decided:
            abstain_pairs += 1
            continue
        decided_checks += rep.decided
        sat = [r for r in rep.results if r.verdict == "satisfied"]
        sat_checks += len(sat)
        if sat:
            labeled_pairs += 1
            for r in sat:
                by_card[r.rule_id] += 1

    prec = micro["both"] / micro["fires"] if micro["fires"] else 0.0
    rec_ = micro["both"] / micro["applicable"] if micro["applicable"] else 0.0
    ho = None
    if split:
        hp = micro_ho["both"] / micro_ho["fires"] if micro_ho["fires"] else 0.0
        hr = micro_ho["both"] / micro_ho["applicable"] if micro_ho["applicable"] else 0.0
        ho = {
            "rows": len(pairs) - split,
            "judge_applicable_checks": micro_ho["applicable"],
            "predicate_fires": micro_ho["fires"],
            "agreement": micro_ho["both"],
            "precision": round(hp, 3),
            "recall": round(hr, 3),
        }
    out = {
        "cards": args.cards,
        "cards_with_predicates": len(per_card),
        "pairs_scanned": len(pairs),
        "predicate_micro_heldout": ho,
        "predicate_micro": {
            "judge_applicable_checks": micro["applicable"],
            "predicate_fires": micro["fires"],
            "agreement": micro["both"],
            "precision": round(prec, 3),
            "recall": round(rec_, 3),
        },
        "yield_with_required_applicability": {
            "pairs_with_any_decision": labeled_pairs,
            "satisfied_checks": sat_checks,
            "decided_checks": decided_checks,
            "pairs_abstaining_entirely": abstain_pairs,
            "top_label_cards": dict(by_card.most_common(6)),
        },
        "per_card": per_card,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "per_card"}, ensure_ascii=False, indent=2))
    print("\nper-card (predicates vs judge):")
    for cid, st in sorted(per_card.items(), key=lambda kv: -kv[1]["both"]):
        print(
            f"  {cid} judged={st['judged_pairs']:4d} judge_app={st['judge_applicable']:4d} "
            f"fires={st['predicate_fires']:4d} agree={st['both']:4d} "
            f"P={st['precision']:.2f} R={st['recall']:.2f} F1={st['f1']:.2f}  {st['statement']}"
        )


if __name__ == "__main__":
    main()
