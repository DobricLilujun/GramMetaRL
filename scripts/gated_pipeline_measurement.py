#!/usr/bin/env python3
"""End-to-end measurement of the QUALITY-GATED verifier: labels and false penalties.

Every component so far is measured separately (constraint agreement, predicate precision/recall).
This script composes them the way training would use them, and reports only what survives:

  a card may emit a verdict for a pair only if
    (a) it has an applicability predicate that was validated at >= --min-precision against the judge
        decisions (held-out half when available), AND
    (b) its machine constraints have reliability weight > 0 (measured agreement with human
        translations, support-shrunk).

Then, over the judged pairs:
  - label yield  : pairs with >= 1 satisfied check (usable dense grammar supervision)
  - FP rate      : pairs whose grammar reward is negative although the translation is professional,
                   reported unweighted AND reliability-weighted.

Without gating, "yield" is inflated by predicates that fire on nearly every sentence; that inflation
is exactly what this script refuses to count.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grammetarl.rewards.grammar_reward import grammar_reward_verdicts, grammar_reward_weighted
from grammetarl.rewards.reliability import ReliabilityTable
from grammetarl.schema import MBGCard
from grammetarl.validator import TranslationVerifier, has_source_conditional
from scripts.label_with_judge import key


def main() -> None:
    ap = argparse.ArgumentParser(description="Measure the gated verifier end to end")
    ap.add_argument("--cards", default="data/rules/mbg_cards_recompiled_manual.jsonl")
    ap.add_argument("--reliability", default="data/rules/reliability_recompiled.json")
    ap.add_argument("--trigger-quality", default="artifacts/audit/trigger_quality_manual.json")
    ap.add_argument("--pairs", default="data/external/pairs_en_lb.jsonl")
    ap.add_argument("--cache", default="data/external/judge_cache.jsonl")
    ap.add_argument("--rows", type=int, default=1200, help="0 = every pair in the pool")
    ap.add_argument("--source-filter", default=None,
                    help="Only score pairs from this source id. FP rates are only meaningful for "
                    "sources whose reference is a professional human translation (techgov_lu_en); "
                    "LuxAlign pairs are semantically similar by its own dataset card.")
    ap.add_argument("--labeled-out", default="data/processed/labeled_gated.jsonl")
    ap.add_argument("--min-precision", type=float, default=0.5,
                    help="Predicate precision vs the judge (held-out number when available)")
    ap.add_argument("--quality", default="",
                    help="artifacts/audit/rule_quality_matrix*.json: only rules certified usable "
                         "(agreement AND discrimination AND not self-restating) may label pairs")
    ap.add_argument("--allow-provisional", action="store_true",
                    help="also admit rules that pass quality but fall below the support floor")
    ap.add_argument("--min-agreement", type=float, default=0.0,
                    help="Also require the card's constraint agreement (support-shrunk) under the SAME "
                    "applicability gating the pipeline uses. Without it a card with a good predicate but "
                    "a coin-flip constraint still ships - which is exactly what inflated the earlier "
                    "false-penalty estimate.")
    ap.add_argument("--out", default="artifacts/audit/gated_pipeline.json")
    args = ap.parse_args()

    cards = [MBGCard.model_validate_json(l) for l in Path(args.cards).open("r", encoding="utf-8") if l.strip()]
    subset = cards
    table = ReliabilityTable.load(args.reliability)
    tq = json.loads(Path(args.trigger_quality).read_text("utf-8"))

    ok_pred = set()
    for cid, st in tq["per_card"].items():
        ho = st.get("heldout") or {}
        prec = ho.get("precision", st["precision"])  # prefer the held-out number
        if st["judged_pairs"] > 0 and prec >= args.min_precision:
            ok_pred.add(cid)
    ok_rel = {cid for cid in {c.id for c in cards}
              if table.weight(cid) > 0.0 and table.agreement(cid) >= args.min_agreement}
    # A source-conditional rule is gated by its own condition (the source pattern), so it does not
    # need a separately validated target-side predicate - the axis every compiled predicate failed.
    allowed = {c.id for c in cards if c.id in ok_rel and (c.id in ok_pred or has_source_conditional(c))}

    # A rule whose constraint cannot fail (discrimination ~0) contributes a CONSTANT reward: it can
    # never penalize, so "0% false penalties" over such a set is a measurement artifact, not quality.
    # --quality ties the label gate to the three-axis matrix so the same rules are certified once.
    if args.quality:
        qm = json.loads(Path(args.quality).read_text("utf-8"))
        keep = {k for k, v in qm.items() if v.get("usable")
                or (args.allow_provisional and v.get("provisional"))}
        dropped = sorted(allowed - keep)
        allowed &= keep
        print(f"quality-gated: dropped {len(dropped)} {dropped} -> allowed {len(allowed)}")
    subset = [c for c in cards if c.id in allowed]

    cache: dict[str, bool] = {}
    for line in Path(args.cache).open("r", encoding="utf-8"):
        r = json.loads(line)
        cache.setdefault(r["key"], bool(r["applicable"]))
    pairs = []
    with Path(args.pairs).open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if args.source_filter and rec.get("source") != args.source_filter:
                continue
            pairs.append(rec)
            if args.rows and len(pairs) >= args.rows:
                break

    v = TranslationVerifier(require_applicability=True)
    stats = Counter()
    rewards = {"unweighted": [], "weighted": []}
    labels = []
    for rec in pairs:
        src, tgt = rec["src"], rec["tgt_ref"]
        if not subset:
            break
        rep = v.verify(src, tgt, subset)
        if not rep.decided:
            stats["pairs_no_decision"] += 1
            continue
        verdicts = rep.verdicts()
        rule_ids = rep.rule_ids()
        ru = grammar_reward_verdicts(verdicts := [r.verdict for r in rep.results])
        rw = grammar_reward_weighted(verdicts, table.weights(rep.rule_ids()))
        stats["pairs_scored"] += 1
        stats["decided_checks"] += rep.decided
        stats["false_penalty_unweighted"] += ru < -1e-9
        stats["false_penalty_weighted"] += rw < -1e-9
        sat = sorted({r.rule_id for r in rep.results if r.verdict == "satisfied"})
        if sat:
            stats["labeled_pairs"] += 1
            stats["label_checks"] += sum(1 for r in rep.results if r.verdict == "satisfied")
            tags = sorted({t for c in subset if c.id in set(sat) for t in c.phenomenon_tags})
            pid = hashlib.sha1(f"{rec['src']}|{rec['tgt_ref']}".encode("utf-8")).hexdigest()[:16]
            labels.append(
                {
                    "id": f"gated_{pid}",
                    "src": rec["src"],
                    "tgt_ref": rec["tgt_ref"],
                    "lang_pair": "en->lb",
                    "difficulty": "gated",
                    "phenomena_tags": tags,
                    "gold_rules": sat,
                    "gold_lexicon": [],
                    "source": rec.get("source", "external"),
                    "license": rec.get("license", "unknown"),
                }
            )

    n = stats["pairs_scored"] or 1
    by_pair: dict[tuple, dict] = {}
    for r in labels:
        k = (r["src"], r["tgt_ref"])
        row = by_pair.setdefault(k, {**r, "gold_rules": []})
        row["gold_rules"] = sorted(set(row["gold_rules"] + r["gold_rules"]))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "cards_total": len(cards),
        "cards_predicate_validated": len(ok_pred),
        "cards_reliability_positive": len(ok_rel),
        "cards_allowed": len(allowed),
        "allowed_card_ids": sorted(allowed),
        "pairs_scanned": len(pairs),
        "pairs_no_decision": stats["pairs_no_decision"],
        "pairs_scored": stats["pairs_scored"],
        "labeled_pairs": len(by_pair),
        "distinct_rules_in_labels": len({g for row in by_pair.values() for g in row["gold_rules"]}),
        "decided_checks": stats["decided_checks"],
        "false_penalty_rate_unweighted": stats["false_penalty_unweighted"] / n if (n := stats["pairs_scored"] or 1) else 0.0,
        "false_penalty_rate_weighted": stats["false_penalty_weighted"] / n,
    }
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), "utf-8")
    lbl_path = Path(args.labeled_out)
    with lbl_path.open("w", encoding="utf-8") as f:
        for row in by_pair.values():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"labeled pairs -> {lbl_path}")


if __name__ == "__main__":
    main()
