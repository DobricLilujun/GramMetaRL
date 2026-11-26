#!/usr/bin/env python3
"""Does reliability weighting reduce false penalties on professional human translations?

A professional human translation must not be punished by the grammar term. Any pair whose grammar
reward is negative is by construction a false penalty, so the fraction of such pairs is a direct
measure of reward quality - no LLM needed, the judge cache is replayed.

Compares the unweighted mean over verdicts (what a naive card-to-reward compiler gives) with the
reliability-weighted mean, where each rule counts only as much as its measured agreement with human
translations.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from collections import Counter
from pathlib import Path

from grammetarl.rewards.grammar_reward import grammar_reward_verdicts, grammar_reward_weighted
from grammetarl.rewards.reliability import ReliabilityTable
from grammetarl.schema import MBGCard
from grammetarl.validator import TranslationVerifier
from scripts.label_with_judge import key  # identical key derivation, so the cache is reusable


def main() -> None:
    ap = argparse.ArgumentParser(description="False-penalty rate: unweighted vs reliability-weighted")
    ap.add_argument("--cards", default="data/rules/mbg_cards_recompiled.jsonl")
    ap.add_argument("--reliability", default="data/rules/reliability.json")
    ap.add_argument("--cache", default="data/external/judge_cache.jsonl")
    ap.add_argument("--pairs", default="data/external/pairs_en_lb.jsonl")
    ap.add_argument("--rows", type=int, default=1200)
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--require-applicability", action="store_true",
                    help="Use compiled trigger predicates instead of the offline judge")
    ap.add_argument("--out", default="artifacts/audit/reward_sanity.json")
    args = ap.parse_args()

    cards = [MBGCard.model_validate_json(l) for l in Path(args.cards).open("r", encoding="utf-8") if l.strip()]
    cards_by_id = {c.id: c for c in cards}
    table = ReliabilityTable.load(args.reliability)

    # The judge cache stores only sha1(src|tgt|card_id) -> bool, so rebuild the keys locally.
    cache: dict[str, bool] = {}
    with Path(args.cache).open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            cache.setdefault(rec["key"], bool(rec["applicable"]))

    def judgements(src: str, tgt: str) -> tuple[bool, set[str]]:
        """(was this pair judged at all, which cards were judged applicable)"""
        judged_any = False
        applicable: set[str] = set()
        for cid in cards_by_id:
            k = key(src, tgt, cid)
            if k in cache:
                judged_any = True
                if cache[k]:
                    applicable.add(cid)
        return judged_any, applicable

    pairs = []
    with Path(args.pairs).open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            pairs.append(r)
            if len(pairs) >= args.rows:
                break

    v = TranslationVerifier(require_applicability=args.require_applicability)
    stats = Counter()
    rew = {"unweighted": [], "weighted": []}
    flips = Counter()

    for rec in pairs:
        src, tgt = rec["src"], rec["tgt_ref"]
        judged_any, applicable = judgements(src, tgt)
        if not judged_any:
            stats["pairs_not_judged"] += 1
            continue
        if not applicable and not args.require_applicability:
            stats["pairs_no_rule_applicable"] += 1
            continue
        subset = [cards_by_id[c] for c in sorted(applicable)] if applicable else cards
        rep = v.verify(src, tgt, subset)
        verdicts = rep.verdicts()
        rule_ids = rep.rule_ids()
        if not rep.decided:
            stats["pairs_no_verdict"] += 1
            continue
        ru = grammar_reward_verdicts(verdicts)
        rw = grammar_reward_weighted(verdicts, table.weights(rule_ids))
        rew["unweighted"].append(ru)
        rew["weighted"].append(rw)
        stats["pairs_scored"] += 1
        stats["false_penalty_unweighted"] += ru < -1e-9
        stats["false_penalty_weighted"] += rw < -1e-9
        if ru < -1e-9 and rw >= -1e-9:
            flips["weighting_removed_penalty"] += 1
        elif rw < -1e-9 and ru >= -1e-9:
            flips["weighting_added_penalty"] += 1
        elif abs(ru - rw) > 1e-9:
            flips["magnitude_only"] += 1

    n = stats["pairs_scored"] or 1
    out = {
        "cards": args.cards,
        "reliability": args.reliability,
        "require_applicability": args.require_applicability,
        "pairs_scored": stats["pairs_scored"],
        "pairs_no_verdict": stats["pairs_no_verdict"],
        "false_penalty_rate_unweighted": stats["false_penalty_unweighted"] / n,
        "false_penalty_rate_weighted": stats["false_penalty_weighted"] / n,
        "mean_reward_unweighted": sum(rew["unweighted"]) / n,
        "mean_reward_weighted": sum(rew["weighted"]) / n,
        "sign_flips": dict(flips),
        "rules_with_weight_gt0": table.table and sum(1 for r in table.table.values() if r.get("weight", 0) > 0),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
