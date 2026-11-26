#!/usr/bin/env python3
"""Turn the cached constraint audit into a per-rule reliability table for the reward.

The audit already measures, per card, how often its machine check agrees with professional human
translations on judge-applicable pairs. This makes that measurable prior available to training:
`grammar_reward_weighted` then weights each verdict by how much that rule has earned it, instead of
treating every compiled constraint as an oracle. Re-measuring is free (the judge cache is replayed
with zero model calls), so the table can be regenerated whenever the card set changes.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json

from grammetarl.rewards.reliability import DEFAULT_SHRINKAGE, ReliabilityTable, agreement_probability


def main() -> None:
    ap = argparse.ArgumentParser(description="Build per-rule reliability weights from the audit")
    ap.add_argument("--audit", default="artifacts/audit/precision_mbg_cards_triaged.json")
    ap.add_argument("--out", default="data/rules/reliability.json")
    ap.add_argument("--shrinkage", type=float, default=DEFAULT_SHRINKAGE)
    args = ap.parse_args()

    audit = json.loads(Path(args.audit).read_text("utf-8"))
    per_card = audit["per_card"]
    rules: dict[str, dict] = {}
    for cid, st in per_card.items():
        applicable = int(st.get("applicable", 0))
        violated = int(st.get("violated", 0))
        satisfied = int(st.get("satisfied", max(0, applicable - violated)))
        agree = agreement_probability(satisfied, applicable, args.shrinkage)
        weight = max(0.0, min(1.0, 2.0 * agree - 1.0))
        rules[cid] = {
            "applicable": applicable,
            "satisfied": satisfied,
            "violated": violated,
            "raw_agreement": (satisfied / applicable) if applicable else None,
            "agreement": round(agree, 4),
            "weight": round(weight, 4),
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_audit": args.audit,
        "shrinkage": args.shrinkage,
        "n_rules_measured": len(rules),
        "n_rules_weight_gt_0": sum(1 for r in rules.values() if r["weight"] > 0),
        "total_applicable_checks": sum(r["applicable"] for r in rules.values()),
        "total_decided_checks": sum(r["applicable"] for r in rules.values()),
        "rules": rules,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")

    t = ReliabilityTable.load(out, args.shrinkage)
    print(f"rules={len(t)} with_weight>0={payload['n_rules_weight_gt_0']} -> {out}")
    top = sorted(rules.items(), key=lambda kv: -kv[1]["weight"])[:10]
    for cid, r in top:
        raw = f"{r['raw_agreement']:.2f}" if r["raw_agreement"] is not None else "n/a"
        print(f"  {cid} n={r['applicable']:4d} raw_agree={raw} shrunk={r['agreement']:.2f} weight={r['weight']:.2f}")
    zero = [(cid, r) for cid, r in rules.items() if r["weight"] == 0.0]
    print(f"  weight=0 rules (coin-flip or unmeasured): {len(zero)}")


if __name__ == "__main__":
    main()
