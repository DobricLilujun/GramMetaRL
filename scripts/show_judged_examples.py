#!/usr/bin/env python3
"""Print judge-labeled POSITIVE / NEGATIVE examples per card, for hand-authoring predicates."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.label_with_judge import key


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards", default="data/rules/mbg_cards_recompiled.jsonl")
    ap.add_argument("--pairs", default="data/external/pairs_en_lb.jsonl")
    ap.add_argument("--cache", default="data/external/judge_cache.jsonl")
    ap.add_argument("--rows", type=int, default=1200)
    ap.add_argument("--ids", default="")
    ap.add_argument("--pos", type=int, default=5)
    ap.add_argument("--neg", type=int, default=6)
    args = ap.parse_args()

    cards = {c["id"]: c for c in (json.loads(l) for l in Path(args.cards).open("r", encoding="utf-8") if l.strip())}
    cache: dict[str, bool] = {}
    for line in Path(args.cache).open("r", encoding="utf-8"):
        r = json.loads(line)
        cache.setdefault(r["key"], bool(r["applicable"]))
    pairs = [json.loads(l) for l in Path(args.pairs).open("r", encoding="utf-8")][: args.rows]

    ids = [i for i in args.ids.split(",") if i]
    if not ids:
        counts = {cid: sum(1 for p in pairs if cache.get(key(p["src"], p["tgt_ref"], cid))) for cid in cards}
        ids = [cid for cid, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:4]]

    for cid in ids:
        c = cards[cid]
        pos, neg = [], []
        for p in pairs:
            k = key(p["src"], p["tgt_ref"], cid)
            if k not in cache:
                continue
            (pos if cache[k] else neg).append(p)
        print("=" * 100)
        print(cid, f"POS={len(pos)} NEG={len(neg)}")
        print("STMT:", c["statement"][:160])
        print("TRIG:", [t if isinstance(t, str) else t.get("source_text") for t in c["trigger_conditions"]][:3])
        print("OWN_TGT:", [e["target_sentence"][:70] for e in c["examples"][:2]])
        print("OWN_SRC:", [e["source_sentence"][:60] for e in c["examples"][:2]])
        print(f"-- POS ({len(pos)}: judge says the rule governs this pair)")
        for p in pos[: args.pos]:
            print("   LB:", p["tgt_ref"][:110])
        print(f"-- NEG ({len(neg)}: judge says it does not)")
        for p in neg[: args.neg]:
            print("   LB:", p["tgt_ref"][:110])


if __name__ == "__main__":
    main()
