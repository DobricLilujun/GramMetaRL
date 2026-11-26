#!/usr/bin/env python3
"""Decompose a rule's VIOLATIONS into structural buckets.

A rule that disagrees with professional translations 30% of the time is only usable if we know WHY it
disagrees: un-encoded alternative constructions (fixable), paraphrase/NP restructuring (needs phrase
structure, P3), or genuine form errors (the signal we want). Buckets here are deliberately surface
patterns with no interpretation attached, because interpretation caused a false finding: Luxembourgish
capitalizes every noun, so "article + capitalized token" is NOT a proper-name test (it matched
`den Attentater` = "the attackers").
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
from grammetarl.validator import TranslationVerifier


def main() -> None:
    ap = argparse.ArgumentParser(description="structural decomposition of a rule's violations")
    ap.add_argument("--cards", default="data/rules/mbg_cards_condonly.jsonl")
    ap.add_argument("--rule", required=True)
    ap.add_argument("--pairs", default="data/external/pairs_en_lb.jsonl")
    ap.add_argument("--source-filter", default="techgov_lu_en")
    ap.add_argument("--rows", type=int, default=0)
    ap.add_argument("--bucket", action="append", default=[],
                    help="NAME=REGEX over the TARGET; first match wins, residual bucket is 'residual'")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cards = [MBGCard.model_validate_json(l) for l in Path(args.cards).open("r", encoding="utf-8") if l.strip()]
    by_id = {c.id: c for c in cards}
    if args.rule not in by_id:
        raise SystemExit(f"unknown rule {args.rule}")
    card = by_id[args.rule]
    v = TranslationVerifier(require_applicability=True)

    buckets: list[tuple[str, re.Pattern[str]]] = []
    for spec in args.bucket:
        name, _, pat = spec.partition("=")
        try:
            buckets.append((name, re.compile(pat, re.I)))
        except re.error as exc:
            raise SystemExit(f"bucket {name!r}: invalid regex: {exc}")

    cnt: collections.Counter = collections.Counter()
    ex: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
    n = 0
    with Path(args.pairs).open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if args.source_filter and rec.get("source") != args.source_filter:
                continue
            n += 1
            if args.rows and n > args.rows:
                break
            rep = v.verify(rec["src"], rec["tgt_ref"], [card])
            res = rep.results[0]
            if res.verdict == "satisfied":
                cnt["satisfied"] += 1
                continue
            if res.verdict != "violated":
                cnt[f"undecided:{res.verdict}"] += 1
                continue
            name = "residual"
            for bname, rx in buckets:
                if rx.search(rec["tgt_ref"]):
                    name = bname
                    break
            cnt[name] += 1
            if len(ex[name]) < 4:
                ex[name].append((rec["src"][:78], rec["tgt_ref"][:78]))

    viol = sum(c for k, c in cnt.items() if k not in ("satisfied",) and not k.startswith("undecided"))
    print(f"rule={args.rule} pairs_scanned={n} satisfied={cnt['satisfied']} violated={viol}")
    print(f"{'bucket':38} {'n':>6}  share_of_violations")
    for k, c in cnt.most_common():
        share = (c / viol) if (viol and k != "satisfied" and not k.startswith("undecided")) else 0.0
        print(f"  {k:36} {c:6}  {share:6.1%}")
    for k, rows in ex.items():
        print(f"\n[{k}]")
        for a, b in rows[:3]:
            print(f"  EN {a}\n  LB {b}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"rule": args.rule, "counts": dict(cnt),
                                              "violations": viol, "pairs_scanned": n},
                                             ensure_ascii=False, indent=2), "utf-8")
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
