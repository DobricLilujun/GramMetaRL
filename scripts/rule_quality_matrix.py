#!/usr/bin/env python3
"""Rule quality matrix: agreement on correct translations x discrimination on broken ones.

Agreement alone is not enough. A constraint that can never fail scores agreement 1.00 and 0% false
penalties while contributing a CONSTANT reward term - measured directly: the -en plural rule caught
6.4% of token-level perturbations designed to break it (63/978), because "some token ends in -en"
holds for almost any sentence. A reward term that cannot fail cannot shape a policy.

For every candidate rule:
  agreement      = fraction of predicate-gated checks that agree with a professional human
                   translation (satisfied / decided)
  discrimination = fraction of minimal perturbations of a satisfied check that the rule flags as
                   violated
Usable supervision needs both high. Anything with low discrimination is reported as vacuous rather
than promoted for its 0% false-penalty rate.
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
from scripts.verifier_discrimination import corrupt, corrupt_token


def main() -> None:
    ap = argparse.ArgumentParser(description="agreement x discrimination per rule")
    ap.add_argument("--cards", default="data/rules/mbg_cards_recompiled_manual.jsonl")
    ap.add_argument("--pairs", default="data/external/pairs_en_lb.jsonl")
    ap.add_argument("--source-filter", default="techgov_lu_en")
    ap.add_argument("--reliability", default="data/rules/reliability_predicate_gated_techgov.json")
    ap.add_argument("--min-weight", type=float, default=0.05)
    ap.add_argument("--samples-per-rule", type=int, default=120)
    ap.add_argument("--perturbations", type=int, default=3)
    ap.add_argument("--out", default="artifacts/audit/rule_quality_matrix.json")
    args = ap.parse_args()

    cards = [MBGCard.model_validate_json(l) for l in Path(args.cards).open("r", encoding="utf-8") if l.strip()]
    rel = json.loads(Path(args.reliability).read_text("utf-8"))["rules"]
    cand = [c for c in cards if rel.get(c.id, {}).get("weight", 0) >= args.min_weight
            and any(p.accepted and p.kind == "regex" and p.pattern for p in c.applicability)]
    print(f"candidate rules (weight >= {args.min_weight}, predicate present): {len(cand)}")

    vpred = TranslationVerifier(require_applicability=True)   # predicate-gated: find satisfied checks
    vcons = TranslationVerifier(require_applicability=False)  # constraint under test

    pairs = []
    with Path(args.pairs).open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if args.source_filter and rec.get("source") != args.source_filter:
                continue
            pairs.append(rec)

    satisfied_by_rule: dict[str, list[dict]] = collections.defaultdict(list)
    for rec in pairs:
        need = [c for c in cand if len(satisfied_by_rule[c.id]) < args.samples_per_rule]
        if not need:
            break
        rep = vpred.verify(rec["src"], rec["tgt_ref"], need)
        for r in rep.results:
            if r.verdict == "satisfied" and len(satisfied_by_rule[r.rule_id]) < args.samples_per_rule:
                satisfied_by_rule[r.rule_id].append(rec)

    out = {}
    for card in cand:
        pats, token_level = [], False
        for c in card.output_constraints:
            if c.kind == "must_include_regex" and c.value:
                pats.append(c.value)
                token_level = token_level or c.level == "token"
            elif c.kind == "must_include_any_regex":
                pats.extend([x for x in (c.values or []) if x])
                token_level = token_level or c.level == "token"
        stats = collections.Counter()
        escaped = []
        for i, rec in enumerate(satisfied_by_rule[card.id]):
            for k in range(args.perturbations):
                bad, how = (corrupt_token(vcons, rec["tgt_ref"], pats, i + k) if token_level
                            else corrupt(vcons, rec["tgt_ref"], pats, i + k))
                if bad is None:
                    stats["unperturbable"] += 1
                    continue
                rep = vcons.verify(rec["src"], bad, [card])
                verdict = rep.results[0].verdict
                stats["tested"] += 1
                stats[verdict] += 1
                if verdict != "violated" and len(escaped) < 2:
                    escaped.append({"how": how, "orig": rec["tgt_ref"][:90], "bad": bad[:90], "verdict": verdict})
        tested = stats["tested"]
        disc = stats["violated"] / tested if tested else 0.0
        r = rel[card.id]
        out[card.id] = {
            "statement": card.statement[:80],
            "constraint_kinds": [c.kind for c in card.output_constraints],
            "token_level": token_level,
            "agreement": r["agreement"],
            "n_agreement": r["applicable"],
            "weight": r["weight"],
            "perturbations_tested": tested,
            "discrimination": round(disc, 3),
            "escaped_examples": escaped,
            "usable": bool(r["agreement"] >= 0.65 and disc >= 0.5),
        }
        flag = "USABLE" if out[card.id]["usable"] else ("VACUOUS" if disc < 0.2 else "weak")
        print(f"  {card.id} agree={r['agreement']:.2f} (n={r['applicable']:4d}) "
              f"discrimination={disc:6.1%} (tested={tested:4d}) -> {flag}  {card.statement[:52]}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), "utf-8")
    usable = [k for k, v in out.items() if v["usable"]]
    vacuous = [k for k, v in out.items() if v["discrimination"] < 0.2]
    print(f"usable (agree>=0.65 AND discrimination>=0.5): {len(usable)} {usable}")
    print(f"vacuous (discrimination<0.2): {len(vacuous)} {vacuous}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
