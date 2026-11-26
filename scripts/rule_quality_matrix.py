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
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grammetarl.schema import MBGCard
from grammetarl.validator import TranslationVerifier
from scripts.verifier_discrimination import (
    corrupt,
    corrupt_all,
    corrupt_token,
    inject_forbidden,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="agreement x discrimination per rule")
    ap.add_argument("--cards", default="data/rules/mbg_cards_recompiled_manual.jsonl")
    ap.add_argument("--pairs", default="data/external/pairs_en_lb.jsonl")
    ap.add_argument("--source-filter", default="techgov_lu_en")
    ap.add_argument("--reliability", default="data/rules/reliability_predicate_gated_techgov.json")
    ap.add_argument("--min-weight", type=float, default=0.05)
    ap.add_argument("--samples-per-rule", type=int, default=120)
    ap.add_argument("--skip", type=int, default=0, help="Skip the first N candidate pairs (measure on the held-out half)")
    ap.add_argument("--min-support", type=int, default=30,
                    help="Decided checks required before a rule may be called usable (a single "
                         "lucky pair must not satisfy the quality gate)")
    ap.add_argument("--min-tested", type=int, default=20,
                    help="Perturbations that must have been attempted before discrimination counts")
    ap.add_argument("--perturbations", type=int, default=3)
    ap.add_argument("--out", default="artifacts/audit/rule_quality_matrix.json")
    args = ap.parse_args()

    cards = [MBGCard.model_validate_json(l) for l in Path(args.cards).open("r", encoding="utf-8") if l.strip()]
    rel = json.loads(Path(args.reliability).read_text("utf-8"))["rules"]
    def has_source_conditional(c: MBGCard) -> bool:
        return any(x.kind == "source_conditional" for x in c.output_constraints)

    # A source-conditional rule is gated by its own source condition (measured precision 0.50-0.57),
    # so it must not be filtered out for lacking a target-side predicate (measured 0.04-0.43).
    cand = [c for c in cards
            if rel.get(c.id, {}).get("weight", 0) >= args.min_weight
            and (any(p.accepted and p.kind == "regex" and p.pattern for p in c.applicability)
                 or has_source_conditional(c))]
    print(f"candidate rules (weight >= {args.min_weight}, predicate present): {len(cand)}")

    vpred = TranslationVerifier(require_applicability=True)   # predicate-gated: find satisfied checks
    vcons = TranslationVerifier(require_applicability=False)  # constraint under test

    skipped = 0
    pairs = []
    with Path(args.pairs).open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if args.source_filter and rec.get("source") != args.source_filter:
                continue
            if skipped < args.skip:
                skipped += 1
                continue
            pairs.append(rec)

    satisfied_by_rule: dict[str, list[dict]] = collections.defaultdict(list)
    # Agreement is measured on the SAME pairs discrimination is measured on; otherwise a --skip run
    # silently re-prints the fit-half numbers from the reliability table and calls them held-out.
    seen: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for rec in pairs:
        rep = vpred.verify(rec["src"], rec["tgt_ref"], cand)
        for r in rep.results:
            if r.verdict not in ("satisfied", "violated"):
                continue
            seen[r.rule_id][r.verdict] += 1
            if r.verdict == "satisfied" and len(satisfied_by_rule[r.rule_id]) < args.samples_per_rule:
                satisfied_by_rule[r.rule_id].append(rec)

    def pattern_literals(pats: list[str]) -> set[str]:
        """Literal surface material inside a pattern, anchors/quantifiers/escapes removed."""
        out_set: set[str] = set()
        for pat in pats:
            if not pat:
                continue
            cleaned = re.sub(r"\\[bBwWsSdD]|\(\?:|\)|\(|\^|\$|[|*+?{}\[\]]", " ", pat)
            for tok in re.split(r"[^0-9A-Za-zÀ-ÿ'’]+", cleaned):
                if len(tok) >= 2 and not tok.isspace():
                    out_set.add(tok.lower())
        return out_set

    def restates_constraint(card: MBGCard) -> tuple[bool, str]:
        """True when the applicability predicate and the output constraint are the same test.

        Such a rule cannot verify a translation: it fires exactly when its own constraint is
        satisfied, so the "check" is a restatement of the trigger. Discrimination measured against
        perturbations of that same span looks good while the rule verifies nothing about the source.
        """
        pred_lits: set[str] = set()
        for pr in card.applicability:
            if pr.accepted and pr.pattern:
                pred_lits |= pattern_literals([pr.pattern])
        cons_lits: set[str] = set()
        for c in card.output_constraints:
            pats = ([c.value] if c.value else []) + list(c.values or [])
            cons_lits |= pattern_literals(pats)
        if pred_lits and cons_lits and pred_lits == cons_lits:
            return True, f"identical literals {sorted(pred_lits)}"
        if pred_lits and cons_lits and pred_lits <= cons_lits and len(pred_lits) >= 1 and len(cons_lits - pred_lits) == 0:
            return True, f"predicate subset {sorted(pred_lits)}"
        return False, ""

    out = {}
    for card in cand:
        pats, token_level, negated_pats = [], False, []
        for c in card.output_constraints:
            # source_conditional: perturb the CONSEQUENCE side (target pattern), which is the claim the
            # rule makes; the condition side is what makes the rule govern the pair at all.
            if c.kind == "must_include_regex" and c.value:
                pats.append(c.value)
                token_level = token_level or c.level == "token"
            elif c.kind == "source_conditional" and c.negate:
                negated_pats.extend([x for x in (c.values or []) if x])
            elif c.kind in ("must_include_any_regex", "source_conditional"):
                pats.extend([x for x in (c.values or []) if x])
                token_level = token_level or c.level == "token"
        stats = collections.Counter()
        escaped = []
        for i, rec in enumerate(satisfied_by_rule[card.id]):
            for k in range(args.perturbations):
                if negated_pats:
                    # a prohibition is discriminated by inserting the forbidden shape
                    bad, how = inject_forbidden(vcons, rec["tgt_ref"], negated_pats, i + k)
                else:
                    multi, mhow = corrupt_all(vcons, rec["tgt_ref"], pats, i + k)
                    single, show = (corrupt_token(vcons, rec["tgt_ref"], pats, i + k) if token_level
                                    else corrupt(vcons, rec["tgt_ref"], pats, i + k))
                    # prefer the multi-instance corruption when it exists: it is the honest test for a
                    # per-occurrence rule; the single-token number stays in `how` for comparison
                    if multi is not None:
                        bad, how = multi, f"all-instances: {mhow}"
                        stats["multi_instance"] += 1
                    else:
                        bad, how = single, f"single-token: {show}"
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
        redundant, why_redundant = restates_constraint(card)
        n_meas = sum(seen[card.id].values())
        # Prefer the agreement measured on THESE very pairs; the reliability-table value is a
        # cross-check (and, for a --skip run, a fit-half number that must not be mistaken for held-out).
        agree_meas = (seen[card.id]["satisfied"] / n_meas) if n_meas else None
        agree_used = agree_meas if agree_meas is not None else r["agreement"]
        out[card.id] = {
            "statement": card.statement[:80],
            "constraint_kinds": [c.kind for c in card.output_constraints],
            "token_level": token_level,
            "agreement": r["agreement"],
            "n_agreement": r["applicable"],
            "agreement_measured": round(agree_meas, 4) if agree_meas is not None else None,
            "n_measured": n_meas,
            "weight": r["weight"],
            "perturbations_tested": tested,
            "discrimination": round(disc, 3),
            "escaped_examples": escaped,
            "predicate_restates_constraint": redundant,
            "redundancy_note": why_redundant,
            # three-sided: agrees with correct translations, bites on broken ones, and is not just
            # its own trigger restated.
            "usable": bool(agree_used >= 0.65 and disc >= 0.5 and not redundant
                           and n_meas >= args.min_support and tested >= args.min_tested),
            # quality passes but the sample is too thin to certify: report it, do not bank it
            "provisional": bool(agree_used >= 0.65 and disc >= 0.5 and not redundant
                                and (n_meas < args.min_support or tested < args.min_tested)),
        }
        if redundant:
            flag = "TAUTOLOGICAL (predicate == constraint)"
        elif disc < 0.2:
            flag = "VACUOUS"
        elif agree_used < 0.65:
            flag = "weak (agreement)"
        elif disc < 0.5:
            # the printed flag must use the SAME predicate as `usable`, or a row reads as USABLE while
            # being excluded from the usable list (found: discrimination 42% printed as USABLE, and a
            # rule with n=1 printed as USABLE while the support floor excluded it)
            flag = "weak (discrimination)"
        elif n_meas < args.min_support or tested < args.min_tested:
            flag = "provisional (support too thin)"
        else:
            flag = "USABLE"
        meas = f"{agree_used:.2f}/n={n_meas:4d}" if agree_meas is not None else "(no pairs measured)"
        print(f"  {card.id} agree={meas} [table {r['agreement']:.2f}/n={r['applicable']:4d}] "
              f"discrimination={disc:6.1%} (tested={tested:4d}) -> {flag}  {card.statement[:44]}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), "utf-8")
    usable = [k for k, v in out.items() if v["usable"]]
    vacuous = [k for k, v in out.items() if v["discrimination"] < 0.2]
    print(f"usable (agree>=0.65 AND discrimination>=0.5): {len(usable)} {usable}")
    print(f"vacuous (discrimination<0.2): {len(vacuous)} {vacuous}")
    prov = [k for k, v in out.items() if v.get("provisional")]
    print(f"provisional (quality ok, support < {args.min_support} checks or < {args.min_tested} "
          f"perturbations): {len(prov)} {prov}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
