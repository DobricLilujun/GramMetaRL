#!/usr/bin/env python3
"""Emit cards for corpus-mined source conditionals.

Provenance is the point: these rules were NOT extracted from the grammar book. They were screened
against professional translations (scripts/screen_source_conditionals.py) and certified on the held-out
half. They therefore carry source.source_id = CORPUS_MINED_TECHGOV and a note naming the license and the
measured agreement, and they live in their own card file so a reader can tell book-derived rules from
corpus-derived ones.

Examples are taken from the FIT half of the pool (never the held-out half used for certification), so the
card's self-test does not leak the certification measurement.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grammetarl.schema import MBGCard
from grammetarl.validator import TranslationVerifier


def main() -> None:
    ap = argparse.ArgumentParser(description="Emit mined-conditional cards")
    ap.add_argument("--specs", default="data/rules/screened_conditionals.json")
    ap.add_argument("--fit", default="artifacts/audit/screen_conditionals_fit.json")
    ap.add_argument("--heldout", default="artifacts/audit/screen_conditionals_heldout.json")
    ap.add_argument("--pairs", default="data/external/pairs_en_lb.jsonl")
    ap.add_argument("--source-filter", default="techgov_lu_en")
    ap.add_argument("--fit-rows", type=int, default=4000, help="pairs used for examples (must match --skip of the fit run)")
    ap.add_argument("--min-agreement", type=float, default=0.65)
    ap.add_argument("--min-support", type=int, default=30)
    ap.add_argument("--out", default="data/rules/mbg_cards_mined.jsonl")
    args = ap.parse_args()

    specs = {c["name"]: c for c in json.loads(Path(args.specs).read_text("utf-8"))["candidates"]}
    fit = json.loads(Path(args.fit).read_text("utf-8"))["candidates"]
    hold = json.loads(Path(args.heldout).read_text("utf-8"))["candidates"]

    # both halves must clear the floor: selection on the fit half alone over-fits (measured:
    # 'also_too' 0.74 fit -> 0.64 held-out and is refuted there)
    keep = []
    for name, f in fit.items():
        h = hold.get(name, {})
        ok = (f["agreement"] or 0) >= args.min_agreement and (f["applicable"] or 0) >= args.min_support \
            and (h.get("agreement") or 0) >= args.min_agreement and (h.get("applicable") or 0) >= args.min_support
        if ok:
            keep.append(name)
    print(f"certified on BOTH halves: {len(keep)} {sorted(keep)}")

    v = TranslationVerifier()
    ex: dict[str, list[dict]] = {n: [] for n in keep}
    scanned = 0
    for line in Path(args.pairs).open("r", encoding="utf-8"):
        rec = json.loads(line)
        if rec.get("source") != args.source_filter:
            continue
        if scanned >= args.fit_rows:
            break
        scanned += 1
        s, t = v.normalize_text(rec["src"]), v.normalize_text(rec["tgt_ref"])
        for n in keep:
            if len(ex[n]) >= 2:
                continue
            sp = specs[n]
            if not re.search(sp["source_pattern"], s):
                continue
            if not any(re.search(x, t) for x in sp["values"]):
                continue
            ex[n].append(rec)

    cards = []
    for i, n in enumerate(sorted(keep), start=1):
        sp, f, h = specs[n], fit[n], hold[n]
        stmt = (f"English {n} is realised by an overt target cue "
                f"(measured: {f['applicable']} applicable pairs, agreement {f['agreement']} on the fit half, "
                f"{h['agreement']} on the held-out half; discrimination "
                f"{(h.get('discrimination') or 0)*100:.0f}%)")
        card = MBGCard.model_validate({
            "id": f"LB_MIN{i:02d}_{n}",
            "language": "lb",
            "statement": stmt,
            "rule_status": "variable",   # no 'statistical' in the schema; 'variable' = measured, not categorical
            "operation_type": "negate" if n == "negation_not" else "insert",
            "scope": f"the target clause realising the English {n} construction",
            "confidence": round(min(f["agreement"], h["agreement"]), 3),
            "phenomenon_tags": ["function_words", "mined_from_corpus"],
            "trigger_conditions": [f"source matches {sp['source_pattern']}"],
            "applicability": [],
            "output_constraints": [{
                "kind": "source_conditional", "level": "sentence",
                "source_pattern": sp["source_pattern"], "values": sp["values"],
                "description": f"corpus-mined cue check for {n}; applicability is source-derived"}],
            "exceptions_and_variants": [{"type": "register_variant",
                                         "description": "divergent restructuring of the clause is scored as a violation; "
                                                        "this is why six sibling candidates (superlative, ability_can, "
                                                        "reflexive_self, while_durative, as_as, both) were refuted",
                                         "condition": "the professional translation does not keep the construction"}],
            "examples": [{"source_sentence": r["src"], "target_sentence": r["tgt_ref"],
                          "note": f"corpus instance {r['id']} ({r.get('license')}); fit half, used for the card self-test only"}
                         for r in ex[n]],
            "source": {"source_id": "CORPUS_MINED_TECHGOV", "page_start": 0, "page_end": 0,
                       "section_title": "screened and certified by scripts/screen_source_conditionals.py",
                       "excerpt": "no book evidence: derived from corpus agreement, not extracted"},
        })
        cards.append(card)
        print(f"  {card.id:28} examples={len(card.examples)} conf={card.confidence}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out).open("w", encoding="utf-8") as f:
        for c in cards:
            f.write(c.model_dump_json() + "\n")
    print(f"-> {args.out}  ({len(cards)} cards)")


if __name__ == "__main__":
    main()
