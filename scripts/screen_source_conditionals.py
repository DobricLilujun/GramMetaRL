#!/usr/bin/env python3
"""Screen candidate source-conditional constraints against professional translations.

Why this exists: four hand-authored conditionals derived from the book (present perfect, past
perfect, 2sg clitic, eng+cardinal) were REFUTED on the fit half (agreement 0.40 / 0.30 / 0.23 / 0.08)
even after form-list repair, while the two that certify (possessive articles, comparative 'méi') are
ones whose TARGET form is near-obligatory. That says the binding property is target obligatoriness,
not regex care. So instead of guessing from the book, screen many candidates and keep the survivors.

Unlike scripts/apply_source_conditionals.py, a candidate here is NOT attached to a card: screening asks
only (a) does the source condition fire, (b) does the target form hold on professional translations,
(c) can the check bite (perturbations). Provenance is a separate question - a survivor must acquire a
card (book-derived) or an explicit mined-rule label before it may ship in a card set.

Protocol mirrors the certification protocol: screen with --skip 0 (fit half), certify with --skip N
(held-out half). Verdict floors are the same as scripts/rule_quality_matrix.py.
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grammetarl.validator import TranslationVerifier

_spec = importlib.util.spec_from_file_location(
    "vd", Path(__file__).resolve().parent / "verifier_discrimination.py")
vd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vd)  # type: ignore[attr-defined]


def decide(tgt: str, values: list[str], negate: bool) -> str:
    hit = any(re.search(v, tgt) for v in values)
    if negate:
        return "violated" if hit else "satisfied"
    return "satisfied" if hit else "violated"


def main() -> None:
    ap = argparse.ArgumentParser(description="Screen candidate source conditionals")
    ap.add_argument("--specs", default="data/rules/screened_conditionals.json")
    ap.add_argument("--pairs", default="data/external/pairs_en_lb.jsonl")
    ap.add_argument("--source-filter", default="techgov_lu_en")
    ap.add_argument("--skip", type=int, default=0, help="Skip first N pairs (fit vs held-out half)")
    ap.add_argument("--rows", type=int, default=0)
    ap.add_argument("--samples-per-rule", type=int, default=40)
    ap.add_argument("--perturbations", type=int, default=3)
    ap.add_argument("--min-support", type=int, default=30)
    ap.add_argument("--min-tested", type=int, default=20)
    ap.add_argument("--min-agreement", type=float, default=0.65)
    ap.add_argument("--min-discrimination", type=float, default=0.5)
    ap.add_argument("--out", default="artifacts/audit/screen_conditionals_fit.json")
    args = ap.parse_args()

    specs = json.loads(Path(args.specs).read_text("utf-8"))["candidates"]
    for c in specs:
        re.compile(c["source_pattern"])
        for v in c["values"]:
            re.compile(v)

    v = TranslationVerifier()
    sat: dict[str, list[dict]] = collections.defaultdict(list)
    app = collections.Counter()
    verdict = collections.defaultdict(collections.Counter)
    scanned = 0
    with Path(args.pairs).open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if args.source_filter and rec.get("source") != args.source_filter:
                continue
            if scanned < args.skip:
                scanned += 1
                continue
            scanned += 1
            s = v.normalize_text(rec["src"])
            t = v.normalize_text(rec["tgt_ref"])
            for c in specs:
                if not re.search(c["source_pattern"], s):
                    continue
                app[c["name"]] += 1
                d = decide(t, c["values"], bool(c.get("negate")))
                verdict[c["name"]][d] += 1
                if d == "satisfied" and len(sat[c["name"]]) < args.samples_per_rule:
                    sat[c["name"]].append(rec)
            if args.rows and scanned - args.skip >= args.rows:
                break

    out: dict[str, dict] = {}
    for c in specs:
        n = app[c["name"]]
        ns, nv = verdict[c["name"]]["satisfied"], verdict[c["name"]]["violated"]
        agree = ns / n if n else None
        # can the check bite? break every match of every value pattern (positional prohibitions
        # are the mirror case: inject the forbidden shape)
        tested = flips = 0
        for i, rec in enumerate(sat[c["name"]]):
            tgt = v.normalize_text(rec["tgt_ref"])
            for k in range(args.perturbations):
                if c.get("negate"):
                    p, note = vd.inject_forbidden(v, tgt, c["values"], seed=1000 + i * 7 + k)
                else:
                    p, note = vd.corrupt_all(v, tgt, c["values"], seed=1000 + i * 7 + k,
                                            min_instances=1)
                if p is None or p == tgt:
                    continue
                tested += 1
                if decide(p, c["values"], bool(c.get("negate"))) == "violated":
                    flips += 1
        disc = flips / tested if tested else None
        if agree is None or n < args.min_support or tested < args.min_tested:
            why = "no support" if agree is None else "support too thin"
            status = "provisional" if agree is not None and agree >= args.min_agreement and \
                (disc or 0) >= args.min_discrimination else "no-support"
        elif agree < args.min_agreement:
            why, status = "agreement below floor", "refuted"
        elif disc is not None and disc < args.min_discrimination:
            why, status = "cannot bite (perturbations keep the check satisfied)", "vacuous"
        else:
            why, status = "", "USABLE"
        out[c["name"]] = {"applicable": n, "satisfied": ns, "violated": nv,
                          "agreement": None if agree is None else round(agree, 4),
                          "perturbations_tested": tested,
                          "discrimination": None if disc is None else round(disc, 4),
                          "status": status, "why": why,
                          "source_pattern": c["source_pattern"], "values": c["values"]}
        print(f"  {c['name']:22} n={n:5} agree="
              f"{'  n/a' if agree is None else f'{agree:.2f}'}"
              f" disc={'  n/a' if disc is None else f'{disc*100:5.1f}%'}"
              f" (tested={tested:4}) -> {status}{'  ' + why if why else ''}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"pairs_scanned": scanned, "skip": args.skip,
                                          "candidates": out}, ensure_ascii=False, indent=1),
                              encoding="utf-8")
    usable = [k for k, x in out.items() if x["status"] == "USABLE"]
    print(f"scanned={scanned} candidates={len(specs)} USABLE={len(usable)} {usable}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
