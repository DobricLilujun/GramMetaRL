#!/usr/bin/env python3
"""Discrimination test: does a certified rule actually BITE, or does it never fire a violation?

A rule with 0% false penalties on professional translations can be either
  (a) a correct, precise check, or
  (b) a check that can never fail on anything.
Only (a) is usable as RL supervision: a term that is constant +1 whenever it applies cannot shape a
policy. The agreement audit cannot tell them apart, because it only ever looks at CORRECT
translations.

So: for each certified (pair, rule) label, build a minimal perturbation that should break THAT rule -
delete or corrupt the exact span the satisfied constraint matched - and require the verifier to flip
to `violated`. The discrimination rate is the fraction of perturbations caught. Rules that never bite
are reported as such.
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


def corrupt_token(v: TranslationVerifier, text: str, patterns: list[str], seed: int) -> tuple[str | None, str]:
    """Token-wise minimal edit for anchored token-level constraints.

    A token-level constraint like ^.*en$ matches "some token ends in -en", so a span-based edit
    would rewrite the whole sentence and stop being minimal. Here we pick one matched TOKEN and
    change only its inflection: drop the -en (singularize) or swap it for a wrong plural marker.
    """
    toks = re.split(r"(\s+)", text)
    cands = []
    for idx, tok in enumerate(toks):
        if not tok or tok.isspace():
            continue
        bare = re.sub(r"[^0-9A-Za-zÀ-ÿ'’-]+$", "", tok)
        if not bare:
            continue
        for pat in patterns:
            core = pat.strip("^$")
            try:
                if re.search(core, v.normalize_text(bare)):
                    cands.append((idx, tok, bare))
                    break
            except re.error:
                continue
    if not cands:
        return None, "no token matches the constraint"
    idx, tok, bare = cands[seed % len(cands)]
    kind = seed % 3
    if kind == 0 and bare.lower().endswith("en"):
        repl = bare[:-2]
        how = f"singularize {bare!r} -> {repl!r}"
    elif kind == 1:
        repl = re.sub(r"en$", "eren", bare) if bare.lower().endswith("en") else bare + "eren"
        how = f"wrong plural {bare!r} -> {repl!r}"
    else:
        repl = re.sub(r"en$", "er", bare)
        how = f"swap ending {bare!r} -> {repl!r}"
    if repl == bare or not repl:
        return None, "perturbation was a no-op"
    out = list(toks)
    out[idx] = tok.replace(bare, repl)
    return "".join(out).strip(), how


def pattern_literal(pat: str) -> str:
    """First literal surface chunk of a pattern, anchors/quantifiers/escapes stripped."""
    cleaned = re.sub(r"\\[bBwWsSdD]|\\d\\+|\\w\\*|\(\?|\)|\(|\^|\$|[|*+?{}\[\]]", " ", pat)
    for tok in re.split(r"[^0-9A-Za-zÀ-ÿ'’]+", cleaned):
        if len(tok) >= 2:
            return tok
    return ""


def inject_forbidden(v: TranslationVerifier, text: str, patterns: list[str],
                     seed: int) -> tuple[str | None, str]:
    """PROHIBITIONS (negate=True) have nothing to break in a satisfied check, so discrimination is the
    opposite move: INSERT the forbidden shape and require the verdict to flip to violated. Positional
    bans (a form may not be sentence-initial) are injected at the front."""
    import random

    cand = [p for p in patterns if p]
    if not cand:
        return None, "no forbidden pattern"
    pat = cand[seed % len(cand)]
    lit = pattern_literal(pat)
    if not lit:
        return None, f"no literal in {pat!r}"
    toks = text.split()
    if pat.lstrip().startswith("^"):
        return " ".join([lit] + toks), f"prepended {lit!r}"
    if not toks:
        return None, "empty target"
    i = random.Random(seed).randrange(len(toks) + 1)
    return " ".join(toks[:i] + [lit] + toks[i:]), f"injected {lit!r} at token {i}"


def corrupt_all(v: TranslationVerifier, text: str, patterns: list[str], seed: int,
                min_instances: int = 2) -> tuple[str | None, str]:
    """Corrupt EVERY occurrence of the required shape.

    A rule that must hold for each instance (every possessor, every comparative) cannot be
    discriminated by breaking one of several: with 2+ matching tokens, a single-token corruption
    legitimately leaves the check satisfied, and the naive metric then calls the rule vacuous.
    """
    # Count INSTANCES, not per-pattern hits: a rule with several alternatives (mäi | hir | eise)
    # matches once per alternative, and overlapping alternatives can describe the same token - so
    # count the union of match spans.
    spans: set[tuple[int, int]] = set()
    for pat in patterns:
        try:
            spans |= {(m.start(), m.end()) for m in re.finditer(pat, text, re.I)}
        except re.error:
            continue
    merged: list[tuple[int, int]] = []
    for st, en in sorted(spans):
        if merged and st <= merged[-1][1]:      # overlapping alternatives = one instance
            merged[-1] = (merged[-1][0], max(merged[-1][1], en))
        else:
            merged.append((st, en))
    n_hits = len(merged)
    if n_hits < min_instances:
        return None, f"only {n_hits} instance(s), single-token corruption is enough"
    out = text
    how = []
    for i, pat in enumerate(patterns):
        try:
            rx = re.compile(pat, re.I)
        except re.error:
            continue
        if not rx.search(out):
            continue
        rep, _cnt = rx.subn(lambda m: _break_word(m.group(0), seed + i), out)
        out = rep
        how.append(pat)
    if out == text:
        return None, "nothing corrupted"
    return out, f"corrupted all instances of {len(how)} pattern(s)"


def _break_word(word: str, seed: int) -> str:
    """Replace the WHOLE matched span with a token no required-shape pattern can match.

    Keeping a prefix does not work: the constraint patterns end in \w* by design (they must match
    an inflected paradigm), so `hiren` -> `hirxx5` still satisfies `\bhir\w*\b` and the perturbation
    is a no-op (measured: 126/126 single-token perturbations failed to flip for exactly this reason).
    """
    return f"fzz{seed % 9}"


def corrupt(v: TranslationVerifier, text: str, patterns: list[str], seed: int) -> tuple[str | None, str]:
    """Minimal edit: delete / replace / mangle the first span matched by a constraint pattern."""
    norm = v.normalize_text(text)
    for pat in patterns:
        for lvl_pat in (pat, pat.strip("^$")):
            try:
                m = re.search(lvl_pat, norm)
            except re.error:
                continue
            if not m:
                continue
            span = m.group(0)
            if not span.strip():
                continue
            om = re.search(re.escape(span), text, flags=re.IGNORECASE)
            if not om:
                continue
            kind = seed % 3
            if kind == 0:
                out = (text[: om.start()] + text[om.end():]).strip()
                how = f"delete {span!r}"
            elif kind == 1:
                out = (text[: om.start()] + "deen" + text[om.end():]).strip()
                how = f"replace {span!r} -> 'deen'"
            else:
                out = (text[: om.start()] + span[:-1] + "x" + text[om.end():]).strip()
                how = f"mangle {span!r}"
            if out and out != text:
                return out, how
    return None, "no matchable span"


def main() -> None:
    ap = argparse.ArgumentParser(description="Do certified rules actually fire violations when broken?")
    ap.add_argument("--cards", default="data/rules/mbg_cards_recompiled_manual.jsonl")
    ap.add_argument("--labels", default="data/processed/labeled_gated_zerofp.jsonl")
    ap.add_argument("--reliability", default="data/rules/reliability_predicate_gated_techgov.json")
    ap.add_argument("--min-agreement", type=float, default=0.65)
    ap.add_argument("--rows", type=int, default=400)
    ap.add_argument("--perturbations", type=int, default=3, help="edits per (pair, rule)")
    ap.add_argument("--out", default="artifacts/audit/verifier_discrimination.json")
    args = ap.parse_args()

    cards = {
        c.id: c
        for c in (MBGCard.model_validate_json(l) for l in Path(args.cards).open("r", encoding="utf-8") if l.strip())
    }
    rel = json.loads(Path(args.reliability).read_text("utf-8"))["rules"]
    allowed = [cid for cid, r in rel.items() if r["agreement"] >= args.min_agreement and cid in cards]
    if not allowed:
        raise SystemExit(f"no card passes agreement >= {args.min_agreement}")

    v = TranslationVerifier(require_applicability=False)  # the constraint itself is under test
    per_rule = collections.defaultdict(collections.Counter)
    notes: dict[str, list[str]] = collections.defaultdict(list)

    rows = []
    with Path(args.labels).open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) >= args.rows:
                break

    for i, rec in enumerate(rows):
        for cid in rec["gold_rules"]:
            if cid not in allowed:
                continue
            card = cards[cid]
            pats: list[str] = []
            token_level = False
            for c in card.output_constraints:
                if c.kind == "must_include_regex" and c.value:
                    pats.append(c.value)
                    token_level = token_level or c.level == "token"
                elif c.kind == "must_include_any_regex":
                    pats.extend([x for x in (c.values or []) if x])
                    token_level = token_level or c.level == "token"
            if not pats:
                per_rule[cid]["no_testable_constraint"] += 1
                continue
            for k in range(args.perturbations):
                # token-level constraints get token-level perturbations, so the edit stays minimal
                bad, how = (corrupt_token(v, rec["tgt_ref"], pats, i + k) if token_level
                            else corrupt(v, rec["tgt_ref"], pats, i + k))
                if bad is None:
                    per_rule[cid]["unperturbable"] += 1
                    continue
                rep = v.verify(rec["src"], bad, [card])
                verdict = rep.results[0].verdict
                per_rule[cid]["tested"] += 1
                per_rule[cid][verdict] += 1
                if verdict != "violated" and len(notes[cid]) < 3:
                    notes[cid].append({"how": how, "orig": rec["tgt_ref"][:90], "bad": bad[:90], "verdict": verdict})

    out = {"labels_sampled": len(rows), "cards_allowed": allowed, "per_rule": {}, "escaped_examples": notes}
    print(f"labels_sampled={len(rows)} allowed={allowed}")
    for cid, c in per_rule.items():
        tested = c["tested"]
        caught = c["violated"]
        rate = caught / tested if tested else 0.0
        out["per_rule"][cid] = {
            "tested": tested,
            "violated": caught,
            "satisfied": c["satisfied"],
            "unknown": c["unknown"],
            "discrimination_rate": round(rate, 3),
            "unperturbable": c["unperturbable"],
            "agreement_on_real_data": rel[cid]["agreement"],
            "n_agreement": rel[cid]["applicable"],
        }
        print(f"  {cid} tested={tested:4d} caught={caught:4d} discrimination={rate:.1%} "
              f"(agreement on real data {rel[cid]['agreement']:.2f} over n={rel[cid]['applicable']})")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), "utf-8")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
