#!/usr/bin/env python3
"""Recompile card constraints into kinds that match what the rule actually says.

Why: the audit showed a 46-64% constraint-violation rate on professional human translations. The
dominant cause is kind mismatch, not typos - extraction could only emit must_include/must_exclude,
so rules like "nouns do not take case markers" or "nominative and accusative forms are identical"
were compiled into containment checks that can never pass. This pass asks the model to re-choose the
kind (with real alternatives), then applies strict local admission tests:

  1. the value must compile and must not be prose or a bare identifier;
  2. include-style kinds must fire on at least one of the card's OWN example targets;
  3. exclude-style kinds must NOT fire on any of the card's own example targets - if a card's own
     example trips its exclusion, the exclusion is wrong, full stop;
  4. equivalence / optionality are accepted as explicit deferred checks (they abstain), which is
     honest and keeps them out of the machine-checked reward until a morphology engine exists (P3).

Anything failing admission keeps its previous form and records the reason, so the pass is auditable.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grammetarl.llm_extract import looks_like_prose
from grammetarl.llm_http import chat_text
from grammetarl.schema import MBGCard, RuleConstraint
from grammetarl.validator import TranslationVerifier

ALLOWED = {
    "must_include_regex",
    "must_exclude_regex",
    "must_include_any_regex",
    "token_count_range",
    "equivalence",
    "optionality",
    "llm_check",
}

SYSTEM = """You re-compile grammar-rule constraints into the RIGHT kind of machine check.

You get cards with their current constraints, their own example pairs (src=English, tgt=Luxembourgish),
and counter-examples where the current constraint failed on a professional human translation.

For every constraint, choose the kind that matches what the rule says:
  must_exclude_regex      - rule states an impossibility ("does not cliticize", "no case marker")
  must_include_regex      - one specific form/shape must appear
  must_include_any_regex  - several attested alternatives, any one is fine (values = real forms)
  optionality             - something may be present or absent (not decidable by containment)
  equivalence             - two forms/paradigms are identical (e.g. nominative = accusative)
  llm_check               - needs meaning/world knowledge (e.g. "the referent is female")

Return strict JSON:
[{"card_id": "...", "constraints": [
  {"index": 0, "kind": "...", "value": "<regex or null>", "values": ["<alt>", ...],
   "level": "token|phrase|sentence", "why": "<= 12 words"}
]}]

Hard rules:
- `value`/`values` entries must be Python regexes over a normalized (casefolded, NFKC, punctuation-
  stripped) sentence, or a single token when level=token (then anchor with ^...$).
- Only put forms in `values` that appear in the card's own examples or are direct inflections of them.
- Do not invent forms you cannot ground in the examples.
- Do not restate the rule in English inside `value`."""


def main() -> None:
    ap = argparse.ArgumentParser(description="Recompile constraint kinds with admission tests")
    ap.add_argument("--cards", default="data/rules/mbg_cards_repaired.jsonl")
    ap.add_argument("--audit", default="artifacts/audit/constraint_precision.json")
    ap.add_argument("--out", default="data/rules/mbg_cards_recompiled.jsonl")
    ap.add_argument("--report", default="artifacts/recompile/report.json")
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key-env", default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    cards = [MBGCard.model_validate_json(l) for l in Path(args.cards).open("r", encoding="utf-8") if l.strip()]
    audit_path = Path(args.audit)
    audit = json.loads(audit_path.read_text("utf-8")) if audit_path.exists() else {}
    counters: dict[str, list[dict]] = collections.defaultdict(list)
    for ex in audit.get("counter_examples", []):
        if len(counters[ex["card_id"]]) < 2:
            counters[ex["card_id"]].append({"en": ex["en"], "lb": ex["lb"], "messages": ex["messages"][:2]})
    per_card_audit = audit.get("per_card", {})

    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    v = TranslationVerifier()
    proposals: dict[str, list[dict]] = {}

    def compile_batch(batch: list[MBGCard]) -> None:
        payload = []
        for c in batch:
            payload.append(
                {
                    "card_id": c.id,
                    "statement": c.statement[:220],
                    "operation_type": c.operation_type,
                    "constraints": [
                        {"index": i, "kind": x.kind, "value": x.value, "description": x.description[:120]}
                        for i, x in enumerate(c.output_constraints)
                    ],
                    "examples": [{"src": e.source_sentence, "tgt": e.target_sentence} for e in c.examples[:3]],
                    "audit": per_card_audit.get(c.id),
                    "counter_examples": counters.get(c.id, []),
                }
            )
        text, meta = chat_text(
            endpoint=args.endpoint,
            model=args.model,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": "Cards:\n" + json.dumps(payload, ensure_ascii=False)}],
            api_key=api_key,
            timeout=args.timeout,
            max_tokens=args.max_tokens,
        )
        if meta.get("finish_reason") == "length":
            print(f"[RECOMPILE][WARN] truncated at max_tokens={args.max_tokens}; salvaging")
        start, end = text.find("["), text.rfind("]")
        if start < 0 or end <= start:
            return
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            dec, i, out = json.JSONDecoder(), start, []
            while i < end:
                j = text.find("{", i)
                if j < 0 or j > end:
                    break
                try:
                    obj, k = dec.raw_decode(text[j : end + 1])
                    out.append(obj)
                    i = j + k
                except json.JSONDecodeError:
                    i = j + 1
            data = out
        for d in data:
            if isinstance(d, dict) and d.get("card_id"):
                proposals[d["card_id"]] = d.get("constraints", [])

    for i in range(0, len(cards), args.batch_size):
        try:
            compile_batch(cards[i : i + args.batch_size])
            print(f"[RECOMPILE] batch {i // args.batch_size + 1}/{-(-len(cards) // args.batch_size)} ok")
        except Exception as exc:
            print(f"[RECOMPILE][WARN] batch {i // args.batch_size + 1} failed: {exc}")

    stats = collections.Counter()
    reasons: collections.Counter[str] = collections.Counter()
    out_cards: list[MBGCard] = []

    def targets(card: MBGCard) -> list[str]:
        return [v.normalize_text(e.target_sentence) for e in card.examples if e.target_sentence]

    def admits(card: MBGCard, kind: str, patterns: list[str], level: str) -> tuple[bool, str]:
        if kind not in ALLOWED:
            return False, "unknown kind"
        if kind in ("equivalence", "optionality", "llm_check", "token_count_range"):
            return True, "deferred/allowed"
        pats = [p for p in patterns if p]
        if not pats:
            return False, "no patterns"
        for p in pats:
            if looks_like_prose(p):
                return False, "prose or identifier value"
            try:
                re.compile(p)
            except re.error:
                return False, "invalid regex"
        tg = targets(card)
        if not tg:
            return False, "no example targets to test on"
        if kind.startswith("must_exclude"):
            for t in tg:
                for p in pats:
                    n, _ = v._pattern_hits(p, t, level)
                    if n > 0:
                        return False, "exclusion fires on the card's own example"
            return True, "admitted"
        fired = 0
        for t in tg:
            for p in pats:
                n, _ = v._pattern_hits(p, t, level)
                if n > 0:
                    fired += 1
                    break
        if fired == 0:
            return False, "include never fires on own examples"
        return True, "admitted"

    for c in cards:
        prop = {int(p.get("index", -1)): p for p in proposals.get(c.id, []) if isinstance(p, dict)}
        new_constraints: list[RuleConstraint] = []
        for idx, old in enumerate(c.output_constraints):
            p = prop.get(idx)
            if not p:
                new_constraints.append(old)
                stats["kept_no_proposal"] += 1
                continue
            kind = p.get("kind", old.kind)
            values = [str(x) for x in (p.get("values") or []) if x]
            value = p.get("value") or old.value
            level = p.get("level") if p.get("level") in ("token", "phrase", "sentence", "any") else (old.level or "phrase")
            pats = values or ([value] if value else [])
            ok, why = admits(c, kind, pats, level)
            if not ok:
                reasons[why] += 1
                stats["kept_rejected"] += 1
                new_constraints.append(
                    old.model_copy(update={"description": f"{old.description} [recompile rejected: {why}]"})
                )
                continue
            stats[f"kind::{kind}"] += 1
            if kind != old.kind:
                stats["changed_kind"] += 1
            new_constraints.append(
                RuleConstraint(
                    kind=kind,
                    description=f"{old.description} [recompiled from {old.kind}: {str(p.get('why', ''))[:60]}]",
                    value=(None if kind in ("equivalence", "optionality") else (value if len(pats) <= 1 else None)),
                    values=(values if kind in ("must_include_any_regex", "optionality", "equivalence") else []),
                    level=level,
                    min_tokens=old.min_tokens,
                    max_tokens=old.max_tokens,
                )
            )
        c.output_constraints = new_constraints
        out_cards.append(c)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for c in out_cards:
            f.write(json.dumps(c.model_dump(), ensure_ascii=False) + "\n")
    report = {"stats": dict(stats), "reject_reasons": dict(reasons)}
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
