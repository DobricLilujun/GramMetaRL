#!/usr/bin/env python3
"""Compile prose trigger_conditions into executable applicability predicates.

Why: all 87 extracted cards state their triggers as prose (100 trigger strings, 5 regex-shaped),
and the verifier never evaluated them. Result measured on 48,436 real en->lb pairs: 97.8% of
sentences retrieved at least one executable card, and 85.2% of the resulting rule checks failed -
largely because the retrieved rule did not govern the sentence. A binary pass/fail reward over
those checks is mostly noise, so GRPO would amplify it.

The compiler asks the model for machine-checkable predicates, then applies *self-test admission*:
a predicate is accepted only if it
  (a) compiles as a regex,
  (b) is not prose wearing a regex costume,
  (c) actually fires on at least one of the card's own examples on the declared side.
Rejected predicates are recorded with a reason, and the verifier abstains on cards whose
applicability is unknown instead of penalizing them.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grammetarl.llm_http import chat_text
from grammetarl.schema import MBGCard, TriggerCondition
from grammetarl.validator import TranslationVerifier

SYSTEM = """You compile grammar-rule triggers into executable applicability predicates.

For each card you get: id, statement, scope, prose trigger_conditions, and examples
(source = English side, target = Luxembourgish side).

Return ONE JSON array. Each element is:
  {"card_id": "...", "predicates": [
     {"source_text": "<the exact prose trigger being compiled>",
      "side": "target" | "source",
      "kind": "regex" | "llm_check",
      "pattern": "<python regex, or null if kind=llm_check>",
      "level": "token" | "phrase" | "sentence",
      "negated": false}
  ]}

Hard rules:
- `pattern` must be a Python regex matched (case-insensitively, after NFKC + casefold +
  trailing-punctuation stripping) against either the whole normalized sentence (level=phrase or
  sentence) or a single whitespace token (level=token).
- Prefer the observable side. "a dative preposition is followed by the feminine article der" is
  observable on the TARGET; "the referent is a female person" is only observable on the SOURCE or
  needs meaning, so use kind=llm_check for those.
- level=token only for word-form inventories, and then the regex must be anchored (^(a|b|c)$).
- Never put a sentence or a gloss in `pattern`. If you cannot express it, use kind=llm_check.
- Emit at most 3 predicates per card. Do not invent triggers that are not in the list given."""


def prose_masquerading(pattern: str) -> bool:
    """Reject patterns that are ordinary words in regex clothing (no regex machinery)."""
    bare_words = re.findall(r"[A-Za-zÀ-ÿ](?:[A-Za-zÀ-ÿ'’-]*[ )]|\b)", pattern)
    machinery = any(t in pattern for t in ("\\b", "\\s", "^", "$", "[", "]", "(", "|", "?", "*", "+", "{", "."))
    words = re.findall(r"[A-Za-zÀ-ÿ']{2,}", pattern)
    return len(words) >= 3 and not machinery and not bare_words


def admission(card: MBGCard, pred: TriggerCondition, v: TranslationVerifier) -> tuple[bool, str]:
    if pred.kind != "regex":
        return False, f"kind={pred.kind} is not executable"
    if not pred.pattern:
        return False, "empty pattern"
    try:
        compiled = re.compile(pred.pattern, re.IGNORECASE)
    except re.error as exc:
        return False, f"invalid regex: {exc}"
    if prose_masquerading(pred.pattern):
        return False, "prose masquerading as regex"

    texts: list[str] = []
    for ex in card.examples:
        raw = ex.target_sentence if pred.side == "target" else ex.source_sentence
        if raw:
            texts.append(v.normalize_text(raw))
    if not texts:
        return False, f"card has no {pred.side}-side examples to test on"

    hits = 0
    for t in texts:
        n, _ = v._pattern_hits(pred.pattern, t, pred.level)
        if n < 0:
            return False, "pattern failed to match"
        if (n == 0) if pred.negated else (n > 0):
            hits += 1
    if hits == 0:
        return False, f"pattern does not fire on any of the card's own {len(texts)} {pred.side} examples"
    return True, f"admitted: fires on {hits}/{len(texts)} own examples"


def compile_batch(cards: list[MBGCard], endpoint: str, model: str, api_key_env: str | None,
                  max_tokens: int, timeout: int) -> dict[str, list[dict]]:
    payload = [
        {
            "id": c.id,
            "statement": c.statement,
            "scope": c.scope,
            "trigger_conditions": c.trigger_conditions,
            "examples": [{"source": e.source_sentence, "target": e.target_sentence} for e in c.examples[:2]],
        }
        for c in cards
    ]
    user = "Cards:\n" + json.dumps(payload, ensure_ascii=False)
    text, meta = chat_text(
        endpoint=endpoint,
        model=model,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        max_tokens=max_tokens,
        timeout=timeout,
        api_key=os.environ.get(api_key_env) if api_key_env else None,
    )
    if meta.get("finish_reason") == "length":
        print(f"[TRIG][WARN] response truncated at max_tokens={max_tokens}; salvaging complete objects")
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        return {}
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        # salvage complete objects
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
    return {d.get("card_id", ""): d.get("predicates", []) for d in data if isinstance(d, dict)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Compile prose triggers into executable predicates")
    ap.add_argument("--cards", default="data/rules/mbg_cards.jsonl")
    ap.add_argument("--out", default="data/rules/mbg_cards_app.jsonl")
    ap.add_argument("--report", default="artifacts/triggers/compile_report.json")
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key-env", default=None)
    ap.add_argument("--batch-size", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    cards = [MBGCard.model_validate_json(l) for l in Path(args.cards).open("r", encoding="utf-8") if l.strip()]
    v = TranslationVerifier()
    compiled: dict[str, list[dict]] = {}
    for i in range(0, len(cards), args.batch_size):
        batch = cards[i : i + args.batch_size]
        try:
            compiled.update(compile_batch(batch, args.endpoint, args.model, args.api_key_env,
                                          args.max_tokens, args.timeout))
            print(f"[TRIG] batch {i // args.batch_size + 1}/{(len(cards) + args.batch_size - 1) // args.batch_size} ok")
        except Exception as exc:  # keep partial progress
            print(f"[TRIG][WARN] batch {i // args.batch_size + 1} failed: {exc}")

    out_cards: list[MBGCard] = []
    stats = {"cards": len(cards), "triggers": sum(len(c.trigger_conditions) for c in cards),
             "predicates_returned": 0, "accepted": 0, "rejected": 0, "cards_with_applicability": 0,
             "reject_reasons": {}}
    for c in cards:
        preds_raw = compiled.get(c.id, [])
        preds: list[TriggerCondition] = []
        for p in preds_raw:
            # Strict enum validation: smaller models happily invent side="left" or kind="statement".
            # Silently coercing those into valid defaults would fabricate predicates the model never
            # meant, so reject them with a reason instead.
            kind_raw, side_raw, level_raw = p.get("kind"), p.get("side"), p.get("level", "sentence")
            if kind_raw not in ("regex", "llm_check", "note") or side_raw not in ("source", "target"):
                stats["predicates_returned"] += 1
                stats["rejected"] += 1
                key = "invalid enum from model"
                stats["reject_reasons"][key] = stats["reject_reasons"].get(key, 0) + 1
                print(f"[TRIG][REJECT] {c.id}: invalid kind={kind_raw!r} side={side_raw!r}")
                continue
            pred = TriggerCondition(
                source_text=str(p.get("source_text", "")),
                kind=kind_raw,
                side=side_raw,
                pattern=p.get("pattern"),
                level=level_raw if level_raw in ("token", "phrase", "sentence", "any") else "sentence",
                negated=bool(p.get("negated", False)),
            )
            ok, reason = admission(c, pred, v)
            pred.accepted, pred.reject_reason = ok, None if ok else reason
            if ok:
                stats["accepted"] += 1
            else:
                stats["rejected"] += 1
                stats["reject_reasons"][reason.split(":")[0] if ":" in reason else reason] = \
                    stats["reject_reasons"].get(reason.split(":")[0] if ":" in reason else reason, 0) + 1
            preds.append(pred)
        c = c.model_copy(update={"applicability": preds})
        if any(p.accepted for p in preds):
            stats["cards_with_applicability"] += 1
        out_cards.append(c)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for c in out_cards:
            f.write(json.dumps(c.as_dict() if hasattr(c, "as_dict") else c.model_dump(), ensure_ascii=False) + "\n")
    rep = Path(args.report)
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(json.dumps(stats, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"cards with executable applicability: {stats['cards_with_applicability']}/{stats['cards']} -> {out}")


if __name__ == "__main__":
    main()
