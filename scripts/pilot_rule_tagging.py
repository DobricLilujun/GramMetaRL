#!/usr/bin/env python3
"""Pilot: rule-conditioned filtering of existing en->lb parallel data (P1 data engine).

The grammar book gave us only 14 sentence-grade pairs, so sentence-level training data
has to come from existing parallel corpora. But unlabeled pairs are useless for the
grammar reward: we need to know WHICH rule each pair instantiates.

Idea under test here: use the compiled card constraints as a *labeler*. For each
external (en, lb) pair, retrieve candidate cards from the English side and run the
verifier on the Luxembourgish side. A constraint that fires on a natural translation
is a candidate gold_rule label; a constraint that never fires anywhere is a verifier
with no recall, i.e. an invisible (therefore noisy) term in the P1 reward.

This measures that recall before we spend any GPU time.

Public HF datasets-server API only (no token, no `datasets` dependency).
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grammetarl.index import RuleIndex
from grammetarl.schema import MBGCard
from grammetarl.validator import TranslationVerifier

API = "https://datasets-server.huggingface.co/rows"


def _get_json(url: str, tries: int = 6) -> dict:
    """datasets-server returns sporadic 502s; retry with backoff instead of failing the run."""
    last: Exception | None = None
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=90) as resp:
                return json.load(resp)
        except Exception as exc:  # 502/503/timeouts are transient on that service
            last = exc
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"datasets-server failed after {tries} attempts: {last}")


def fetch_rows(dataset: str, config: str, split: str, n_rows: int, want_prefix: str | None):
    rows: list[dict] = []
    offset = 0
    page = 100
    while len(rows) < n_rows:
        params = {
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": offset,
            "length": min(page, n_rows),
        }
        url = f"{API}?{urllib.parse.urlencode(params)}"
        payload = _get_json(url)
        batch = payload.get("rows", [])
        if not batch:
            break
        for item in batch:
            row = item.get("row", {})
            if want_prefix and row.get("prefix") and row["prefix"] != want_prefix:
                continue
            rows.append(row)
            if len(rows) >= n_rows:
                break
        offset += len(batch)
    return rows


def _unwrap(hit):
    """RuleIndex.retrieve may yield (score, card) tuples or objects with attributes."""
    if isinstance(hit, tuple) and len(hit) == 2:
        first, second = hit
        if isinstance(second, MBGCard):
            return second, float(first)
        return first, float(second)
    card = getattr(hit, "card", hit)
    return card, float(getattr(hit, "score", 0.0))


def _has_executable(card: MBGCard) -> bool:
    """Card carries at least one machine-checkable constraint (regex kind)."""
    all_constraints = list(card.output_constraints) + list(card.trigger_conditions)
    return any("regex" in (getattr(c, "kind", "") or "") for c in all_constraints)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tag external parallel pairs with card rules")
    parser.add_argument("--dataset", default="mbarnig/Tatoeba-en-lb")
    parser.add_argument("--config", default="default")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--prefix",
        default="translate english to luxembourgish",
        help="Keep only this translation direction (Tatoeba-en-lb flags direction in `prefix`)",
    )
    parser.add_argument("--src-field", default="input_text")
    parser.add_argument("--tgt-field", default="target_text")
    parser.add_argument("--rows", type=int, default=600)
    parser.add_argument("--min-tokens", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=40)
    parser.add_argument("--cards", default="data/rules/mbg_cards.jsonl")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--out", default="data/processed/external_tagged.jsonl")
    args = parser.parse_args()

    cards = []
    with Path(args.cards).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cards.append(MBGCard.model_validate_json(line))
    compilable = [c for c in cards if _has_executable(c)]
    print(f"cards={len(cards)} with_constraints={len(compilable)}")

    rows = fetch_rows(args.dataset, args.config, args.split, args.rows, args.prefix)
    print(f"fetched rows (direction filter): {len(rows)}")

    index = RuleIndex(cards)
    verifier = TranslationVerifier()
    by_id = {c.id: c for c in cards}

    fired_counter: collections.Counter[str] = collections.Counter()
    failed_counter: collections.Counter[str] = collections.Counter()
    vacuous_counter: collections.Counter[str] = collections.Counter()
    tagged: list[dict] = []
    skipped_len = 0
    pairs_with_hit = 0
    pairs_fired = 0

    for row in rows:
        src = str(row.get(args.src_field, "")).strip()
        tgt = str(row.get(args.tgt_field, "")).strip()
        n_src = len(src.split())
        if not src or not tgt or n_src < args.min_tokens or n_src > args.max_tokens:
            skipped_len += 1
            continue
        hits = [_unwrap(h) for h in index.retrieve(src, top_k=args.top_k)]
        cand = [c for c, _ in hits if _has_executable(c)]
        if not cand:
            continue
        pairs_with_hit += 1
        report = verifier.verify(src, tgt, cand)
        fired: list[str] = []
        for res in getattr(report, "results", []) or []:
            rid = getattr(res, "rule_id", None)
            passed = bool(getattr(res, "passed", False))
            msgs = getattr(res, "messages", []) or []
            card = by_id.get(rid)
            if card is None or not _has_executable(card):
                continue
            # A rule with only llm_check constraints passes vacuously; that is not a label.
            if any("llm_check" in str(m) or "deferred" in str(m) for m in msgs) and not any(
                "llm_check" not in str(m) for m in msgs
            ):
                continue
            if bool(getattr(res, "vacuous", False)) and passed:
                vacuous_counter[rid] += 1
                continue
            if passed:
                fired.append(rid)
                fired_counter[rid] += 1
            else:
                failed_counter[rid] += 1
        if fired:
            pairs_fired += 1
            tagged.append(
                {
                    "id": f"ext_{len(tagged):06d}",
                    "src": src,
                    "tgt_ref": tgt,
                    "lang_pair": "en->lb",
                    "difficulty": "external",
                    "phenomena_tags": sorted({t for c in cand if c.id in fired for t in c.phenomenon_tags}),
                    "gold_rules": fired,
                    "gold_lexicon": [],
                    "source": args.dataset,
                }
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in tagged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"length-filtered out: {skipped_len}")
    print(f"pairs with >=1 candidate card: {pairs_with_hit}/{len(rows)}")
    print(f"pairs where >=1 constraint fired: {pairs_fired}/{len(rows)}")
    print(f"distinct cards that ever fire on natural data: {len(fired_counter)} / {len(compilable)}")
    print(f"tagged rows written: {len(tagged)} -> {out}")
    print("top firing cards:")
    for rid, n in fired_counter.most_common(10):
        print(f"  {n:4d}  {rid}")
    print("sample tagged pairs:")
    for r in tagged[:3]:
        print(f"  rules={r['gold_rules']}\n    en: {r['src'][:90]}\n    lb: {r['tgt_ref'][:90]}")
    print(f"vacuous (never-testable) exclusions counted separately: {sum(vacuous_counter.values())}")
    print("worst misfiring cards (constraint fails on natural translations):")
    for rid, n in failed_counter.most_common(10):
        print(f"  {n:4d}  {rid}")
    if fired_counter:
        hits = sum(fired_counter.values())
        misses = sum(failed_counter.values())
        print(f"constraint fire/fail ratio: {hits} fired vs {misses} failed "
              f"(fail rate {misses / max(1, hits + misses):.2%})")
    else:
        print("NO constraint fired anywhere: the compiled verifier set has zero recall "
              "on natural translations and cannot carry a grammar reward.")


if __name__ == "__main__":
    main()
