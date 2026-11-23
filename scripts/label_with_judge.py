#!/usr/bin/env python3
"""Offline applicability judging: use an LLM where an LLM belongs - at dataset-build time.

Where this fits. The three-valued verifier showed that only 5.8% of retrieved-rule checks are
decidable today: 66 of 87 cards state their triggers as prose, and 69 of the 73 predicates the
compiler rejected were rejected honestly (`llm_check` - "the referent is a female person" is not
surface-observable). Without applicability the grammar term must abstain, so the usable label rate
on real pairs is only 3.1%.

This script raises the label rate WITHOUT putting an LLM judge into the RL hot loop:
  1. retrieve candidate executable cards for a pair;
  2. ask the model only "does this rule govern this pair?" (applicability, from the prose triggers);
  3. for pairs where it does, run the *compiled* output constraints - machine code, not the judge -
     to get the satisfied/violated label;
  4. cache every judgement keyed by (pair, card) so the labels are reproducible and free to reuse.

The judge therefore never decides correctness of a translation; it only gates which machine check
runs. That keeps the reward auditable: labels come from regex constraints, the LLM only scopes them.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grammetarl.index import RuleIndex
from grammetarl.llm_http import chat_text
from grammetarl.schema import MBGCard
from grammetarl.validator import TranslationVerifier

JUDGE_SYSTEM = """You decide whether a grammar rule GOVERNS a given translation pair. You are not
judging whether the translation is good.

You get pairs (en -> lb) and, for each pair, candidate rules with their prose trigger conditions.
Return strict JSON:
[{"pair_id": "...", "applicable": ["<rule id>", ...]}]

Rules:
- A rule is applicable only if its trigger conditions are satisfied by this pair.
- If a trigger needs world knowledge you do not have, or you are unsure, leave the rule out.
- Never invent rule ids. Never explain."""


def key(pair_src: str, pair_tgt: str, card_id: str) -> str:
    return hashlib.sha1(f"{pair_src}|{pair_tgt}|{card_id}".encode("utf-8")).hexdigest()


def load_cache(path: Path) -> dict[str, bool]:
    out: dict[str, bool] = {}
    if path.exists():
        for line in path.open("r", encoding="utf-8"):
            if line.strip():
                rec = json.loads(line)
                out[rec["key"]] = bool(rec["applicable"])
    return out


def judge_batch(pairs: list[dict], cards_by_id: dict[str, MBGCard], endpoint: str, model: str,
                api_key: str | None, max_tokens: int, timeout: int) -> dict[str, list[str]]:
    payload = []
    for p in pairs:
        payload.append(
            {
                "pair_id": p["pair_id"],
                "en": p["src"],
                "lb": p["tgt_ref"],
                "candidates": [
                    {
                        "id": cid,
                        "statement": cards_by_id[cid].statement[:160],
                        "triggers": cards_by_id[cid].trigger_conditions[:3],
                    }
                    for cid in p["candidate_ids"]
                ],
            }
        )
    user = "Pairs:\n" + json.dumps(payload, ensure_ascii=False)
    text, meta = chat_text(
        endpoint=endpoint,
        model=model,
        messages=[{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}],
        api_key=api_key,
        timeout=timeout,
        max_tokens=max_tokens,
    )
    if meta.get("finish_reason") == "length":
        print(f"[JUDGE][WARN] truncated at max_tokens={max_tokens}; salvaging complete objects")
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        return {}
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
    return {
        d["pair_id"]: [c for c in d.get("applicable", []) if isinstance(c, str)]
        for d in data
        if isinstance(d, dict) and d.get("pair_id")
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Label pairs with judged applicability + machine constraints")
    ap.add_argument("--pairs", default="data/external/pairs_en_lb.jsonl")
    ap.add_argument("--cards", default="data/rules/mbg_cards_app.jsonl")
    ap.add_argument("--out", default="data/processed/tagged_judged.jsonl")
    ap.add_argument("--cache", default="data/external/judge_cache.jsonl")
    ap.add_argument("--rows", type=int, default=200)
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--batch-pairs", type=int, default=4)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key-env", default=None)
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--skip-source-filter", default=None)
    args = ap.parse_args()

    cards = [MBGCard.model_validate_json(l) for l in Path(args.cards).open("r", encoding="utf-8") if l.strip()]
    cards_by_id = {c.id: c for c in cards}
    # Only cards with a machine-checkable output constraint can yield a label; the judge only scopes them.
    executable = [c for c in cards if any("regex" in (x.kind or "") for x in c.output_constraints)]
    index = RuleIndex(cards)
    verifier = TranslationVerifier(require_applicability=False)
    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    cache_path = Path(args.cache)
    cache = load_cache(cache_path)
    print(f"cards={len(cards)} executable={len(executable)} cache_entries={len(cache)}")

    pairs: list[dict] = []
    with Path(args.pairs).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if args.skip_source_filter and rec.get("source") != args.skip_source_filter:
                continue
            src, tgt = rec.get("src", ""), rec.get("tgt_ref", "")
            if not src or not tgt:
                continue
            pairs.append(rec)
            if len(pairs) >= args.rows:
                break

    # Build candidate lists: rules whose applicability the verifier cannot decide by itself.
    for rec in pairs:
        hits = []
        for h in index.retrieve(rec.get("src", ""), top_k=args.top_k):
            card = None
            if isinstance(h, tuple):
                for item in h:
                    if isinstance(item, MBGCard):
                        card = item
                        break
            else:
                card = getattr(h, "card", h)
            if card is None or card.id not in cards_by_id:
                continue
            c = cards_by_id[card.id]
            if not any("regex" in (x.kind or "") for x in c.output_constraints):
                continue
            if any(p.accepted for p in c.applicability):
                continue  # already decidable by the verifier; no judge needed
            hits.append(c.id)
        rec["candidate_ids"] = hits
        rec["pair_id"] = hashlib.sha1(f"{rec.get('src')}|{rec.get('tgt_ref')}".encode("utf-8")).hexdigest()[:16]
    pairs = [p for p in pairs if p["candidate_ids"]]
    print(f"pairs with undecidable-but-executable candidates: {len(pairs)}")

    stats = collections.Counter()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache_f = cache_path.open("a", encoding="utf-8")
    written = 0

    with out_path.open("w", encoding="utf-8") as out:
        for i in range(0, len(pairs), args.batch_pairs):
            batch = pairs[i : i + args.batch_pairs]
            # Reuse cached judgements. Applicability is a property of (pair, rule), so after a
            # constraint repair we can re-derive labels from the cache with ZERO model calls.
            verdict_map: dict[str, list[str]] = {}
            uncached: list[dict] = []
            for rec in batch:
                keys = {cid: key(rec["src"], rec["tgt_ref"], cid) for cid in rec["candidate_ids"]}
                if all(k in cache for k in keys.values()):
                    verdict_map[rec["pair_id"]] = [cid for cid, k in keys.items() if cache[k]]
                    stats["cache_hit_checks"] += len(keys)
                else:
                    uncached.append(rec)
            if uncached:
                try:
                    fresh = judge_batch(uncached, cards_by_id, args.endpoint, args.model, api_key,
                                        args.max_tokens, args.timeout)
                except Exception as exc:
                    print(f"[JUDGE][WARN] batch {i // args.batch_pairs + 1} failed: {exc}")
                    continue
                for rec in uncached:
                    got = {c for c in fresh.get(rec["pair_id"], []) if c in set(rec["candidate_ids"])}
                    for cid in rec["candidate_ids"]:
                        cache[key(rec["src"], rec["tgt_ref"], cid)] = cid in got
                    verdict_map[rec["pair_id"]] = sorted(got)
            for rec in batch:
                applicable = [cid for cid in verdict_map.get(rec["pair_id"], []) if cid in set(rec["candidate_ids"])]
                for cid in rec["candidate_ids"]:
                    cache_f.write(json.dumps({"key": key(rec["src"], rec["tgt_ref"], cid),
                                              "applicable": cid in applicable}) + "\n")
                    stats["judged"] += 1
                if not applicable:
                    stats["no_rule_applies"] += 1
                    continue
                report = verifier.verify(rec["src"], rec["tgt_ref"], [cards_by_id[c] for c in applicable])
                satisfied = [r.rule_id for r in report.results if r.verdict == "satisfied"]
                violated = [r.rule_id for r in report.results if r.verdict == "violated"]
                stats["pairs_applicable"] += 1
                if satisfied:
                    stats["pairs_labeled"] += 1
                    written += 1
                    out.write(
                        json.dumps(
                            {
                                "id": f"judged_{rec['pair_id']}",
                                "src": rec["src"],
                                "tgt_ref": rec["tgt_ref"],
                                "lang_pair": "en->lb",
                                "difficulty": "judged",
                                "phenomena_tags": sorted({t for cid in satisfied for t in cards_by_id[cid].phenomenon_tags}),
                                "gold_rules": satisfied,
                                "violated_rules": violated,
                                "judged_applicable": applicable,
                                "gold_lexicon": [],
                                "source": rec.get("source", "external"),
                                "license": rec.get("license", "unknown"),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                else:
                    stats["applicable_but_violated_only"] += 1
            print(f"[JUDGE] batch {i // args.batch_pairs + 1}/{(len(pairs) + args.batch_pairs - 1) // args.batch_pairs} "
                  f"labeled={written} judged={stats['judged']}")
    cache_f.close()

    print(json.dumps(dict(stats), ensure_ascii=False, indent=2))
    n = len(pairs)
    print(f"labeled pairs: {written} / {n} = {written / max(1, n):.1%} "
          f"(three-valued verifier alone gave 3.1% on 3,000 pairs)")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
