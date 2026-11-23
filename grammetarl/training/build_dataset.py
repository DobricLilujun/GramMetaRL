from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from grammetarl.schema import MBGCard
from grammetarl.workspace import get_workspace_paths


# ---------------------------------------------------------------------------
# LOD lexicon parsing
# The LOD XML was converted to JSONL by scripts/convert_lod_xml_to_jsonl.py.
# Each JSONL line is a lossless tree node: {"record_type", "record_id", "record"}.
# The relevant shape is:
#   entry -> lemma (text)
#          -> microStructure -> partOfSpeech (text = POS code)
#                              -> meaning -> targetLanguage (attrs.lang)
#                                          -> translation (text = translated word)
# ---------------------------------------------------------------------------

DEFAULT_TARGET_LANG = "en"  # English is the source language for en->lb translation


def _iter_lod(path: Path, max_entries: int | None = None):
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_entries is not None and i >= max_entries:
                break
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _find_all(node: dict, tag: str) -> list[dict]:
    out: list[dict] = []
    if node.get("tag") == tag:
        out.append(node)
    for child in node.get("children") or []:
        out.extend(_find_all(child, tag))
    return out


def _first_text(node: dict, tag: str) -> str:
    matches = _find_all(node, tag)
    for m in matches:
        text = (m.get("text") or "").strip()
        if text:
            return text
    return ""


def _parse_entry(entry: dict) -> dict:
    """Return a normalized lexicon entry with lemma, pos, and per-language translations."""
    inner = entry.get("record", entry)
    lemma = _first_text(inner, "lemma")
    pos = _first_text(inner, "partOfSpeech")

    translations: dict[str, list[str]] = {}
    for tl in _find_all(inner, "targetLanguage"):
        lang = str((tl.get("attrs") or {}).get("lang", "")).strip().lower()
        if not lang:
            continue
        for t in _find_all(tl, "translation"):
            value = (t.get("text") or "").strip()
            if value:
                translations.setdefault(lang, [])
                if value not in translations[lang]:
                    translations[lang].append(value)

    # Morphology (inflection forms), if present.
    morphology: list[str] = []
    for form in _find_all(inner, "form"):
        value = (form.get("text") or "").strip()
        if value and value != lemma:
            morphology.append(value)

    return {
        "lemma": lemma,
        "pos": pos,
        "translations": translations,
        "morphology": morphology[:8],
    }


def _stable_id(seed: str) -> str:
    """Deterministic id. The previous code used Python's hash(), which is salted per process,
    so ids and the train/valid split were not reproducible across runs."""
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]


def load_cards_light(cards_path: Path) -> list[MBGCard]:
    """Load cards without importing the PDF/OCR stack."""
    cards: list[MBGCard] = []
    with cards_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cards.append(MBGCard.model_validate_json(line))
    return cards


def _classify_example(src: str, tgt: str) -> str:
    """Classify a card example pair by whether it can serve as a translation sentence.

    Grammar-book examples are mostly word/paradigm snippets with IPA, e.g.
      "(I) ask vs. (you) ask" -> "froen [ˈfʀoːən] vs. frees [fʀeːs]"
    Those are valuable for morphological drills but useless as MT training pairs
    (gate G12: mean src length 3.0 tokens). Keep them, but out of the sentence set.
    """
    text = f"{src} {tgt}"
    if "[" in text and "]" in text:
        return "ipa_gloss"
    low = f"{src} {tgt}".lower()
    if " vs " in low or " vs. " in low or ">" in text or "~" in text:
        return "paradigm"
    if len(src.split()) >= 5 and any(c in src for c in "!?") or (
        len(src.split()) >= 6 and src[:1].isalpha()
    ):
        return "sentence"
    return "phrase"


def build_sentence_dataset(
    cards_path: Path,
    out_train: Path,
    out_valid: Path,
    lang_pair: str = "en->lb",
    valid_ratio: float = 0.1,
    seed: int = 0,
    out_subsentence: Path | None = None,
) -> dict[str, object]:
    """Build sentence-level TranslationExample rows from the grammar cards' own example pairs.

    This replaces the word-gloss rows that made the multi-turn grammar environment untrainable
    (gate G12: mean src length was 1.6 tokens and gold_rules was empty in 200/200 rows).

    Split discipline: split BY CARD, so every example of one rule stays in one split. Otherwise
    a rule seen in train also appears in valid and 'did the policy learn the rule' is unanswerable.
    """
    cards = load_cards_light(cards_path)

    by_card: dict[str, list[dict[str, object]]] = {}
    sub_sentences: list[dict[str, object]] = []
    seen_pairs: set[tuple[str, str]] = set()
    skipped_no_source = 0
    class_counts: dict[str, int] = {}
    for card in cards:
        rows: list[dict[str, object]] = []
        for ex in card.examples:
            src = (ex.source_sentence or "").strip()
            tgt = (ex.target_sentence or "").strip()
            if not tgt:
                # Monolingual card example: keep it for grammar-verification, not as a pair.
                skipped_no_source += 1
                continue
            key = (src.lower(), tgt.lower())
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            kind = _classify_example(src, tgt)
            class_counts[kind] = class_counts.get(kind, 0) + 1
            row = {
                "id": f"card_{_stable_id(card.id + '|' + src + '|' + tgt)}",
                "src": src,
                "tgt_ref": tgt,
                "lang_pair": lang_pair,
                "difficulty": kind,
                "phenomena_tags": list(card.phenomenon_tags),
                "gold_rules": [card.id],
                "gold_lexicon": [],
                "source_pages": [card.source.page_start, card.source.page_end],
            }
            if kind == "sentence":
                rows.append(row)
            else:
                sub_sentences.append(row)
        if rows:
            by_card[card.id] = rows

    card_ids = sorted(by_card)
    random.Random(seed).shuffle(card_ids)
    n_valid_cards = int(round(len(card_ids) * valid_ratio))
    valid_ids, train_ids = card_ids[:n_valid_cards], card_ids[n_valid_cards:]

    train = [r for cid in train_ids for r in by_card[cid]]
    valid = [r for cid in valid_ids for r in by_card[cid]]
    random.Random(seed).shuffle(train)
    random.Random(seed + 1).shuffle(valid)

    for path, rows in ((out_train, train), (out_valid, valid)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if out_subsentence is not None:
        out_subsentence.parent.mkdir(parents=True, exist_ok=True)
        with out_subsentence.open("w", encoding="utf-8") as f:
            for r in sub_sentences:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return {
        "cards_with_pairs": len(by_card),
        "train_rows": len(train),
        "valid_rows": len(valid),
        "monolingual_examples_skipped": skipped_no_source,
        "example_classes": class_counts,
        "subsentence_rows": len(sub_sentences),
        "subsentence_output": str(out_subsentence) if out_subsentence else None,
        "train_card_ids": len(train_ids),
        "valid_card_ids": len(valid_ids),
        "train_output": str(out_train),
        "valid_output": str(out_valid),
    }


def build_from_tagged_external(
    tagged_paths: list[Path],
    out_train: Path,
    out_valid: Path,
    out_rule_heldout: Path,
    lang_pair: str = "en->lb",
    valid_ratio: float = 0.1,
    heldout_rule_ratio: float = 0.15,
    seed: int = 0,
) -> dict[str, object]:
    """Merge verifier-tagged external pairs with card sentence examples.

    Two splits, because the paper asks two different questions:
      train/valid          pair-level split -> "can the policy translate"
      train/valid_heldout  RULE-level split  -> "did the policy learn rules it never saw"
    The rule-level split is what makes the grammar claim falsifiable: every row whose
    gold_rules intersect a held-out rule set is removed from train entirely.
    """
    rows: list[dict[str, object]] = []
    licenses: dict[str, int] = {}
    for path in tagged_paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if not r.get("gold_rules"):
                    continue
                licenses[r.get("license", "unknown")] = licenses.get(r.get("license", "unknown"), 0) + 1
                rows.append(r)

    rule_ids = sorted({rid for r in rows for rid in r["gold_rules"]})
    random.Random(seed).shuffle(rule_ids)
    n_heldout = int(round(len(rule_ids) * heldout_rule_ratio))
    heldout_rules = set(rule_ids[:n_heldout])

    heldout_rows = [r for r in rows if set(r["gold_rules"]) & heldout_rules]
    pool = [r for r in rows if not (set(r["gold_rules"]) & heldout_rules)]

    random.Random(seed).shuffle(pool)
    n_valid = int(round(len(pool) * valid_ratio))
    valid, train = pool[:n_valid], pool[n_valid:]
    random.Random(seed + 1).shuffle(train)

    for path, data in ((out_train, train), (out_valid, valid), (out_rule_heldout, heldout_rows)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # The falsifiable claim is: a rule held out of train never appears in train.
    # Held-out rows may still mention non-held-out rules, so compare against the held-out
    # rule SET, not the union of rules appearing in held-out rows.
    rules_path = out_rule_heldout.parent / "heldout_rules.json"
    rules_path.write_text(json.dumps(sorted(heldout_rules), ensure_ascii=False, indent=1), "utf-8")

    train_rules = {rid for r in train for rid in r["gold_rules"]}
    return {
        "rows_total": len(rows),
        "train_rows": len(train),
        "valid_rows": len(valid),
        "heldout_rows": len(heldout_rows),
        "rules_total": len(rule_ids),
        "rules_heldout": len(heldout_rules),
        "rule_overlap": len(train_rules & heldout_rules),
        "heldout_rules_file": str(rules_path),
        "licenses": licenses,
        "train_output": str(out_train),
        "valid_output": str(out_valid),
        "heldout_output": str(out_rule_heldout),
    }


def build_lexicon(lod_path: Path, out_path: Path, max_entries: int | None = None) -> int:
    """Write one lexicon record per (lemma, pos) that has at least one translation."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    seen: set[tuple[str, str]] = set()
    with out_path.open("w", encoding="utf-8") as f:
        for entry in _iter_lod(lod_path, max_entries):
            info = _parse_entry(entry)
            if not info["lemma"]:
                continue
            for lang, trans in info["translations"].items():
                key = (info["lemma"], info["pos"])
                if key in seen:
                    continue
                record = {
                    "record_type": "lexicon_entry",
                    "lemma": info["lemma"],
                    "pos": info["pos"],
                    "senses": {lang: trans for lang, trans in info["translations"].items()},
                    "morphology": info["morphology"],
                    "examples": [],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                seen.add(key)
                n += 1
    return n


def build_training_set(
    lod_path: Path,
    out_train: Path,
    out_valid: Path,
    lang_pair: str = "en->lb",
    target_lang: str = DEFAULT_TARGET_LANG,
    max_entries: int | None = None,
    seed: int = 0,
    valid_ratio: float = 0.1,
) -> tuple[int, int]:
    """Build TranslationExample rows from real (lb lemma, target-language) pairs.

    For en->lb we invert: src = English translation, tgt_ref = Luxembourgish lemma.
    """
    examples: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for entry in _iter_lod(lod_path, max_entries):
        info = _parse_entry(entry)
        if not info["lemma"]:
            continue
        en_words = info["translations"].get(target_lang, [])
        if not en_words:
            continue
        pos = info["pos"].lower()
        phenomena = _pos_to_tags(pos)
        for en in en_words:
            pair = (en, info["lemma"])
            if pair in seen:
                continue
            seen.add(pair)
            examples.append(
                {
                    "id": f"lex_{info['lemma']}_{abs(hash(en)) % 100000}",
                    "src": en,
                    "tgt_ref": info["lemma"],
                    "lang_pair": lang_pair,
                    "difficulty": "easy",
                    "phenomena_tags": phenomena,
                    "gold_rules": [],
                    "gold_lexicon": [info["lemma"]],
                }
            )

    rng = random.Random(seed)
    rng.shuffle(examples)
    split = int(len(examples) * (1 - valid_ratio))
    train, valid = examples[:split], examples[split:]

    out_train.parent.mkdir(parents=True, exist_ok=True)
    out_valid.parent.mkdir(parents=True, exist_ok=True)
    for path, rows in ((out_train, train), (out_valid, valid)):
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(train), len(valid)


def _pos_to_tags(pos: str) -> list[str]:
    mapping = {
        "subst": ["noun"],
        "verb": ["verb", "inflection"],
        "adj": ["adjective", "agreement"],
        "adv": ["adverb"],
        "prep": ["preposition_governance"],
        "interj": ["interjection"],
        "art": ["definiteness"],
        "pron": ["pronoun"],
    }
    return mapping.get(pos, [])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build lexicon + translation training set")
    parser.add_argument("--lod", default=None, help="Path to LOD JSONL (default: assets/lb/new_lod-art.jsonl)")
    parser.add_argument("--max-entries", type=int, default=5000, help="Max LOD entries to scan (speed)")
    parser.add_argument("--namespace", default=None, help="Workspace namespace for default output paths")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument(
        "--from-cards",
        default=None,
        help="MBG cards JSONL; sentence-level pairs come from the cards' own example pairs "
        "(default: data/rules/mbg_cards.jsonl when present)",
    )
    parser.add_argument("--lang-pair", default="en->lb")
    parser.add_argument("--no-lexicon", action="store_true", help="Skip the LOD lexicon build")
    parser.add_argument(
        "--external-tagged",
        nargs="*",
        default=None,
        help="Verifier-tagged external pairs (scripts/pilot_rule_tagging.py output); "
        "enables the rule-level held-out split",
    )
    parser.add_argument(
        "--lexicon-pairs",
        action="store_true",
        help="Also write the old word-gloss rows (to data/processed/train_lexicon.jsonl)",
    )
    args = parser.parse_args()

    ws = get_workspace_paths(namespace=args.namespace)
    ws.ensure_layout()
    valid_path = ws.root / "data" / "processed" / (
        "valid.jsonl" if ws.namespace == "default" else f"valid_{ws.namespace}.jsonl"
    )

    tagged = args.external_tagged or []

    cards_path = Path(args.from_cards) if args.from_cards else (ws.root / "data" / "rules" / "mbg_cards.jsonl")
    if not cards_path.exists():
        alt = ws.mbg_cards_file
        cards_path = alt if alt.exists() else cards_path

    if cards_path.exists() and not tagged:
        stats = build_sentence_dataset(
            cards_path,
            ws.train_dataset_file,
            valid_path,
            lang_pair=args.lang_pair,
            valid_ratio=args.valid_ratio,
            seed=args.seed,
            out_subsentence=ws.root / "data" / "processed" / "pairs_subsentences.jsonl",
        )
        print(
            f"Sentence dataset from cards ({cards_path}): "
            f"cards_with_pairs={stats['cards_with_pairs']} "
            f"train={stats['train_rows']} valid={stats['valid_rows']} "
            f"(split by card: train_cards={stats['train_card_ids']} valid_cards={stats['valid_card_ids']}; "
            f"monolingual examples kept out of the pair set: {stats['monolingual_examples_skipped']})"
        )
        print(f"  example classes: {stats['example_classes']}")
        print(f"  subsentence drills -> {stats['subsentence_output']} ({stats['subsentence_rows']} rows)")
        print(f"  -> {stats['train_output']}\n  -> {stats['valid_output']}")
    elif not tagged:
        print(
            f"[WARN] no cards at {cards_path}; falling back to LOD word-gloss rows. "
            "Word glosses cannot train the multi-turn grammar environment (gate G12)."
        )

    lod_path = Path(args.lod) if args.lod else (Path(ws.root) / "assets" / "lb" / "new_lod-art.jsonl")
    if args.no_lexicon:
        pass
    elif not lod_path.exists():
        print(f"[WARN] LOD lexicon not found: {lod_path} (skipping lexicon)")
    else:
        n_lex = build_lexicon(lod_path, ws.lexicon_file, max_entries=args.max_entries)
        print(f"Lexicon entries written: {n_lex} -> {ws.lexicon_file}")

    if tagged:
        stats = build_from_tagged_external(
            [Path(t) for t in tagged],
            ws.train_dataset_file,
            valid_path,
            ws.root / "data" / "processed" / "valid_rules_heldout.jsonl",
            lang_pair=args.lang_pair,
            valid_ratio=args.valid_ratio,
            seed=args.seed,
        )
        print(
            f"Tagged external data: rows={stats['rows_total']} "
            f"train={stats['train_rows']} valid={stats['valid_rows']} "
            f"rule-heldout={stats['heldout_rows']} "
            f"(rules {stats['rules_heldout']}/{stats['rules_total']} held out, "
            f"overlap={stats['rule_overlap']})"
        )
        print(f"  licenses: {stats['licenses']}")
        print(f"  -> {stats['train_output']}\n  -> {stats['valid_output']}\n  -> {stats['heldout_output']}")

    if args.lexicon_pairs:
        if not lod_path.exists():
            raise FileNotFoundError(f"LOD lexicon not found: {lod_path}")
        lex_out = ws.root / "data" / "processed" / (
            "train_lexicon.jsonl" if ws.namespace == "default" else f"train_lexicon_{ws.namespace}.jsonl"
        )
        n_train, n_valid = build_training_set(
            lod_path,
            lex_out,
            ws.root / "data" / "processed" / (
                "valid_lexicon.jsonl" if ws.namespace == "default" else f"valid_lexicon_{ws.namespace}.jsonl"
            ),
            max_entries=args.max_entries,
            seed=args.seed,
            valid_ratio=args.valid_ratio,
        )
        print(f"Lexicon word-gloss rows (NOT the training set): {n_train} train / {n_valid} valid -> {lex_out}")


if __name__ == "__main__":
    main()