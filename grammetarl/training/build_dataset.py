from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

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
    parser = argparse.ArgumentParser(description="Build lexicon and translation training set from LOD")
    parser.add_argument("--lod", default=None, help="Path to LOD JSONL (default: assets/lb/new_lod-art.jsonl)")
    parser.add_argument("--max-entries", type=int, default=5000, help="Max LOD entries to scan (speed)")
    parser.add_argument("--namespace", default=None, help="Workspace namespace for default output paths")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    args = parser.parse_args()

    ws = get_workspace_paths(namespace=args.namespace)
    ws.ensure_layout()

    lod_path = Path(args.lod) if args.lod else (Path(ws.root) / "assets" / "lb" / "new_lod-art.jsonl")
    if not lod_path.exists():
        raise FileNotFoundError(f"LOD lexicon not found: {lod_path}")

    n_lex = build_lexicon(lod_path, ws.lexicon_file, max_entries=args.max_entries)
    n_train, n_valid = build_training_set(
        lod_path,
        ws.train_dataset_file,
        ws.root / "data" / "processed" / ("valid.jsonl" if ws.namespace == "default" else f"valid_{ws.namespace}.jsonl"),
        max_entries=args.max_entries,
        seed=args.seed,
        valid_ratio=args.valid_ratio,
    )
    print(f"Lexicon entries written: {n_lex} -> {ws.lexicon_file}")
    print(f"Training examples: {n_train} -> {ws.train_dataset_file}")
    print(f"Validation examples: {n_valid}")


if __name__ == "__main__":
    main()