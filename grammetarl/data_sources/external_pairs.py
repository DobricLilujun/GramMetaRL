"""External en<->lb parallel data ingestion with license/provenance bookkeeping.

Why this module exists: the grammar book yielded only 14 sentence-grade pairs, so sentence
data must come from external corpora. But a corpus without recorded license and provenance
cannot go into a paper or into a reward pipeline, and each source has different caveats:

  techgov_lu_en  10,880 pairs, professional translators, CC0 (public domain)  <- safest
  luxalign_v3    43,357 pairs, derived from RTL.lu news, CC BY-NC 4.0; the dataset card
                 itself warns pairs are "not always exact translations", only semantically
                 similar, so a wrong lb side must not be treated as a gold reference
  tatoeba_en_lb  42.8k rows, community sentences, noisy (targets like "Luxembourgish",
                 UI strings, "^..." markup); probe only

Every emitted row carries source + license so downstream filters can drop NC or noisy sets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "external" / "raw"
OUT_DIR = ROOT / "data" / "external"

SOURCES: dict[str, dict] = {
    "techgov_lu_en": {
        "path": RAW / "techingov_lu_en" / "LU-EN" / "LU-EN.jsonl",
        "src_key": "en",
        "tgt_key": "lb",
        "license": "CC0-1.0",
        "url": "https://data.public.lu/en/datasets/meisproochegen-iwwersetzungskorpus-fir-dletzebuergescht/",
        "note": "Tech-in-GOV 2025: professional translations, public domain",
        "commercial_ok": True,
    },
    "luxalign_v3_lb_en": {
        "path": RAW / "luxalign_v3_lb_en.json",
        "src_key": "en",
        "tgt_key": "lb",
        "license": "CC-BY-NC-4.0",
        "url": "https://huggingface.co/datasets/fredxlpy/LuxAlign",
        "note": "RTL.lu news-derived; dataset card warns pairs are semantically similar, "
        "not always exact translations",
        "commercial_ok": False,
    },
}

_WS = re.compile(r"\s+")
# Rows that are clearly not sentences (observed in the Tatoeba-derived HF export).
_JUNK_TGT = re.compile(r"^(luxembourgish|english|missing plugin|n\.?a\.?|[-–—]*)$", re.I)


def _clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text or ""))
    return _WS.sub(" ", text).strip()


def _stable_id(source: str, src: str, tgt: str) -> str:
    return hashlib.sha1(f"{source}|{src}|{tgt}".encode("utf-8")).hexdigest()[:16]


def usable(src: str, tgt: str, min_tokens: int = 4, max_tokens: int = 60) -> bool:
    if not src or not tgt or src.lower() == tgt.lower():
        return False
    if _JUNK_TGT.match(tgt):
        return False
    n_src, n_tgt = len(src.split()), len(tgt.split())
    if not (min_tokens <= n_src <= max_tokens and min_tokens <= n_tgt <= max_tokens):
        return False
    # Require the target to actually be Luxembourgish-ish: at least one lb-specific char.
    if not re.search(r"[ëéäàùçâîôû]", tgt, re.I):
        return False
    return True


def iter_source(name: str, path: Path, src_key: str, tgt_key: str):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield _clean(row.get(src_key, "")), _clean(row.get(tgt_key, ""))


def ingest(names: list[str], out_pairs: Path, out_manifest: Path, limit: int | None = None) -> dict:
    seen: set[tuple[str, str]] = set()
    stats: dict[str, dict] = {}
    out_pairs.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_pairs.open("w", encoding="utf-8") as out:
        for name in names:
            cfg = SOURCES[name]
            path = Path(cfg["path"])
            if not path.exists():
                stats[name] = {"error": f"missing {path}"}
                continue
            n_raw = n_ok = n_dup = n_rej = 0
            for src, tgt in iter_source(name, path, cfg["src_key"], cfg["tgt_key"]):
                n_raw += 1
                if limit and n_ok >= limit:
                    break
                key = (src.lower(), tgt.lower())
                if not usable(src, tgt):
                    n_rej += 1
                    continue
                if key in seen:
                    n_dup += 1
                    continue
                seen.add(key)
                n_ok += 1
                written += 1
                out.write(
                    json.dumps(
                        {
                            "id": _stable_id(name, src, tgt),
                            "src": src,
                            "tgt_ref": tgt,
                            "lang_pair": "en->lb",
                            "difficulty": "external",
                            "phenomena_tags": [],
                            "gold_rules": [],
                            "gold_lexicon": [],
                            "source": name,
                            "license": cfg["license"],
                            "source_note": cfg["note"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            stats[name] = {
                "raw": n_raw,
                "written": n_ok,
                "rejected": n_rej,
                "duplicate": n_dup,
                "license": cfg["license"],
                "commercial_ok": cfg["commercial_ok"],
                "path": str(path),
                "url": cfg["url"],
                "note": cfg["note"],
            }
    manifest = {
        "generated_from": {k: SOURCES[k]["url"] for k in names},
        "pairs_file": str(out_pairs),
        "total_pairs": written,
        "sources": stats,
        "usage_note": "License recorded per row. CC-BY-NC sources must not be used in any "
        "commercial artefact; LuxAlign pairs are not guaranteed exact translations and are "
        "used as verifier probes / candidate labels, never as gold MT references.",
    }
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest licensed external parallel pairs")
    parser.add_argument("--sources", default=",".join(SOURCES))
    parser.add_argument("--limit", type=int, default=None, help="Max rows per source")
    parser.add_argument("--out", default=str(OUT_DIR / "pairs_en_lb.jsonl"))
    parser.add_argument("--manifest", default=str(OUT_DIR / "MANIFEST.json"))
    args = parser.parse_args()
    manifest = ingest(
        [s for s in args.sources.split(",") if s],
        Path(args.out),
        Path(args.manifest),
        limit=args.limit,
    )
    print(f"total pairs: {manifest['total_pairs']} -> {manifest['pairs_file']}")
    for name, s in manifest["sources"].items():
        if "error" in s:
            print(f"  {name}: {s['error']}")
        else:
            print(
                f"  {name}: raw={s['raw']} written={s['written']} rejected={s['rejected']} "
                f"dup={s['duplicate']} license={s['license']}"
            )
    print(f"manifest -> {args.manifest}")


if __name__ == "__main__":
    main()
