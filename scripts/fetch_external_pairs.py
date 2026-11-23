#!/usr/bin/env python3
"""Fetch external en<->lb sentence pairs into data/external/ with a license manifest.

Sources stay separate and are never merged blindly: each has its own license and quality
profile, and the paper must be able to state which corpus produced which number.
MANIFEST.json records url, license, fetch time and row counts.

  luxalign   fredxlpy/LuxAlign (HF datasets-server, config lb-en)   CC BY-NC 4.0
  tatoeba    mbarnig/Tatoeba-en-lb (HF datasets-server)              probe only
  publiclu   data.public.lu open-data portal (Tech-in-GOV corpus)    per-resource license
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "external"
MANIFEST = OUT_DIR / "MANIFEST.json"
ROWS_API = "https://datasets-server.huggingface.co/rows"
TOKEN_FILE = Path.home() / ".cache" / "huggingface" / "token"


def token():
    if TOKEN_FILE.exists():
        t = TOKEN_FILE.read_text().strip()
        return t or None
    return None


def get_json(url, tries=5, auth=False):
    headers = {"User-Agent": "grammetarl/0.1 (academic research)"}
    if auth:
        tok = token()
        if tok:
            headers["Authorization"] = f"Bearer {tok}"
    last = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.load(resp)
        except Exception as exc:
            last = exc
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"failed after {tries} tries: {url} :: {last}")


def load_manifest():
    if MANIFEST.exists():
        try:
            return json.loads(MANIFEST.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {"sources": {}}


def save_manifest(entry):
    m = load_manifest()
    m["sources"][entry["source"]] = entry
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def fetch_hf_rows(dataset, config, split, limit, page=100):  # server caps /rows at 100
    rows, offset = [], 0
    while len(rows) < limit:
        want = min(page, limit - len(rows))
        url = (f"{ROWS_API}?dataset={urllib.parse.quote(dataset, safe='')}"
               f"&config={config}&split={split}&offset={offset}&length={want}")
        payload = get_json(url, auth=True)
        batch = payload.get("rows", [])
        if not batch:
            break
        rows.extend(item.get("row", {}) for item in batch)
        offset += len(batch)
        if len(batch) < want:
            break
    return rows


def cmd_luxalign(args):
    out = OUT_DIR / "luxalign_lb_en.jsonl"
    rows = fetch_hf_rows("fredxlpy/LuxAlign", args.config, "train", args.limit)
    pairs = []
    for r in rows:
        lb = (r.get("lb") or "").strip()
        en = (r.get("en") or "").strip()
        if lb and en:
            # Task direction is en -> lb; flip once here instead of letting every consumer
            # remember that the corpus is stored lb-first.
            pairs.append({"src": en, "tgt_ref": lb, "lang_pair": "en->lb", "origin": "luxalign"})
    write_jsonl(out, pairs)
    save_manifest({
        "source": "luxalign", "dataset": "fredxlpy/LuxAlign", "config": args.config,
        "url": "https://huggingface.co/datasets/fredxlpy/LuxAlign",
        "license": "CC BY-NC 4.0 (non-commercial; cite LuxEmbedder, Philippy et al. 2025)",
        "fetched_at": datetime.now(timezone.utc).isoformat(), "rows": len(pairs),
        "note": "corpus has a single train split; GramMetaRL does its own by-rule split and "
                "keeps Lux-MT-Test-Suite fully held out",
    })
    print(f"luxalign: {len(pairs)} pairs -> {out}")


def cmd_tatoeba(args):
    out = OUT_DIR / "tatoeba_en_lb.jsonl"
    rows = fetch_hf_rows("mbarnig/Tatoeba-en-lb", "default", args.split, args.limit)
    pairs = []
    for r in rows:
        if r.get("prefix") and r["prefix"] != args.prefix:
            continue
        src = (r.get("input_text") or "").strip()
        tgt = (r.get("target_text") or "").strip()
        if src and tgt:
            pairs.append({"src": src, "tgt_ref": tgt, "lang_pair": "en->lb", "origin": "tatoeba"})
    write_jsonl(out, pairs)
    save_manifest({
        "source": "tatoeba", "dataset": "mbarnig/Tatoeba-en-lb",
        "url": "https://huggingface.co/datasets/mbarnig/Tatoeba-en-lb",
        "license": "Tatoeba CC BY 2.0 FR (derived dataset, quality unverified)",
        "fetched_at": datetime.now(timezone.utc).isoformat(), "rows": len(pairs),
        "note": "PROBE ONLY: template garbage (targets like 'Luxembourgish') and UI strings",
    })
    print(f"tatoeba: {len(pairs)} pairs -> {out}")


def cmd_publiclu(args):
    base = "https://data.public.lu/api/1/datasets"
    if args.list:
        payload = get_json(f"{base}/?q={urllib.parse.quote(args.query)}&page_size=20")
        for d in payload.get("data", []):
            print(f"{d.get('id')}  |  {d.get('title')}")
        return
    ident = args.dataset_id or args.query
    d = get_json(f"{base}/{urllib.parse.quote(ident, safe='')}")
    lic = d.get("license") or {}
    if isinstance(lic, str):
        lic = {"id": lic, "title": lic}
    print(f"title: {d.get('title')}")
    print(f"license: {lic.get('id') or lic.get('title')}")
    res = d.get("resources") or []
    for r in res:
        print(f"  {str(r.get('format')):>8}  {r.get('latest')}")
    if args.download:
        out_dir = OUT_DIR / "publiclu"
        out_dir.mkdir(parents=True, exist_ok=True)
        got = 0
        for r in res:
            url = r.get("latest") or r.get("original_url")
            if not url:
                continue
            if args.only and args.only not in (r.get("format") or "").lower():
                continue
            name = out_dir / (url.rstrip("/").split("/")[-1] or "resource")
            req = urllib.request.Request(url, headers={"User-Agent": "grammetarl/0.1 (academic research)"})
            with urllib.request.urlopen(req, timeout=300) as resp, name.open("wb") as fh:
                fh.write(resp.read())
            got += 1
            print(f"  downloaded -> {name} ({name.stat().st_size} bytes)")
        save_manifest({
            "source": "publiclu", "dataset": d.get("id"), "title": d.get("title"),
            "url": base + "/" + str(d.get("id")),
            "license": lic.get("id") or lic.get("title") or "see dataset page",
            "fetched_at": datetime.now(timezone.utc).isoformat(), "resources": got,
        })


def main():
    p = argparse.ArgumentParser(description="Fetch external en<->lb pairs with licensing")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("luxalign"); a.add_argument("--config", default="lb-en")
    a.add_argument("--limit", type=int, default=43357); a.set_defaults(fn=cmd_luxalign)
    b = sub.add_parser("tatoeba"); b.add_argument("--split", default="train")
    b.add_argument("--limit", type=int, default=2000)
    b.add_argument("--prefix", default="translate english to luxembourgish"); b.set_defaults(fn=cmd_tatoeba)
    c = sub.add_parser("publiclu"); c.add_argument("--list", action="store_true")
    c.add_argument("--query", default="iwwersetzungskorpus"); c.add_argument("--dataset-id")
    c.add_argument("--download", action="store_true"); c.add_argument("--only")
    c.set_defaults(fn=cmd_publiclu)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
