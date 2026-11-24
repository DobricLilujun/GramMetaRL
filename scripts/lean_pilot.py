#!/usr/bin/env python3
"""P6.1 pilot: static formal analysis of the 87 grammar cards + Lean certificates.

Outputs (under artifacts/lean/):
  analysis.json   per-constraint statuses, dead/vacuous/universal lists, unsatisfiable cards,
                  pairwise obligation conflicts, alphabet used
  certificates.lean  kernel-checkable claims (bounded depth stated in the file)
  summary.txt     the numbers, for the report
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grammetarl.formal.card_analysis import analyze_cards, build_alphabet
from grammetarl.formal.cards2lean import emit_lean
from grammetarl.schema import MBGCard


def load_cards(path: Path) -> list[MBGCard]:
    with path.open("r", encoding="utf-8") as f:
        return [MBGCard.model_validate_json(l) for l in f if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards", default="data/rules/mbg_cards.jsonl")
    ap.add_argument("--pairs", default="data/external/pairs_en_lb.jsonl,"
                        "data/processed/tagged_cc0.jsonl")
    ap.add_argument("--out", default="artifacts/lean")
    ap.add_argument("--max-chars", type=int, default=70)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--max-claims", type=int, default=30)
    ap.add_argument("--lean", default=None, help="path to the lean binary; omit to skip kernel check")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cards = load_cards(Path(args.cards))
    pair_paths = [Path(x) for x in args.pairs.split(",") if x]
    missing = [str(x) for x in pair_paths if not x.exists()]
    if missing:
        print(f"[WARN] corpus file(s) missing: {missing}")
    alpha = build_alphabet(pair_paths, max_chars=args.max_chars)
    if len(alpha.chars) < 40:
        print(f"[WARN] alphabet has only {len(alpha.chars)} chars: the empty/universal verdicts "
              "below are artifacts of a degenerate model, not card properties. Point --pairs at "
              "a real corpus.")
    res = analyze_cards(cards, alpha)
    (out / "analysis.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    facts_by_card: dict[str, list[dict]] = {}
    for f in res["facts"]:
        facts_by_card.setdefault(f["card_id"], []).append(f)

    # sub-alphabet for the Lean encoding: only chars appearing in witnesses
    wit = {cid: w for cid, w in res["satisfiable_witnesses"].items() if w}
    chars = sorted({c for w in wit.values() for c in w if c not in ("\x00", "\x01")})
    alpha_map = {c: i + 1 for i, c in enumerate(chars)}
    src, emitted = emit_lean(cards, facts_by_card, alpha, max_claims=args.max_claims)
    (out / "certificates.lean").write_text(src, encoding="utf-8")

    lines = [
        f"cards={len(cards)} constraints={res['constraints']} alphabet={res['alphabet_size']}",
        f"by_status={res['by_status']}",
        f"out_of_fragment_reasons={res['out_of_fragment_reasons']}",
        f"dead_patterns={len(res['cards_dead'])} vacuous_exclusions={len(res['cards_vacuous'])} "
        f"universal={len(res['cards_universal'])} out_of_fragment={len(res['cards_out_of_fragment'])}",
        f"cards_with_formalizable_obligation={res['cards_with_obligation']}",
        f"unsatisfiable_cards={len(res['unsatisfiable_cards'])} {res['unsatisfiable_cards'][:6]}",
        f"pair_conflicts={len(res['pair_conflicts'])} (budget {res['pair_budget_used']} pairs) "
        f"{[c['cards'] for c in res['pair_conflicts'][:6]]}",
        f"lean_automata_emitted={len(emitted)} claims={[e.claims for e in emitted][:5]}",
    ]
    if args.lean:
        proc = subprocess.run([args.lean, str(out / "certificates.lean")],
                              capture_output=True, text=True, timeout=1200)
        lines.append(f"lean_exit={proc.returncode}")
        lines.append("lean_stderr_head=" + (proc.stderr or "")[:900].replace("\n", " | "))
    (out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
