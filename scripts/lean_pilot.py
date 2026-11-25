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
    ap.add_argument(
        "--lean",
        default=None,
        help="lean binary; default $LEAN_BIN, else the pinned install under "
             "tools/ or ~/tools (see README 'Formal layer' section); pass '' to skip the kernel check",
    )
    args = ap.parse_args()
    if args.lean is None:
        import os
        cand = [os.environ.get("LEAN_BIN", "")]
        for root in (Path(__file__).resolve().parents[1] / "tools",
                     Path.home() / "tools", Path("/raid/yewei/tools")):
            if root.exists():
                cand += [str(x) for x in sorted(root.glob("lean-*/bin/lean"))]
        args.lean = next((c for c in cand if c and Path(c).exists()), None)
        if not args.lean:
            print("[WARN] no lean binary found: emitting certificates but skipping the kernel check")

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
    src, emitted, case_blocks = emit_lean(cards, facts_by_card, alpha, max_claims=args.max_claims)
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
        # Per-automaton kernel check: one file per card-constraint so a rejection names the case
        # instead of just stopping the first file. The exit code is read from lean itself, never
        # from a pipe (an earlier version measured `head`, which always returns 0).
        prelude = src.split('-- card ')[0]
        blocks = case_blocks
        cases = out / "cases"
        cases.mkdir(exist_ok=True)
        npass = nfail = 0
        failures = []
        for i, blk in enumerate(blocks):
            f = cases / f"case_{i:03d}.lean"
            f.write_text(prelude + "\n" + blk + "\nend GramMetaRL\n", encoding="utf-8")
            try:
                proc = subprocess.run([args.lean, str(f)], capture_output=True, text=True, timeout=300)
            except subprocess.TimeoutExpired:
                nfail += 1
                failures.append({"case": i, "why": "kernel timeout"})
                continue
            if proc.returncode == 0:
                npass += 1
            else:
                nfail += 1
                head = (proc.stderr or "").strip().split("\n")[:2]
                failures.append({"case": i, "why": " | ".join(head)[:260]})
        lines.append(f"lean_cases={npass + nfail} kernel_pass={npass} kernel_fail={nfail}")
        lines.append("kernel_failures=" + json.dumps(failures[:8], ensure_ascii=False))
    (out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
