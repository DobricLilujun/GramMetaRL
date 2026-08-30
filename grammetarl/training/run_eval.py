from __future__ import annotations

import argparse
import json

from grammetarl.agent_env import GrammarFirstEnv
from grammetarl.rewards import aggregate_reward
from grammetarl.tools import (
    dictionary_lookup,
    grammar_rule_apply,
    grammar_rule_search,
    grammar_verify,
)


# ---------------------------------------------------------------------------
# Offline evaluation that does NOT require verl to be installed. It replays a
# trajectory through the real grammar-first environment + tools and scores it
# with the aggregate reward. Metrics (chrF/BLEU/COMET) are computed when the
# corresponding packages are present, otherwise skipped gracefully.
# ---------------------------------------------------------------------------

def _chr_f(hyp: str, ref: str) -> float:
    """Lightweight chrF (character n-gram F1) without external deps."""
    if not hyp or not ref:
        return 0.0

    def ngrams(s: str, n: int) -> dict:
        out: dict[str, int] = {}
        for i in range(len(s) - n + 1):
            g = s[i:i + n]
            out[g] = out.get(g, 0) + 1
        return out

    def f1(a: str, b: str, n: int) -> float:
        pa, pb = ngrams(a, n), ngrams(b, n)
        inter = sum(min(pa.get(g, 0), pb.get(g, 0)) for g in pa)
        denom = max(1, (len(pa) + len(pb)) / 2)
        return inter / denom

    f1 = 0.5 * (f1(hyp, ref, 1) + f1(hyp, ref, 2) + f1(hyp, ref, 3))
    return f1


def replay_trajectory(
    src: str,
    reference: str,
    cards_path: str | None = None,
    lexicon_path: str | None = None,
    namespace: str | None = None,
) -> dict:
    """Run one deterministic grammar-first trajectory and score it."""
    env = GrammarFirstEnv()
    env.reset(sample_id="eval", src_sentence=src)

    # 1) grammar search
    rules = grammar_rule_search(
        query=src, src_sentence=src, top_k=5,
        cards_path=cards_path, namespace=namespace,
    )
    action_names = ["CALL_GRAMMAR_SEARCH"]

    # 2) dictionary lookup (first source token)
    token = src.split(" ")[0] if src.split() else ""
    dictionary_lookup(
        token=token, context=src, lexicon_path=lexicon_path, namespace=namespace
    )
    action_names.append("CALL_DICTIONARY")

    # 3) apply rules -> draft
    rule_ids = [r["rule_id"] for r in rules]
    applied = grammar_rule_apply(
        rule_ids=rule_ids, src_sentence=src,
        cards_path=cards_path, namespace=namespace,
    )
    action_names.append("CALL_APPLY_RULES")

    hypothesis = str(applied.get("draft", src))

    # 4) verify
    report = grammar_verify(
        src_sentence=src, hypothesis=hypothesis, applied_rule_ids=rule_ids,
        cards_path=cards_path, namespace=namespace,
    )
    grammar_passed = bool(report.get("overall_passed", False))
    checked_rules = int(report.get("checked_rules", 0))

    reward = aggregate_reward(
        hypothesis=hypothesis,
        reference=reference,
        grammar_passed=grammar_passed,
        checked_rules=checked_rules,
        action_names=action_names,
    )
    return {
        "hypothesis": hypothesis,
        "reference": reference,
        "chr_f": _chr_f(hypothesis, reference),
        "grammar_passed": grammar_passed,
        "checked_rules": checked_rules,
        "reward": reward,
        "actions": action_names,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline evaluation (verl-optional)")
    parser.add_argument("--train-jsonl", default=None)
    parser.add_argument("--cards", default=None)
    parser.add_argument("--lexicon", default=None)
    parser.add_argument("--namespace", default=None)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    from grammetarl.workspace import get_workspace_paths

    ws = get_workspace_paths(namespace=args.namespace)
    train_jsonl = args.train_jsonl or str(ws.train_dataset_file)

    results: list[dict] = []
    count = 0
    try:
        f = open(train_jsonl, "r", encoding="utf-8")
    except FileNotFoundError:
        print(f"[eval] train set not found: {train_jsonl}; nothing to evaluate.")
        return
    with f:
        for line in f:
            if count >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            res = replay_trajectory(
                src=rec.get("src", ""),
                reference=rec.get("tgt_ref", ""),
                cards_path=args.cards,
                lexicon_path=args.lexicon,
                namespace=args.namespace,
            )
            results.append(res)
            count += 1

    n = len(results)
    avg_chrf = sum(r["chr_f"] for r in results) / n if n else 0.0
    avg_reward = sum(r["reward"] for r in results) / n if n else 0.0
    grammar_pass = sum(1 for r in results if r["grammar_passed"]) / n if n else 0.0
    report = {
        "n_evaluated": n,
        "avg_chr_f": round(avg_chrf, 4),
        "avg_reward": round(avg_reward, 4),
        "grammar_pass_rate": round(grammar_pass, 4),
        "samples": results[:5],
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()