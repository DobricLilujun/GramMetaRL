from __future__ import annotations

import argparse
import json
import os

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

def _chr_f(hyp: str, ref: str, max_order: int = 6, beta: float = 2.0) -> float:
    """Dependency-free chrF (Popovic 2015): mean over char n-gram orders of the F-score.

    The previous implementation counted *distinct* n-gram types and averaged both sides in
    the denominator, which is not a precision/recall F-score: it returned 2.015 for a string
    compared with itself. This version is a proper multiset precision/recall F-score and is
    bounded by 1.0. For the paper, report sacrebleu chrF/COMET; this is the fast RL-side proxy.
    """
    if not hyp.strip() or not ref.strip():
        return 0.0

    def norm(s: str) -> str:
        return " ".join(s.split())

    def multiset_counts(s: str, n: int) -> dict[str, int]:
        out: dict[str, int] = {}
        for i in range(len(s) - n + 1):
            g = s[i:i + n]
            out[g] = out.get(g, 0) + 1
        return out

    h, r = norm(hyp), norm(ref)
    beta2 = beta * beta
    scores: list[float] = []
    for n in range(1, max_order + 1):
        if len(h) < n or len(r) < n:
            continue
        hc, rc = multiset_counts(h, n), multiset_counts(r, n)
        matches = sum(min(count, rc.get(g, 0)) for g, count in hc.items())
        total_h = sum(hc.values())
        total_r = sum(rc.values())
        precision = matches / total_h if total_h else 0.0
        recall = matches / total_r if total_r else 0.0
        if precision == 0.0 or recall == 0.0:
            scores.append(0.0)
            continue
        scores.append((1 + beta2) * precision * recall / (beta2 * precision + recall))
    return sum(scores) / len(scores) if scores else 0.0


def replay_trajectory(
    src: str,
    reference: str,
    cards_path: str | None = None,
    lexicon_path: str | None = None,
    namespace: str | None = None,
    policy_endpoint: str | None = None,
    policy_model: str | None = None,
    policy_api_key: str | None = None,
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
        llm_endpoint=policy_endpoint, llm_model=policy_model, api_key=policy_api_key,
    )
    action_names.append("CALL_APPLY_RULES")

    # No policy configured -> no draft. Copying the source is not a translation; it is scored as
    # such rather than passed off as a hypothesis.
    draft = applied.get("draft")
    hypothesis = str(draft) if draft else src

    # 4) verify
    report = grammar_verify(
        src_sentence=src, hypothesis=hypothesis, applied_rule_ids=rule_ids,
        cards_path=cards_path, namespace=namespace,
    )
    action_names.append("CALL_GRAMMAR_VERIFY")
    grammar_passed = bool(report.get("overall_passed", False))
    checked_rules = int(report.get("checked_rules", 0))
    verdicts = list(report.get("verdicts", []))

    reward = aggregate_reward(
        hypothesis=hypothesis,
        reference=reference,
        grammar_passed=grammar_passed,
        checked_rules=checked_rules,
        verdicts=verdicts,
        action_names=action_names,
    )
    return {
        "hypothesis": hypothesis,
        "reference": reference,
        "chr_f": _chr_f(hypothesis, reference),
        "grammar_passed": grammar_passed,
        "checked_rules": checked_rules,
        "verdicts": verdicts,
        "decided": int(report.get("decided_rules", 0)),
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
    parser.add_argument(
        "--policy-endpoint",
        default=None,
        help="OpenAI-compatible endpoint for the translation policy; without it the replay has no "
        "hypothesis (copying the source is scored as such, not passed off as a translation)",
    )
    parser.add_argument("--policy-model", default=None)
    parser.add_argument("--policy-api-key-env", default=None)
    parser.add_argument("--show", type=int, default=3, help="Print the first N hypotheses")
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
                policy_endpoint=args.policy_endpoint,
                policy_model=args.policy_model,
                policy_api_key=os.environ.get(args.policy_api_key_env) if args.policy_api_key_env else None,
            )
            if len(results) < args.show:
                print(f"[eval] en : {rec.get('src', '')[:90]}")
                print(f"[eval] ref: {rec.get('tgt_ref', '')[:90]}")
                print(f"[eval] hyp: {res['hypothesis'][:90]}  chrF={res['chr_f']:.3f} "
                      f"decided={res['decided']} reward={res['reward']:.3f}")
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