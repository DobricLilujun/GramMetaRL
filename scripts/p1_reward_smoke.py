#!/usr/bin/env python3
"""P1 reward-side smoke: does the grammar reward produce a usable LEARNING SIGNAL?

verl is deliberately not installed in this environment, so this does not train anything. It runs the
part of GRPO that our contribution actually changes - the reward - on real policy rollouts, and
answers the three questions that decide whether an RL run is worth launching:

  1. Degeneracy. GRPO is group-relative: a group whose rollouts all get the same reward contributes
     ZERO advantage, hence zero gradient. If most groups are degenerate, no amount of data helps.
  2. Resolution. Does the reward order rollouts of the SAME source differently, and does the ordering
     track something real (a length/quality proxy), or is it dominated by abstention?
  3. Noise floor. The professional human reference translation is scored with the same function. It
     should score well; every point it loses is measured rule error, not policy error.

Rewards use ONLY rules certified usable by scripts/rule_quality_matrix.py (agreement AND
discrimination AND not self-restating AND above the support floor), weighted by the measured
reliability table. That gate is what stopped a rule with 5.3% discrimination (a constant reward)
from manufacturing a clean-looking signal.
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grammetarl.llm_http import chat_text
from grammetarl.rewards.grammar_reward import grammar_reward_weighted
from grammetarl.rewards.reliability import ReliabilityTable
from grammetarl.schema import MBGCard
from grammetarl.validator import TranslationVerifier

SYSTEM = (
    "You are a professional English to Luxembourgish translator. Translate the user's sentence into "
    "natural, correct Luxembourgish. Output ONLY the translation, no notes, no alternatives."
)


def main() -> None:
    ap = argparse.ArgumentParser(description="GRPO reward-side smoke on policy rollouts")
    ap.add_argument("--cards", default="data/rules/mbg_cards_condonly.jsonl")
    ap.add_argument("--reliability", default="data/rules/reliability_condonly.json")
    ap.add_argument("--quality", default="artifacts/audit/rule_quality_conditional.json",
                    help="three-axis matrix; only usable rules may contribute reward")
    ap.add_argument("--pairs", default="data/external/pairs_en_lb.jsonl")
    ap.add_argument("--source-filter", default="techgov_lu_en")
    ap.add_argument("--prompts", type=int, default=40)
    ap.add_argument("--rollouts", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--policy-endpoint", required=True)
    ap.add_argument("--policy-model", required=True)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--out", default="artifacts/p1/reward_smoke.json")
    ap.add_argument("--rollout-log", default="artifacts/p1/rollouts.jsonl")
    args = ap.parse_args()

    qm = json.loads(Path(args.quality).read_text("utf-8"))
    usable = {k for k, v in qm.items() if v.get("usable")}
    if not usable:
        raise SystemExit("no rule is certified usable; run scripts/rule_quality_matrix.py first")

    cards = [c for c in (MBGCard.model_validate_json(l)
                         for l in Path(args.cards).open("r", encoding="utf-8") if l.strip())
             if c.id in usable]
    rel = ReliabilityTable.load(args.reliability)
    v = TranslationVerifier(require_applicability=True)
    print(f"certified rules: {sorted(c.id for c in cards)}")

    # candidate prompts: the rule must be decidable for the pair, i.e. at least one rollout can matter
    prompts: list[dict] = []
    with Path(args.pairs).open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if args.source_filter and rec.get("source") != args.source_filter:
                continue
            rep = v.verify(rec["src"], rec["tgt_ref"], cards)
            if rep.decided:
                prompts.append(rec)
                if len(prompts) >= args.prompts:
                    break

    print(f"prompts={len(prompts)} rollouts/prompt={args.rollouts}")
    groups: list[dict] = []
    log_path = Path(args.rollout_log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tally: collections.Counter = collections.Counter()

    with log_path.open("w", encoding="utf-8") as logf:
        for pi, rec in enumerate(prompts):
            rewards: list[float] = []
            det_counts: collections.Counter = collections.Counter()
            for k in range(args.rollouts):
                try:
                    text, _meta = chat_text(
                        args.policy_endpoint, args.policy_model,
                        [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": rec["src"]}],
                        api_key=args.api_key, timeout=args.timeout,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature if k else 0.0,
                    )
                except Exception as exc:  # endpoint failure is data, not a crash
                    tally["rollout_error"] += 1
                    print(f"  [warn] rollout failed: {type(exc).__name__}: {exc}")
                    continue
                hyp = (text or "").strip().replace("\n", " ")
                tally["rollout_empty" if not hyp else "rollout_ok"] += 1
                rep = v.verify(rec["src"], hyp, cards)
                w = rel.weights([r.rule_id for r in rep.results if r.verdict in ("satisfied", "violated")])
                r_gh = grammar_reward_weighted(
                    [x for x in rep.verdicts() if x in ("satisfied", "violated")], w)
                rewards.append(r_gh)
                det_counts[rep.results[0].verdict] += 1
                det_counts["decided" if rep.decided else "abstain"] += 1
                logf.write(json.dumps({"prompt_id": rec["id"], "rollout": k, "src": rec["src"],
                                       "hyp": hyp, "tgt_ref": rec["tgt_ref"],
                                       "verdicts": rep.verdicts(), "rule_ids": [
                                           r.rule_id for r in rep.results],
                                       "r_grammar": round(r_gh, 6)}, ensure_ascii=False) + "\n")

            # reference control: the professional translation, same reward function
            rep_ref = v.verify(rec["src"], rec["tgt_ref"], cards)
            w_ref = rel.weights([r.rule_id for r in rep_ref.results
                                 if r.verdict in ("satisfied", "violated")])
            r_ref = grammar_reward_weighted(
                [x for x in rep_ref.verdicts() if x in ("satisfied", "violated")], w_ref)
            det_counts["ref" if r_ref >= 0 else "ref_penalized"] += 1

            if not rewards:
                tally["group_no_rollouts"] += 1
                continue
            spread = max(rewards) - min(rewards)
            groups.append({
                "prompt_id": rec["id"], "rewards": rewards, "spread": spread,
                "mean": statistics.fmean(rewards), "r_ref": r_ref,
                "abstain_rollouts": det_counts["abstain"],
                "verdicts": dict(det_counts),
            })
            tally["group_nonzero_spread" if spread > 1e-9 else "group_zero_spread"] += 1
            tally["ref_penalized" if r_ref < 0 else "ref_ok"] += 1
            print(f"  [{pi + 1}/{len(prompts)}] spread={spread:6.3f} mean={statistics.fmean(rewards):6.3f} "
                  f"r_ref={r_ref:6.3f} abstain={det_counts['abstain']}/{args.rollouts}")

    n = len(groups)
    if not n:
        raise SystemExit("no groups collected; endpoint or rule gating failed")
    spreads = [g["spread"] for g in groups]
    means = [g["mean"] for g in groups]
    res = {
        "cards_used": sorted(c.id for c in cards),
        "prompts": n,
        "rollouts_per_prompt": args.rollouts,
        "policy_model": args.policy_model,
        "rollout_tally": dict(tally),
        "groups_nonzero_spread": tally["group_nonzero_spread"],
        "nonzero_spread_rate": tally["group_nonzero_spread"] / n,
        "mean_spread": statistics.fmean(spreads),
        "median_spread": statistics.median(spreads),
        "reward_mean": statistics.fmean(means),
        "reward_stdev_across_groups": statistics.pstdev(means) if n > 1 else 0.0,
        "reward_min": min(means), "reward_max": max(means),
        "reference_penalized_rate": tally["ref_penalized"] / n,
        "reference_mean_reward": statistics.fmean([g["r_ref"] for g in groups]),
        "policy_minus_reference": statistics.fmean(means) - statistics.fmean([g["r_ref"] for g in groups]),
        "abstain_rollout_rate": sum(g["abstain_rollouts"] for g in groups) / (n * args.rollouts),
    }
    print("\n" + json.dumps(res, ensure_ascii=False, indent=2))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), "utf-8")
    print(f"rollouts -> {args.rollout_log}\n-> {args.out}")


if __name__ == "__main__":
    main()
