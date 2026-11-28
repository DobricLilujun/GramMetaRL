#!/usr/bin/env python3
"""Per-rule signal analysis for the P1 reward smoke (offline, zero model calls).

Why this exists: the first smoke compared a 2-rule run with an 8-rule run, but the smoke selects
prompts as "pairs where the reference has at least one decided check", so more rules also means a
DIFFERENT prompt set. A difference in nonzero_spread_rate across those two runs mixes two effects.
This script removes the mix: it re-scores the SAME stored hypotheses under different rule subsets and
reports, per rule, whether the rule separates the policy from the professional reference and whether
it separates rollouts inside a group (which is what a GRPO advantage actually needs).

Readout that matters for rule mining:
  policy_violated_rate > reference_violated_rate  -> the rule can teach something
  both ~ 0                                        -> saturated cue, adds coverage but no advantage
  reference_violated_rate high                    -> the rule punishes professionals; not certified

Non-zero spread is NOT the objective, and this script measures why. Excluding the three anti-professional
rules from the reward lowered spread (11.9% -> 7.1% on the same 126 hypotheses): spread counts ANY
within-group reward difference, including differences that point AWAY from the professional reference.
The gradient-alignment block therefore compares, for every within-group rollout pair, the sign of the
reward difference against the sign of the lexical-similarity-to-reference difference. Concordant pairs
mean the reward ranks rollouts the way a professional reference would; discordant pairs mean the reward
is teaching the wrong direction. Lexical F1 is a proxy, not a quality oracle - it says nothing about
grammar - but it is the same proxy an MT reward would use, which is exactly the comparison of interest.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grammetarl.rewards.grammar_reward import grammar_reward_weighted
from grammetarl.rewards.reliability import ReliabilityTable
from grammetarl.schema import MBGCard
from grammetarl.validator import TranslationVerifier


def load_cards(path: Path, only: set[str] | None) -> list[MBGCard]:
    cards = [MBGCard.model_validate_json(l) for l in path.open("r", encoding="utf-8") if l.strip()]
    if only:
        cards = [c for c in cards if c.id in only]
    return cards


def usable_ids(path: Path) -> set[str]:
    d = json.loads(path.read_text("utf-8"))
    if isinstance(d, dict) and "rules" in d:  # older list-shaped artifact
        return {r["rule_id"] for r in d["rules"] if r.get("usable")}
    # the quality matrix is keyed by rule id
    return {rid for rid, v in d.items() if isinstance(v, dict) and v.get("usable")}


def _token_f1(a: str, b: str) -> float:
    """Lexical overlap proxy for 'closer to the professional reference'. Case/diacritic-normalised."""
    na = set(re.findall(r"\w+", TranslationVerifier.normalize_text(a)))
    nb = set(re.findall(r"\w+", TranslationVerifier.normalize_text(b)))
    if not na or not nb:
        return 0.0
    tp = len(na & nb)
    if not tp:
        return 0.0
    p, r = tp / len(na), tp / len(nb)
    return 2 * p * r / (p + r)


def score_set(v: TranslationVerifier, cards: list[MBGCard], rel: ReliabilityTable,
              groups: dict[str, list[dict]]) -> tuple[dict, dict]:
    """groups: prompt_id -> [{hyp, tgt_ref, src}]  -> (per-rule stats, group-level spread)."""
    per_rule: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    spread_flags: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    alignment: collections.Counter = collections.Counter()
    rule_alignment: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    group_rewards: dict[str, list[float]] = {}

    for pid, rows in groups.items():
        verdict_vectors: dict[str, list[tuple[str, str]]] = {}
        rewards: list[float] = []
        for k, row in enumerate(rows):
            rep = v.verify(row["src"], row["hyp"], cards)
            dec = [r for r in rep.results if r.verdict in ("satisfied", "violated")]
            w = rel.weights([r.rule_id for r in dec])
            rewards.append(grammar_reward_weighted(
                [x for x in rep.verdicts() if x in ("satisfied", "violated")], w))
            # keyed by rollout index, not by text: two sampled rollouts can be identical strings
            verdict_vectors[k] = [(r.rule_id, r.verdict) for r in dec]
            # coverage of the verifier itself on this prompt distribution: a rollout with no decided
            # verdict cannot produce advantage, whichever way the rules point.
            alignment["rollouts"] += 1
            if not dec:
                alignment["rollouts_with_no_verdict"] += 1
            for r in dec:
                per_rule[r.rule_id]["policy_decided"] += 1
                per_rule[r.rule_id][f"policy_{r.verdict}"] += 1
            # the professional reference, scored with the same rules and the same card set
            ref = v.verify(row["src"], row["tgt_ref"], cards)
            for r in ref.results:
                if r.verdict not in ("satisfied", "violated"):
                    continue
                per_rule[r.rule_id]["ref_decided"] += 1
                per_rule[r.rule_id][f"ref_{r.verdict}"] += 1
        group_rewards[pid] = rewards
        # gradient alignment: does a reward difference between two rollouts point toward the reference?
        sims = [_token_f1(row["hyp"], row["tgt_ref"]) for row in rows]
        seen0 = list(verdict_vectors.values())
        for a in range(len(rows)):
            for b in range(a + 1, len(rows)):
                dr, ds = rewards[a] - rewards[b], sims[a] - sims[b]
                if dr == 0 and ds == 0:
                    alignment["tie_both"] += 1
                elif dr == 0:
                    alignment["tie_reward"] += 1
                elif ds == 0:
                    alignment["tie_similarity"] += 1
                elif (dr > 0) == (ds > 0):
                    alignment["concordant"] += 1
                else:
                    alignment["discordant"] += 1
                # per-rule: only pairs where THIS rule changed its verdict
                da, db = dict(seen0[a]), dict(seen0[b])
                for rid in set(da) & set(db):
                    if da[rid] == db[rid]:
                        continue
                    if ds == 0:
                        rule_alignment[rid]["tie_similarity"] += 1
                    elif (da[rid] == "satisfied") == (ds > 0):
                        rule_alignment[rid]["concordant"] += 1
                    else:
                        rule_alignment[rid]["discordant"] += 1
        # which rule makes two rollouts in this group differ?
        seen = list(verdict_vectors.values())
        differ: set[str] = set()
        for a in range(len(seen)):
            for b in range(a + 1, len(seen)):
                da, db = dict(seen[a]), dict(seen[b])
                for rid in set(da) & set(db):
                    if da[rid] != db[rid]:
                        differ.add(rid)
                if set(da.items()) != set(db.items()):
                    spread_flags["_group_any"]["differs"] += 1
        for rid in differ:
            spread_flags[rid]["creates_group_spread"] += 1
    return per_rule, {"group_rewards": group_rewards, "spread": spread_flags,
                      "alignment": alignment, "rule_alignment": rule_alignment}


def _binom_two_sided_p(k: int, n: int) -> float:
    """Exact two-sided binomial p-value against H0: concordance = 0.5, i.e. the reward ranks a rollout
    pair independently of the professional reference. At n = 10 decisive pairs, 2 vs 8 gives p ~ 0.11:
    the honest reading is NOT "the reward points the wrong way", it is "we cannot yet tell"."""
    if not n:
        return 1.0
    from math import comb

    pmf = [comb(n, i) / 2 ** n for i in range(n + 1)]
    return round(min(1.0, sum(p for p in pmf if p <= pmf[k] + 1e-12)), 4)


def _wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    """Wilson interval; duplicated from p1_reward_smoke.py on purpose - scripts do not import each other."""
    if not n:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return [round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)]


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-rule GRPO signal analysis of stored rollouts")
    ap.add_argument("--rollout-log", required=True, help="rollouts.jsonl from p1_reward_smoke.py")
    ap.add_argument("--cards", default="data/rules/mbg_cards_condv3.jsonl")
    ap.add_argument("--reliability", default="data/rules/reliability_condv3_fit.json")
    ap.add_argument("--quality", default="artifacts/audit/rule_quality_condv3_heldout.json")
    ap.add_argument("--rule-ids", default="", help="comma list; default = every rule certified usable")
    ap.add_argument("--out", default="artifacts/p1/signal_by_rule.json")
    args = ap.parse_args()

    rel = ReliabilityTable.load(Path(args.reliability))
    ids = set(args.rule_ids.split(",")) if args.rule_ids else usable_ids(Path(args.quality))
    cards = load_cards(Path(args.cards), ids)
    v = TranslationVerifier(require_applicability=True)

    groups: dict[str, list[dict]] = collections.defaultdict(list)
    with Path(args.rollout_log).open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            groups[r["prompt_id"]].append({"src": r["src"], "hyp": r["hyp"], "tgt_ref": r["tgt_ref"],
                                           "k": r.get("rollout", 0)})
    print(f"rollout log: {sum(len(x) for x in groups.values())} hypotheses over {len(groups)} groups")
    max_k = max((len(rows) for rows in groups.values()), default=0)
    if max_k < 2:
        raise SystemExit(
            "this log holds at most one hypothesis per prompt, so within-group spread, alignment and "
            "diversity are all 0/undefined BY CONSTRUCTION. Use such a log only as a mining corpus "
            "(scripts/mine_policy_gaps.py) or for applicability coverage; re-run the smoke with "
            "--rollouts >= 2 for a signal number. (Found the hard way: a 140-prompt --rollouts 1 corpus "
            "was briefly quoted as 'advantage coverage 0.0% on unbiased prompts'.)")
    print(f"rules scored: {len(cards)} {sorted(ids)}")

    # rollout diversity: if the policy emits near-identical rollouts, no rule set can produce a spread,
    # and the diagnosis would be wrong. Measured on 42 groups x 3: mean distinct = 2.93, mean
    # greedy-vs-sampled similarity = 0.81, so text DOES vary and the rules are what fail to see it.
    import difflib
    div = {"groups": len(groups)}
    distinct = [len({row["hyp"] for row in rows}) for rows in groups.values()]
    sims: list[float] = []
    for rows in groups.values():
        rows_sorted = sorted(rows, key=lambda r: r["k"])
        anchor = rows_sorted[0]["hyp"]
        for row in rows_sorted[1:]:
            sims.append(difflib.SequenceMatcher(None, anchor, row["hyp"]).ratio())
    div["mean_distinct_hypotheses_per_group"] = round(sum(distinct) / len(distinct), 3) if distinct else 0
    div["groups_all_rollouts_identical"] = sum(1 for d in distinct if d == 1)
    div["mean_similarity_rollout0_vs_rest"] = round(sum(sims) / len(sims), 3) if sims else 0
    print(f"rollout diversity: distinct/group={div['mean_distinct_hypotheses_per_group']} "
          f"all_identical_groups={div['groups_all_rollouts_identical']} "
          f"similarity(greedy vs sampled)={div['mean_similarity_rollout0_vs_rest']}")

    per_rule, g = score_set(v, cards, rel, groups)
    n_groups = len(groups)
    print(f"\n{'rule':26} {'pol_n':>6} {'pol_viol':>9} {'ref_n':>6} {'ref_viol':>9} {'gap':>7} {'spreads':>8}")
    rows = []
    for rid in sorted(per_rule, key=lambda r: -per_rule[r]["policy_decided"]):
        c = per_rule[rid]
        pv = c["policy_violated"] / c["policy_decided"] if c["policy_decided"] else None
        rv = c["ref_violated"] / c["ref_decided"] if c["ref_decided"] else None
        gap = (pv - rv) if (pv is not None and rv is not None) else None
        sp = g["spread"][rid]["creates_group_spread"]
        print(f"{rid:26} {c['policy_decided']:6d} {pv*100 if pv is not None else 0:8.1f}% "
              f"{c['ref_decided']:6d} {rv*100 if rv is not None else 0:8.1f}% "
              f"{gap*100 if gap is not None else 0:6.1f}% {sp:5d}/{n_groups}")
        rows.append({"rule_id": rid, "policy_decided": c["policy_decided"],
                     "policy_violated_rate": pv, "ref_decided": c["ref_decided"],
                     "ref_violated_rate": rv, "gap": gap, "creates_group_spread": sp,
                     "align_concordant": g["rule_alignment"][rid]["concordant"],
                     "align_discordant": g["rule_alignment"][rid]["discordant"],
                     "align_tie_similarity": g["rule_alignment"][rid]["tie_similarity"]})

    nz = sum(1 for pid, rw in g["group_rewards"].items() if max(rw) != min(rw))
    print(f"\ngroups with nonzero spread: {nz}/{n_groups} = {nz/n_groups:.1%}")
    al = g["alignment"]
    dec = al["concordant"] + al["discordant"]
    conc = al["concordant"] / dec if dec else 0.0
    cov = al["rollouts"] - al["rollouts_with_no_verdict"]
    print(f"verifier coverage on this prompt set: {cov}/{al['rollouts']} rollouts reached at least one "
          f"verdict ({cov / max(1, al['rollouts']):.1%}); the rest cannot carry advantage at all")
    al["rollouts_with_a_verdict"] = cov
    al["coverage_rate"] = round(cov / max(1, al["rollouts"]), 4)
    al_p = _binom_two_sided_p(al["concordant"], dec)
    print(f"gradient alignment (reward diff vs similarity-to-reference diff, within group): "
          f"concordant={al['concordant']} discordant={al['discordant']} "
          f"-> {conc:.1%} [95% CI {_wilson(al['concordant'], dec)}; p vs 0.5 = {al_p}] "
          f"ties: reward={al['tie_reward']} similarity={al['tie_similarity']} both={al['tie_both']}")
    al["concordant_p_vs_0.5"] = al_p
    if dec and al_p > 0.05:
        print("  n too small to call the direction: this is NOT evidence that the reward points the "
              "wrong way, it is evidence that the axis was unmeasured.")
    print("  per-rule alignment (only pairs where that rule's verdict differs):")
    for rid in sorted(g["rule_alignment"], key=lambda r: -(g["rule_alignment"][r]["concordant"] + g["rule_alignment"][r]["discordant"])):
        ra = g["rule_alignment"][rid]
        d2 = ra["concordant"] + ra["discordant"]
        if not d2:
            continue
        print(f"    {rid:26} concordant={ra['concordant']:3d} discordant={ra['discordant']:3d} "
              f"tie_sim={ra['tie_similarity']:3d} -> {ra['concordant']/d2:.0%}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "rollout_log": args.rollout_log, "rules_scored": sorted(ids), "groups": n_groups,
        "groups_nonzero_spread": nz, "nonzero_spread_rate": nz / n_groups,
        "nonzero_spread_rate_ci95": _wilson(nz, n_groups),
        "rollout_diversity": div, "per_rule": rows,
        "alignment": {"concordant": al["concordant"], "discordant": al["discordant"],
                      "tie_reward": al["tie_reward"], "tie_similarity": al["tie_similarity"],
                      "tie_both": al["tie_both"], "concordant_rate": round(conc, 4),
                      "concordant_rate_ci95": _wilson(al["concordant"], dec),
                      "concordant_p_vs_0.5": al["concordant_p_vs_0.5"],
                      "null": "concordant_rate 0.5 = the reward ranks a rollout pair independently of "
                              "the professional reference; anything else needs the p-value to mean a thing"},
        "verifier_coverage": {"rollouts": al["rollouts"],
                              "rollouts_with_a_verdict": al["rollouts_with_a_verdict"],
                              "coverage_rate": al["coverage_rate"]},
    }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
