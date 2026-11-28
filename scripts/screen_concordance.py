#!/usr/bin/env python3
"""Screen mined candidates by whether their verdicts RANK rollouts like the professional reference.

Why the axis changed (evidence 15.6-15.7): the previous screening axis was the policy-vs-reference
violation GAP. Gap says the policy differs from the reference; it says nothing about whether the
constraint ranks two candidate translations the way a professional would. Measured on 42 stratified
groups of gemma3:12b rollouts, the 8 certified rules are blind on 89 of 120 within-group rollout pairs,
and on the 10 decisive pairs the grammar reward opposed the professional ranking 8 times out of 10
(20.0% concordance). Gate G31 turns that measurement into a gate, and this script applies the same
measurement to CANDIDATES before they ever become cards.

For every within-group pair of stored rollouts of the same source:
  applicability      = the source cue fires (source-side only, as everywhere else in this repo)
  verdict            = the target-side value regex matches (or, for negate, does not match)
  decisive pair      = the two rollouts get DIFFERENT verdicts under this constraint
  concordant         = the rollout with the satisfied verdict is also the one lexically closer to the
                       professional reference
  discordant         = the opposite

Concordance is bounded by the similarity proxy: token F1 says nothing about grammar, so a constraint
that agrees with lexical overlap may simply be tracking length or word choice. That is why the script
reports the agreement-vs-professionals axis and the held-out half in the same row - a candidate is
interesting only if BOTH axes are good, and it still has to pass the normal three-axis certification
before it becomes a card.

Cost: zero model calls. Regex + set arithmetic over stored rollouts and the pool.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grammetarl.validator import TranslationVerifier

WORD = re.compile(r"\w+", re.UNICODE)


def norm(text: str) -> str:
    return TranslationVerifier.normalize_text(text)


def tokset(text: str) -> set[str]:
    return set(WORD.findall(norm(text)))


def token_f1(a: str, b: str) -> float:
    na, nb = tokset(a), tokset(b)
    if not na or not nb:
        return 0.0
    tp = len(na & nb)
    if not tp:
        return 0.0
    p, r = tp / len(na), tp / len(nb)
    return 2 * p * r / (p + r)


def _wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if not n:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return [round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)]


def _binom_two_sided_p(k: int, n: int) -> float:
    """Exact two-sided binomial p-value for H0: the constraint's verdict is INDEPENDENT of which
    rollout is lexically closer to the reference, i.e. concordance = 0.5.

    Without this, "20% concordance" reads like a finding at n = 10 when it is inside the noise. Every
    concordance number in this repo is reported next to its p-value for that reason.
    """
    if not n:
        return 1.0
    from math import comb

    pmf = [comb(n, i) / 2 ** n for i in range(n + 1)]
    return round(min(1.0, sum(p for p in pmf if p <= pmf[k] + 1e-12)), 4)


def _cluster_concordance_ci(per_group: list[tuple[int, int]], reps: int = 400) -> list[float]:
    """Resample GROUPS (not pairs) with replacement.

    Pairs inside one group share a prompt and a policy, so a Wilson interval over pairs overstates
    power; the exchangeable unit here is the group.
    """
    import random

    g = [(c, c + d) for c, d in per_group if c + d]
    if not g:
        return [0.0, 0.0]
    rng = random.Random(1729)
    rates: list[float] = []
    for _ in range(reps):
        kc = nc = 0
        for _ in range(len(g)):
            c, n = g[rng.randrange(len(g))]
            kc += c
            nc += n
        if nc:
            rates.append(kc / nc)
    if not rates:
        return [0.0, 0.0]
    rates.sort()
    return [round(rates[int(0.025 * len(rates))], 3), round(rates[int(0.975 * (len(rates) - 1))], 3)]


def load_pool(path: Path, source_filter: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if source_filter and rec.get("source") != source_filter:
                continue
            if not rec.get("src") or not rec.get("tgt_ref"):
                continue
            rows.append((norm(rec["src"]), norm(rec["tgt_ref"])))
    return rows


def load_groups(path: Path) -> list[dict]:
    """prompt_id -> {src, ref, [hyp,...]} from a rollout log."""
    groups: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            src, hyp, ref = rec.get("src"), (rec.get("hyp") or "").strip(), (rec.get("tgt_ref") or "").strip()
            if not src or not hyp or not ref:
                continue
            g = groups.setdefault(rec.get("prompt_id", src), {"src": src, "ref": ref, "hyps": []})
            if hyp not in g["hyps"]:
                g["hyps"].append(hyp)
    return [{"src": norm(g["src"]), "ref": g["ref"], "hyp_norms": [norm(h) for h in g["hyps"]]}
            for g in groups.values()]


def _fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Exact two-sided Fisher test for two concordance counts: (a concordant, b discordant) for the
    candidate versus (c, d) pooled over its placebos.

    The 0.5 null is the WRONG reference for this metric: because the similarity proxy is lexical, placebo
    constraints (same cue, unrelated target tokens) land anywhere from 0.14 to 0.83 concordant on the same
    groups. The candidate has to beat ITS OWN placebo band, which is what this tests.
    """
    from math import lgamma

    n1, n2 = a + b, c + d
    tot, k1 = n1 + n2, a + c
    if n1 == 0 or n2 == 0 or tot == 0:
        return 1.0

    def lcomb(nn, kk):
        return lgamma(nn + 1) - lgamma(kk + 1) - lgamma(nn - kk + 1)

    lo, hi = max(0, k1 - n2), min(k1, n1)
    x = a
    if not (lo <= x <= hi):
        return 1.0
    p_obs = lcomb(n1, x) + lcomb(n2, k1 - x)
    total = 0.0
    probs = []
    for i in range(lo, hi + 1):
        lp = lcomb(n1, i) + lcomb(n2, k1 - i)
        probs.append(lp)
        total += _exp(lp)
    return round(min(1.0, sum(_exp(lp) for lp in probs if lp <= p_obs + 1e-9) / max(total, 1e-300)), 4)


def _exp(x: float) -> float:
    from math import exp

    return exp(x) if x > -700 else 0.0


def placebo_specs(spec: dict, pool: list[tuple[str, str]], groups: list[dict], n: int,
                  min_count: int = 2) -> list[dict]:
    """Control for the one thing the alignment proxy CANNOT see: it is LEXICAL.

    Token F1 to the reference rewards sharing surface tokens with the professional translation, so a
    constraint that merely requires (or forbids) ANY frequent target token can score concordant without
    encoding anything grammatical. A placebo keeps the source cue and the constraint kind but swaps the
    target form for unrelated target tokens, so the placebo band says what lexical bookkeeping alone
    achieves on the same groups. A candidate counts as aligned only if it beats that band.

    The vocabulary is deliberately drawn from where the candidate lives - tokens that occur in the
    references of cue-applicable pairs AND in the rollouts for that cue. Sampling any token in the
    language produces placebos that never fire, and a placebo with no decisive pairs measures nothing.
    """
    import random

    src_pat = re.compile(spec["source_pattern"], re.I)
    bad = set(spec["values"])

    ref_counts: collections.Counter = collections.Counter()
    for src, ref in pool:
        if src_pat.search(src):
            ref_counts.update(re.findall(r"\w{3,}", ref))
    hyp_seen: set[str] = set()
    for g in groups:
        if src_pat.search(g["src"]):
            for h in g["hyp_norms"]:
                hyp_seen.update(re.findall(r"\w{3,}", h))
    vocab = [t for t, c in ref_counts.items()
             if c >= min_count and t in hyp_seen and f"\\b{t}\\b" not in bad]
    if len(vocab) < 3:
        return []
    rng = random.Random(abs(hash(spec["name"])) % 2 ** 32)
    out = []
    for i in range(n):
        pick = rng.sample(vocab, k=min(3, len(vocab)))
        out.append({"name": f"{spec['name']}__placebo{i}", "source_pattern": spec["source_pattern"],
                    "values": [rf"\b{re.escape(p)}\b" for p in pick], "negate": spec.get("negate", False)})
    return out


def score_candidate(spec: dict, pool: list[tuple[str, str]], groups: list[dict],
                    fit_boundary: int) -> dict:
    """Score one candidate constraint: agreement with professionals (both halves) and alignment on the
    stored rollouts. Separated from main() so the direction logic can be unit-tested - it was wrong
    twice in this session (once for exclusion candidates, where `negate` inverts what satisfied means)."""
    src_pat = re.compile(spec["source_pattern"], re.I)
    neg = bool(spec.get("negate"))
    vals = [re.compile(v, re.I) for v in spec["values"]]

    def verdict(target_norm: str) -> bool:
        """True = satisfied. For an exclusion the forbidden shape must be ABSENT."""
        hit = any(x.search(target_norm) for x in vals)
        return (not hit) if neg else hit

    fit_n = fit_ok = ho_n = ho_ok = 0
    for i, (src, ref) in enumerate(pool):
        if not src_pat.search(src):
            continue
        ok = verdict(ref)
        if i < fit_boundary:
            fit_n += 1
            fit_ok += ok
        else:
            ho_n += 1
            ho_ok += ok

    conc = disc = tie_sim = decided = 0
    applicable = 0
    groups_diverse = 0
    per_group: list[tuple[int, int]] = []
    for g in groups:
        if not src_pat.search(g["src"]):
            continue
        applicable += 1
        v = [verdict(h) for h in g["hyp_norms"]]
        decided += sum(1 for x in v if x) if not neg else sum(1 for x in v if not x)
        gc = gd = gtot = 0
        for a in range(len(v)):
            for b in range(a + 1, len(v)):
                if v[a] == v[b]:
                    continue
                gtot += 1
                ds = g["sims"][a] - g["sims"][b]
                if ds == 0:
                    tie_sim += 1
                else:
                    # verdict True = SATISFIED for both kinds (an exclusion is satisfied when the
                    # forbidden shape is absent), so the constraint prefers the True side.
                    ok = (v[a] > v[b]) == (ds > 0)
                    conc += ok
                    disc += not ok
                    gc += ok
                    gd += not ok
        if gtot:
            groups_diverse += 1
        if gc + gd:
            per_group.append((gc, gc + gd))

    dec = conc + disc
    return {
        "name": spec.get("name") or spec["source_pattern"],
        "negate": neg, "source_pattern": spec["source_pattern"], "values": spec["values"],
        "fit_n": fit_n, "fit_agreement": round(fit_ok / fit_n, 4) if fit_n else None,
        "heldout_n": ho_n, "heldout_agreement": round(ho_ok / ho_n, 4) if ho_n else None,
        "policy_decided": decided, "applicable_groups": applicable,
        "groups_with_diversity": groups_diverse,
        "diversity_rate": round(groups_diverse / applicable, 4) if applicable else 0.0,
        "decisive_pairs": dec, "concordant": conc, "discordant": disc,
        "tie_similarity": tie_sim,
        "concordance": round(conc / dec, 4) if dec else 0.0,
        "concordance_ci95": _wilson(conc, dec),
        "concordance_p_vs_0.5": _binom_two_sided_p(conc, dec),
        "concordance_ci95_cluster": _cluster_concordance_ci(per_group),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="concordance screening of mined constraint candidates")
    ap.add_argument("--specs", required=True, help="candidate specs from scripts/mine_policy_gaps.py")
    ap.add_argument("--rollout-log", required=True, help="rollout log from scripts/p1_reward_smoke.py")
    ap.add_argument("--pairs", default="data/external/pairs_en_lb.jsonl")
    ap.add_argument("--source-filter", default="techgov_lu_en")
    ap.add_argument("--fit-boundary", type=int, default=4000,
                    help="first N pool pairs = the half the miner saw; agreement is reported for both")
    ap.add_argument("--min-fit-agree", type=float, default=0.65)
    ap.add_argument("--min-heldout-agree", type=float, default=0.65)
    ap.add_argument("--min-decisive", type=int, default=10, help="within-group pairs with differing verdicts")
    ap.add_argument("--min-concordance", type=float, default=0.5)
    ap.add_argument("--min-placebo-gap", type=float, default=0.1,
                    help="candidate concordance must exceed the pooled placebo concordance by this much")
    ap.add_argument("--min-placebo-pairs", type=int, default=10,
                    help="the placebo control is only reported when the placebos themselves produced "
                         "this many decisive pairs; a placebo that never fires measures nothing")
    ap.add_argument("--max-placebo-p", type=float, default=0.05,
                    help="Fisher exact p for candidate vs placebo concordance counts")
    ap.add_argument("--pool-rows", type=int, default=0, help="cap pool rows (0 = all)")
    ap.add_argument("--placebo", type=int, default=20,
                    help="placebo constraints per candidate (same cue, unrelated target tokens). 0 "
                         "disables the lexical-proxy control.")
    ap.add_argument("--out", default="artifacts/p1/concordance_screen.json")
    args = ap.parse_args()

    specs = json.loads(Path(args.specs).read_text("utf-8"))
    cands = specs["candidates"] if isinstance(specs, dict) else specs
    pool = load_pool(Path(args.pairs), args.source_filter)
    if args.pool_rows:
        pool = pool[:args.pool_rows]
    groups = load_groups(Path(args.rollout_log))
    n_hyp = sum(len(g["hyp_norms"]) for g in groups)
    print(f"pool {len(pool)} pairs (fit half = first {args.fit_boundary}), "
          f"rollouts {n_hyp} hypotheses over {len(groups)} groups, candidates {len(cands)}")

    # precompute per-group similarity to the professional reference once per hypothesis
    for g in groups:
        g["sims"] = [token_f1(h, g["ref"]) for h in g["hyp_norms"]]

    out_rows: list[dict] = []
    for spec in cands:
        row = score_candidate(spec, pool, groups, args.fit_boundary)
        if args.placebo:
            pl = [score_candidate(x, pool, groups, args.fit_boundary)
                  for x in placebo_specs(spec, pool, groups, args.placebo)]
            pc = sum(x["concordant"] for x in pl)
            pd = sum(x["discordant"] for x in pl)
            concs = [x["concordance"] for x in pl if x["decisive_pairs"]]
            row["placebo_runs"] = len(pl)
            row["placebo_decisive_pairs"] = pc + pd
            row["placebo_concordance_max"] = max(concs) if concs else None
            if pc + pd >= args.min_placebo_pairs:
                # An advantage over the placebo band must be both large enough and real: the candidate's
                # concordant/discordant counts against the pooled placebo counts on the SAME groups.
                row["placebo_pooled_concordance"] = round(pc / (pc + pd), 3)
                row["placebo_gap"] = round(row["concordance"] - row["placebo_pooled_concordance"], 3)
                row["placebo_p"] = _fisher_two_sided(row["concordant"], row["discordant"], pc, pd)
            else:
                # too few placebo decisive pairs to say anything; that is neither a pass nor a fail
                row["placebo_pooled_concordance"] = None
                row["placebo_gap"] = None
                row["placebo_p"] = None
        why: list[str] = []
        if row["fit_n"] < 30 or row["heldout_n"] < 30:
            why.append(f"support fit={row['fit_n']} heldout={row['heldout_n']}")
        if row["fit_agreement"] is None or row["fit_agreement"] < args.min_fit_agree:
            why.append(f"fit agreement {row['fit_agreement']}")
        if row["heldout_agreement"] is not None and row["heldout_agreement"] < args.min_heldout_agree:
            why.append(f"held-out agreement {row['heldout_agreement']}")
        if row["decisive_pairs"] < args.min_decisive:
            why.append(f"decisive pairs {row['decisive_pairs']}")
        if row["concordance"] < args.min_concordance:
            why.append(f"concordance {row['concordance']:.2f}")
        # Rejected for LACK OF EVIDENCE is a different verdict from rejected BY evidence, and the row
        # has to say which one it is: a candidate with 2 decisive pairs tells us nothing either way.
        if 0 < row["decisive_pairs"] < args.min_decisive and row["concordance_p_vs_0.5"] > 0.05:
            why.append("alignment not distinguishable from chance")
        # The alignment proxy is lexical; beating the 0.5 null is NOT enough if unrelated target tokens
        # score just as well on the same groups.
        if row.get("placebo_gap") is not None and row["placebo_gap"] < args.min_placebo_gap:
            why.append(f"no better than placebo ({row['concordance']:.2f} vs "
                       f"{row['placebo_pooled_concordance']:.2f})")
        if row.get("placebo_p") is not None and row["placebo_p"] > args.max_placebo_p:
            why.append(f"placebo difference not significant (p={row['placebo_p']})")
        row["status"] = "interesting" if not why else "rejected"
        row["why"] = "; ".join(why)
        row["groups_needed_for_decisive_floor"] = (math.ceil(args.min_decisive / row["diversity_rate"])
                                                   if row["diversity_rate"] > 0 else None)
        out_rows.append(row)

    out_rows.sort(key=lambda r: (-r["decisive_pairs"], -(r["concordance"] or 0.0)))
    tot_pairs = sum(r["decisive_pairs"] + r["tie_similarity"] for r in out_rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "specs": args.specs, "rollout_log": args.rollout_log, "pairs": args.pairs,
        "pool_pairs": len(pool), "fit_boundary": args.fit_boundary, "groups": len(groups),
        "hypotheses": n_hyp, "candidates": len(cands),
        "rollouts_per_group": round(n_hyp / len(groups), 2) if groups else 0,
        "max_decisive_pairs_available": int(sum(len(g["hyp_norms"]) * (len(g["hyp_norms"]) - 1) // 2
                                                for g in groups)),
        "floors": {"min_fit_agree": args.min_fit_agree, "min_heldout_agree": args.min_heldout_agree,
                   "min_decisive": args.min_decisive, "min_concordance": args.min_concordance},
        "null": ("concordance = 0.5 means the verdict is independent of which rollout is lexically "
                 "closer to the professional reference; concordance_p_vs_0.5 is the exact two-sided "
                 "binomial p-value against that null. NOTE the proxy is LEXICAL (token F1), so a "
                 "lexical exclusion candidate can concord without encoding grammar at all - that is "
                 "why agreement-vs-professionals is reported in the same row and why a candidate still "
                 "has to clear the three-axis certification."),
        "candidates": out_rows,
        "interesting": [r["name"] for r in out_rows if r["status"] == "interesting"],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"\n{'candidate':<24}{'fitA':>6}{'hoA':>6}{'applG':>6}{'divR':>6}{'decP':>5}{'+':>4}{'-':>4}"
          f"{'conc':>7}{'p':>7}{'plcConc':>9}{'plcDec':>7}{'gap':>7}{'plcP':>7}  status")
    for r in out_rows:
        fa = f"{r['fit_agreement']:.2f}" if r["fit_agreement"] is not None else "  -"
        ha = f"{r['heldout_agreement']:.2f}" if r["heldout_agreement"] is not None else "  -"
        plc = (f"{r['placebo_pooled_concordance']:.2f}"
               if r.get("placebo_pooled_concordance") is not None else "   n/a")
        gap = (f"{r['placebo_gap']:+.2f}" if r.get("placebo_gap") is not None else "  n/a")
        pp = (f"{r['placebo_p']:.3f}" if r.get("placebo_p") is not None else "  n/a")
        print(f"{r['name']:<24}{fa:>6}{ha:>6}{r['applicable_groups']:>6}{r['diversity_rate']:>6.2f}"
              f"{r['decisive_pairs']:>5}{r['concordant']:>4}{r['discordant']:>4}{r['concordance']:>7.2f}"
              f"{r['concordance_p_vs_0.5']:>7.3f}{plc:>9}{r.get('placebo_decisive_pairs', 0):>7}{gap:>7}{pp:>7}"
              f"  {r['status']}"
              + (f"   need {r['groups_needed_for_decisive_floor']} groups" if
                 r["decisive_pairs"] < args.min_decisive and r["groups_needed_for_decisive_floor"] else ""))
    good = [r for r in out_rows if r["status"] == "interesting"]
    print(f"\ninteresting (both agreement axes AND concordance): {len(good)} "
          f"{[r['name'] for r in good][:10]}")
    best = max(out_rows, key=lambda r: r["diversity_rate"], default=None)
    if best:
        print(f"highest rollout-diversity rate: {best['name']} "
              f"({best['diversity_rate']:.2f} of applicable groups -> needs "
              f"{best['groups_needed_for_decisive_floor']} groups for {args.min_decisive} decisive pairs)")
    print(f"decisive pairs available in this log: at most "
          f"{int(sum(len(g['hyp_norms']) * (len(g['hyp_norms']) - 1) // 2 for g in groups))} "
          f"per candidate at {round(n_hyp/len(groups),2) if groups else 0} rollouts/group "
          f"(pairs scale as k(k-1)/2, so rollouts per group buy quadratic power)")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
