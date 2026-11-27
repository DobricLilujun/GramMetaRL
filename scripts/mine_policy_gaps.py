#!/usr/bin/env python3
"""Mine candidate constraints that the POLICY actually violates but professionals do not.

Why this exists (evidence 14.5): certified-rule COUNT is not the binding constraint on GRPO signal.
Going 2 -> 8 rules moved advantage coverage by at most +4.2pp on identical hypotheses, because rules
selected on corpus-average properties are mostly SATURATED (the 12B already satisfies the cue, so the
within-group advantage is zero) or ANTI-PROFESSIONAL (`LB_MIN04_negation_not`: -10.0pp, professionals
are penalised harder than the policy). A reward term needs rules where the policy and the reference
disagree, so candidates are mined from the difference between stored policy rollouts and professional
references.

Two stages, in this order, because each stage's estimator needs a different sample:

  A. POOL stage (robust, ~4,000 pairs in the fit half). Enumerate (English cue, Luxembourgish form)
     pairs where the form is near-OBLIGATORY under the cue among professional translations
     (agreement >= --min-pool-agreement) AND cue-SPECIFIC (lift = p(form|cue) / p(form|cue absent)
     >= --min-lift). The lift axis is what kills co-occurrence junk: `has -> den` looked like a 0.50
     gap because `den` is a dative article professionals write about half the time regardless of the
     cue. A check that fires no matter what is not a check - the same lesson as the 0.39%
     false-penalty Goodhart correction.
  B. POLICY stage (small sample, used only to RANK and gate). Over the stored rollout corpus, keep
     candidates whose policy realisation rate is below the professional rate by --min-gap. These are
     the rules that can produce non-zero advantage.

Stage A statistics are restricted to the fit half of the pool (index < --fit-boundary) so that the
existing two-half certification - scripts/screen_source_conditionals.py --skip <fit-boundary>, then
scripts/rule_quality_matrix.py --skip <fit-boundary> - still measures something: a candidate that
survives has never been seen by the miner on the held-out half.

What this does NOT certify: agreement, lift and support floors are screening axes; discrimination
saturates at 1.00 for cue-type rules (evidence 14.6), and `applicability_sensitivity` (delete the cue,
must stop firing) is trivially 1.00 for a single-token cue, so the axis that is genuinely missing is
judge-scored applicability precision, which needs Strata on gpu7 (scripts/label_with_judge.py
--card-ids, then scripts/validate_triggers.py). Output format is exactly
data/rules/screened_conditionals.json so the existing screener consumes it unchanged.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grammetarl.validator import TranslationVerifier

WORD = re.compile(r"\w+", re.UNICODE)


def toks(text: str) -> set[str]:
    return {t for t in WORD.findall(TranslationVerifier.normalize_text(text)) if len(t) >= 2}


def load_pool(path: Path, source_filter: str, fit_boundary: int) -> tuple[list[tuple[str, str, str]], int]:
    """(src_norm, ref_norm, src_raw) for the mining half, plus how many pairs were skipped."""
    rows: list[tuple[str, str, str]] = []
    skipped = 0
    seen = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if source_filter and rec.get("source") != source_filter:
                continue
            if not rec.get("src") or not rec.get("tgt_ref"):
                continue
            if seen >= fit_boundary:
                skipped += 1
                seen += 1
                continue
            seen += 1
            rows.append((TranslationVerifier.normalize_text(rec["src"]),
                         TranslationVerifier.normalize_text(rec["tgt_ref"]), rec["src"]))
    return rows, skipped


def load_corpus(paths: list[Path], pair_index: dict[str, int], fit_boundary: int) -> list[dict]:
    """Merge rollout logs into (src, ref, [hyp,...]); pairs outside the mining half are dropped."""
    groups: dict[str, dict] = {}
    off_half = 0
    for p in paths:
        if not p.exists():
            print(f"[warn] no rollout log at {p}")
            continue
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                src, hyp, ref = rec.get("src"), (rec.get("hyp") or "").strip(), (rec.get("tgt_ref") or "").strip()
                if not src or not hyp or not ref:
                    continue
                idx = pair_index.get(src)
                if idx is None:
                    continue
                if idx >= fit_boundary:
                    off_half += 1
                    continue
                g = groups.setdefault(src, {"src": src, "ref": ref, "hyps": []})
                if hyp not in g["hyps"]:
                    g["hyps"].append(hyp)
    corpus = [{"src_norm": TranslationVerifier.normalize_text(g["src"]),
               "ref_norm": TranslationVerifier.normalize_text(g["ref"]),
               "hyp_norms": [TranslationVerifier.normalize_text(h) for h in g["hyps"]],
               "src_toks": toks(g["src"]), "ref_toks": toks(g["ref"])} for g in groups.values()]
    n_hyp = sum(len(c["hyp_norms"]) for c in corpus)
    print(f"policy corpus: {len(corpus)} sources, {n_hyp} distinct hypotheses "
          f"({off_half} rollout rows outside the mining half)")
    return corpus


def main() -> None:
    ap = argparse.ArgumentParser(description="Mine policy-gap constraint candidates (pool stage + policy stage)")
    ap.add_argument("--rollout-log", action="append", default=[],
                    help="rollout log(s) from scripts/p1_reward_smoke.py (repeatable)")
    ap.add_argument("--pairs", default="data/external/pairs_en_lb.jsonl")
    ap.add_argument("--source-filter", default="techgov_lu_en")
    ap.add_argument("--fit-boundary", type=int, default=4000,
                    help="mining half = first N filtered pool pairs; certification held-out starts here")
    ap.add_argument("--min-pool-cue", type=int, default=30, help="pool pairs where the cue fires")
    ap.add_argument("--min-pool-agreement", type=float, default=0.5,
                    help="professionals must realise the form this often under the cue (obligatoriness)")
    ap.add_argument("--min-lift", type=float, default=2.0,
                    help="p(form|cue) / p(form|cue absent) on the pool; kills co-occurrence junk")
    ap.add_argument("--min-cue-pairs", type=int, default=5, help="corpus pairs where the cue fires")
    ap.add_argument("--min-gap", type=float, default=0.10,
                    help="professional minus policy realisation on the corpus (the teachable signal)")
    ap.add_argument("--negative", action="store_true",
                    help="also mine EXCLUSION candidates: a target form professionals almost never use "
                         "under the cue while the policy does. This is the mirror of the token stage and "
                         "it targets the policy's own errors (penalising a shape professionals avoid is "
                         "the only cue-level rule family that cannot be saturated)")
    ap.add_argument("--max-pro-rate", type=float, default=0.05,
                    help="professionals may realise a forbidden form at most this often under the cue "
                         "(a higher rate makes the rule anti-professional, like LB_MIN04_negation_not)")
    ap.add_argument("--min-policy-rate", type=float, default=0.15,
                    help="the policy must actually produce the forbidden form this often, else the check "
                         "can never fire and the rule is vacuous in the other direction")
    ap.add_argument("--max-per-cue", type=int, default=2)
    ap.add_argument("--ordered", action="store_true",
                    help="also mine ORDERED target pairs (pattern t1 .* t2): order-type constraints are "
                         "where the policy actually errs, single-token cues are mostly saturated")
    ap.add_argument("--min-ordered-cue", type=int, default=60, help="pool support for order candidates")
    ap.add_argument("--top-k-tokens", type=int, default=4, help="target tokens per cue for order pairs")
    ap.add_argument("--min-ordered-agreement", type=float, default=0.45)
    ap.add_argument("--min-ordered-lift", type=float, default=2.5)
    ap.add_argument("--max-candidates", type=int, default=20)
    ap.add_argument("--cue-blacklist", default="",
                    help="comma-separated cue words to skip (rarely worth a card, e.g. very short particles)")
    ap.add_argument("--spec-out", default="data/rules/policy_gap_candidates.json")
    ap.add_argument("--report", default="artifacts/p1/policy_gap_mining.json")
    args = ap.parse_args()
    if not args.rollout_log:
        raise SystemExit("pass at least one --rollout-log")

    blacklist = {b.strip().casefold() for b in args.cue_blacklist.split(",") if b.strip()}
    pool, skipped = load_pool(Path(args.pairs), args.source_filter, args.fit_boundary)
    pair_index = {raw: i for i, (_, _, raw) in enumerate(pool)}
    print(f"mining half: {len(pool)} pool pairs (skipped {skipped} beyond index {args.fit_boundary})")

    # ---------------- stage A: pool statistics on the mining half -------------------------------
    ref_sets = [toks(r) for _, r, _ in pool]
    form_pairs: collections.Counter = collections.Counter()
    cue_pairs: dict[str, list[int]] = collections.defaultdict(list)
    for i, (src_n, _, _) in enumerate(pool):
        for e in toks(src_n):
            cue_pairs[e].append(i)
        for t in ref_sets[i]:
            form_pairs[t] += 1

    stage_a: list[dict] = []
    pool_ref_norm = [r for _, r, _ in pool]
    pool_src_norm = [s for s, _, _ in pool]
    cue_set: dict[str, set[int]] = {}
    for cue, ids in cue_pairs.items():
        cue_set[cue] = set(ids)
    for cue, ids in cue_pairs.items():
        n_cue = len(ids)
        if n_cue < args.min_pool_cue or cue in blacklist:
            continue
        cnt: collections.Counter = collections.Counter()
        for i in ids:
            for t in ref_sets[i]:
                cnt[t] += 1
        for form, hit in cnt.items():
            agree = hit / n_cue
            if agree < args.min_pool_agreement:
                continue
            base = (form_pairs[form] - hit) / (len(pool) - n_cue)
            if base <= 0:
                continue
            lift = agree / base
            if lift < args.min_lift:
                continue
            stage_a.append({"cue": cue, "form": form, "pattern": rf"\b{re.escape(form)}\b",
                            "kind": "token", "pool_n_cue": n_cue,
                            "pool_agreement": round(agree, 4), "pool_base_rate": round(base, 4),
                            "lift": round(lift, 3)})
    print(f"stage A (pool, single token): {len(stage_a)} (cue, form) pairs pass obligatoriness + specificity")

    if args.ordered:
        n_order = 0
        for cue, ids in cue_pairs.items():
            n_cue = len(ids)
            if n_cue < args.min_ordered_cue or cue in blacklist:
                continue
            cnt2: collections.Counter = collections.Counter()
            for i in ids:
                for t in ref_sets[i]:
                    cnt2[t] += 1
            # rank cue tokens by SPECIFICITY (lift), not frequency: the frequent ones are frequent
            # everywhere, so an order pair built from them cannot clear the lift floor and the search
            # degenerates into "function word appears before another function word"
            scored = []
            for t, hit in cnt2.items():
                agree_t = hit / n_cue
                if agree_t < 0.15:
                    continue
                base_t = (form_pairs[t] - hit) / (len(pool) - n_cue)
                if base_t <= 0:
                    continue
                scored.append((agree_t / base_t, t))
            scored.sort(reverse=True)
            top = [t for _, t in scored[:args.top_k_tokens]]
            id_set = cue_set[cue]
            for t1 in top:
                for t2 in top:
                    if t1 == t2:
                        continue
                    pat = rf"\b{re.escape(t1)}\b.*\b{re.escape(t2)}\b"
                    hit_in = sum(1 for i in ids if re.search(pat, pool_ref_norm[i]))
                    agree = hit_in / n_cue
                    if agree < args.min_ordered_agreement:
                        continue
                    hit_out = sum(1 for i in range(len(pool))
                                  if i not in id_set and re.search(pat, pool_ref_norm[i]))
                    base = hit_out / (len(pool) - n_cue)
                    if base <= 0:
                        continue
                    lift = agree / base
                    if lift < args.min_ordered_lift:
                        continue
                    stage_a.append({"cue": cue, "form": f"{t1}..{t2}", "pattern": pat,
                                    "kind": "order", "pool_n_cue": n_cue,
                                    "pool_agreement": round(agree, 4), "pool_base_rate": round(base, 4),
                                    "lift": round(lift, 3)})
                    n_order += 1
        print(f"stage A (pool, ordered pairs): +{n_order} candidates "
              f"(support>={args.min_ordered_cue}, agree>={args.min_ordered_agreement}, "
              f"lift>={args.min_ordered_lift})")

    # the rollout corpus is needed by both the exclusion stage and the ranking stage
    corpus = load_corpus([Path(p) for p in args.rollout_log], pair_index, args.fit_boundary)

    # ---------------- stage A3: exclusion candidates (professionals avoid it, policy does not) -----
    # For these, agreement is NOT the informative axis: p(professionals satisfy 'do not contain X') is
    # ~1 by construction. The axes that matter are the two measured rates (professional violation <= 
    # --max-pro-rate so the rule cannot be anti-professional; policy violation >= --min-policy-rate so the
    # check actually bites) plus the held-out protocol: a candidate must keep both rates on the other half.
    if args.negative:
        corpus_hyp_toks = [[toks(h) for h in cc["hyp_norms"]] for cc in corpus]
        # policy token rate under each cue, from the rollout corpus
        neg_n = 0
        for cue, ids in cue_pairs.items():
            n_cue = len(ids)
            if n_cue < args.min_pool_cue or cue in blacklist:
                continue
            c_ids = [i for i, cc in enumerate(corpus) if cue in cc["src_toks"]]
            if len(c_ids) < args.min_cue_pairs:
                continue
            pol_cnt: collections.Counter = collections.Counter()
            for i in c_ids:
                for ht in corpus_hyp_toks[i]:
                    for t in ht:
                        pol_cnt[t] += 1
            pro_cnt: collections.Counter = collections.Counter()
            for i in ids:
                for t in ref_sets[i]:
                    pro_cnt[t] += 1
            for form, pc in pol_cnt.items():
                pol_rate = pc / sum(len(corpus_hyp_toks[i]) for i in c_ids)
                if pol_rate < args.min_policy_rate:
                    continue
                pro_rate = pro_cnt.get(form, 0) / n_cue
                if pro_rate > args.max_pro_rate:
                    continue
                stage_a.append({"cue": cue, "form": form, "pattern": rf"\b{re.escape(form)}\b",
                                "kind": "exclude", "pool_n_cue": n_cue,
                                "pool_agreement": round(1.0 - pro_rate, 4),
                                "pool_base_rate": round(pro_rate, 4),
                                "lift": round(min(pol_rate / max(pro_rate, 1e-9), 999.0), 3),
                                "pro_violation_rate": round(pro_rate, 4),
                                "policy_violation_rate": round(pol_rate, 4)})
                neg_n += 1
        print(f"stage A3 (exclusions): +{neg_n} candidates "
              f"(pro violation<={args.max_pro_rate}, policy violation>={args.min_policy_rate})")

    # ---------------- stage B: rank by the policy's own failure --------------------------------
    ranked: list[dict] = []
    for c in stage_a:
        pat = re.compile(c["pattern"])
        cue_pat = re.compile(rf"\b{re.escape(c['cue'])}\b")
        ids = [i for i, cc in enumerate(corpus) if cue_pat.search(cc["src_norm"])]
        if len(ids) < args.min_cue_pairs:
            continue
        pol = 0.0
        for i in ids:
            per_hyp = [1.0 if pat.search(h) else 0.0 for h in corpus[i]["hyp_norms"]]
            pol += sum(per_hyp) / len(per_hyp)
        pol_rate = pol / len(ids)
        ref_corpus = sum(1 for i in ids if pat.search(corpus[i]["ref_norm"])) / len(ids)
        gap = ref_corpus - pol_rate
        if c["kind"] == "exclude":
            # for exclusions the check fires when the form is PRESENT, so the teachable quantity is
            # (policy violation rate - professional violation rate) = -gap; the professional rate is
            # already floored in stage A3 so the rule cannot be anti-professional noise
            if -gap < args.min_gap:
                continue
        elif gap < args.min_gap:
            continue
        c = dict(c)
        c.update({"corpus_n": len(ids), "corpus_ref_rate": round(ref_corpus, 4),
                  "corpus_policy_rate": round(pol_rate, 4),
                  "gap": round(-gap if c["kind"] == "exclude" else gap, 4),
                  "score": round(abs(gap) * (len(ids) ** 0.5), 4)})
        ranked.append(c)
    ranked.sort(key=lambda c: (-c["score"], -c["lift"]))
    if ranked and all(r["corpus_policy_rate"] == 0.0 for r in ranked):
        raise SystemExit("every policy realisation rate is 0.0: the hypothesis token sets are broken, "
                         "not the policy")

    per_cue: collections.Counter = collections.Counter()
    chosen: list[dict] = []
    for c in ranked:
        if per_cue[c["cue"]] >= args.max_per_cue:
            continue
        per_cue[c["cue"]] += 1
        chosen.append(c)
        if len(chosen) >= args.max_candidates:
            break

    def slug(s: str) -> str:
        return re.sub(r"\W+", "_", s).strip("_").lower()

    specs = {
        "_comment": ("candidates mined by scripts/mine_policy_gaps.py: pool obligatoriness + cue "
                     "specificity on the fit half, ranked by the policy-vs-reference gap measured on "
                     "stored rollouts. provenance = POLICY_GAP_MINED, not book-extracted; two-half "
                     "certification (screen_source_conditionals.py --skip) is still required"),
        "candidates": [{
            "name": f"pg_{slug(c['cue'])}_{slug(c['form'])}",
            "source_pattern": f"\\b{re.escape(c['cue'])}\\b",
            "values": [c["pattern"]],
            "negate": c["kind"] == "exclude",
            "why": (f"policy-gap mined: pool agreement {c['pool_agreement']} at n={c['pool_n_cue']} "
                    f"(base {c['pool_base_rate']}, lift {c['lift']}); corpus reference {c['corpus_ref_rate']} "
                    f"vs policy {c['corpus_policy_rate']} over {c['corpus_n']} pairs (gap {c['gap']})"),
        } for c in chosen],
    }
    spec_path = Path(args.spec_out)
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(json.dumps(specs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({
        "mining_half": {"pairs": len(pool), "fit_boundary": args.fit_boundary, "pool": args.pairs,
                        "source_filter": args.source_filter},
        "policy_corpus_pairs": len(corpus),
        "policy_corpus_hypotheses": sum(len(c["hyp_norms"]) for c in corpus),
        "floors": {"min_pool_cue": args.min_pool_cue, "min_pool_agreement": args.min_pool_agreement,
                   "min_lift": args.min_lift, "min_cue_pairs": args.min_cue_pairs,
                   "min_gap": args.min_gap, "max_per_cue": args.max_per_cue},
        "stage_a_passing": len(stage_a),
        "stage_b_ranked": len(ranked),
        "shipped": len(chosen),
        "candidates": chosen,
        "stage_b_rejected_sample": ranked[len(chosen):len(chosen) + 20],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"stage B (policy gap >= {args.min_gap}): {len(ranked)} ranked, {len(chosen)} shipped -> {spec_path}")
    print(f"{'cue':<12}{'form':<18}{'kind':<7}{'poolN':>7}{'agree':>7}{'base':>7}{'lift':>7}"
          f"{'corpus':>8}{'ref':>6}{'pol':>6}{'gap':>7}")
    for c in chosen:
        print(f"{c['cue']:<12}{c['form']:<18}{c['kind']:<7}{c['pool_n_cue']:>7}{c['pool_agreement']:>7.2f}"
              f"{c['pool_base_rate']:>7.2f}{c['lift']:>7.2f}{c['corpus_n']:>8}{c['corpus_ref_rate']:>6.2f}"
              f"{c['corpus_policy_rate']:>6.2f}{c['gap']:>7.2f}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
