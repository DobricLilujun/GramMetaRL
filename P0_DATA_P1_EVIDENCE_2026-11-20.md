# P0 data / P1 verifier evidence — 2026-11-20

Reproducible commands and the measured numbers behind the current P0 gate state
(`PASS=19 FAIL=3 WARN=9`).

## 1. Sentence-level parallel data (license-first)

| corpus | origin | license | raw | usable pairs |
|---|---|---|---|---|
| Tech-in-GOV `Meisproochegen Iwwersetzungskorpus` LU-EN.jsonl | professional translators, Open Data portal LU | **CC0 1.0** | 10,880 | 8,972 |
| `fredxlpy/LuxAlign` v3 `lb-en` | RTL.lu news-derived sentence pairs | **CC BY-NC 4.0** | 43,357 | 39,464 |
| `mbarnig/Tatoeba-en-lb` | community sentences (HF) | CC BY 2.0 FR | 42.8k | probe only (junk targets) |

Ingest: `python -m grammetarl.data_sources.external_pairs`
→ `data/external/pairs_en_lb.jsonl` (48,436 rows, `source` + `license` per row) + `MANIFEST.json`.

Access notes (for the paper's data appendix):
* `data.public.lu` answers **the portal homepage** for `/api/1/...` and `/api/explore/v2.1/...` from
  this host; the real assets are direct zips on `download.data.public.lu`, and the corpus ships
  JSONL **and** TMX. No crawling was needed.
* `LuxAlign` is not gated; the HF datasets-server looked empty only because its config name is
  `lb-en` (not `default`). Its own card warns pairs are semantically similar rather than exact
  translations, so they are used as label candidates, never as gold MT references.

## 2. Verifier-based `gold_rules` labeling over the full pool

`python scripts/pilot_rule_tagging.py --from-file data/external/pairs_en_lb.jsonl --rows 50000`

| measure | value |
|---|---|
| pairs loaded | 48,436 |
| pairs with ≥1 candidate executable card | 47,348 (97.8%) |
| pairs with ≥1 constraint fired → labeled | **25,025 (51.7%)** |
| executable cards that ever fire on natural data | **38 / 61** |
| vacuous exclusion checks (matched nothing) | 34,725 |
| retrieved-rule check failures | **192,771 / 226,403 = 85.2%** |

Per-corpus 1,500-pair samples:

| corpus | ≥1 candidate | labelable | firing cards | fail rate |
|---|---|---|---|---|
| Tatoeba (noisy) | 23% | 6% | 16/61 | 92.0% |
| Tech-in-GOV (CC0) | 93% | 53% | 25/61 | 83.2% |
| LuxAlign (NC) | 98% | 52% | 26/61 | 84.8% |


## 2b. Correction after the three-valued verifier (same 3,000-pair sample)

The 51.7% "labelable" figure above counts a constraint as evidence even when the rule never
governed the sentence. Once checks require decided applicability, the honest numbers are:

| | no applicability (old cards) | compiled applicability (21/87 cards) |
|---|---|---|
| decided checks | **0 / 15,285 (0%)** | **890 / 15,268 (5.8%)** |
| satisfied / violated | - | 368 / 522 |
| provably not applicable (would have been false penalties) | 0 | **1,543 (10.1%)** |
| applicability undecidable (triggers still prose) | 100% | **84.1%** |
| pairs with >=1 decided label | 0 | **93 / 3,000 (3.1%)** |
| executable cards producing decidable checks | 0 | **6 / 61** |

Reading: the three-valued verifier removes a whole class of guaranteed-false penalties, but it
also exposes that the *usable* grammar signal is only 5.8% of checks. Applicability compilation
is the bottleneck, not retrieval and not the constraint engine:
  - 21 predicates accepted, 73 rejected, and **69 of the rejections are the model honestly
    answering `llm_check`** (e.g. "the referent is a female person" is not surface-observable);
  - therefore rule induction must optimize for *decidability*: every card should ship at least
    one surface-observable trigger. Decidability rate (5.8% today) becomes a reported property of
    the card set.

Two ways to raise it: (a) execute `llm_check` predicates offline at dataset-build time (never in
the RL hot loop), then keep the machine-checked output constraint as the label; (b) re-compile
triggers with surface-observable proxies plus a card-design guideline that requires one regex-
decidable trigger per card.

## 3. Dataset splits (P0 → trainable, P1 → falsifiable)

`python -m grammetarl.training.build_dataset --no-lexicon --external-tagged data/processed/tagged_all.jsonl`

| file | rows | purpose |
|---|---|---|
| `data/processed/train.jsonl` | **18,342** | pair-level train (mean src 19.1 tokens, `gold_rules` coverage 100%) |
| `data/processed/valid.jsonl` | 2,038 | pair-level dev |
| `data/processed/valid_rules_heldout.jsonl` | 4,645 | **rule-level held-out** (6 of 38 labeled rules never appear in train) |
| `data/processed/heldout_rules.json` | 6 rule ids | audit trail for the split |
| `data/processed/pairs_subsentences.jsonl` | 150 | grammar-book word/paradigm drills (P4 curriculum) |

License mix inside the training pool: CC0 4,258 / CC BY-NC 4.0 20,767.
A commercially-clean subset means CC0-only (4,258 pairs) — flag before any non-academic release.

## 4. What is still blocking RL (gate state)

| gate | status | meaning |
|---|---|---|
| G12 sentence data | **PASS** | was: 1.6-token glosses, 0/200 gold rules |
| G20 pair leakage | **PASS** | no exact pair in both splits |
| G23 rule-level split + licenses | **PASS** | held-out rules absent from train, every row licensed |
| G10 `grammar_apply` | **FAIL** | tool returns the source sentence as the draft (placeholder) |
| G13 eval replay | **FAIL** | hypothesis == source; consequence of G10 |
| G21 applicability | **FAIL** | 100 trigger conditions exist across 87 cards, but only **5 (5%)** are in executable form |

The 85% failure rate is now explained: rules are retrieved for 97.8% of all sentences and
checked without any applicability test, so most checks are of rules that do not govern the
sentence. Three concrete fixes, in order:
1. compile the existing 100 prose triggers into executable applicability predicates
   (only 5/100 are regex-shaped today, and the verifier never evaluates triggers at all);
2. three-valued verifier: `violated / satisfied / not-applicable` (no reward term when not applicable);
3. re-measure fire/fail per card on the 25,025 labeled pairs; drop or re-synthesize cards that
   stay dead (23 of 61 executable cards never fired once).


## 2c. Constraint precision (judged-applicable checks against human translations)

`scripts/audit_constraint_precision.py` replays the cached applicability judgements and runs only the
machine constraints (no LLM in the loop):

| card set | applicable checks | satisfied | violated | violation rate |
|---|---|---|---|---|
| `mbg_cards_app` (as extracted) | 138 | 50 | 88 | **63.8%** |
| `mbg_cards_repaired` (4 constraints downgraded on evidence) | 110 | 59 | 51 | **46.4%** |

Professional human translations should satisfy the grammar of their own language, so a ~46-64%
violation rate means the compiled programs are wrong. Root causes from the counter-examples:
  - identifiers pasted into regex slots (`same_suffix_for_definite_and_indefinite`);
  - **negative or paradigm-identity statements compiled as `must_include`** ("nouns do not take case
    markers", "nominative and accusative forms are identical") - not expressible as containment at
    all. These need equivalence/optionality constraint kinds, which is exactly the P3
    "rules as code" argument, now with numbers;
  - wrong allomorph sets (`datts|wéinis` vs the book's own example `datt s du`).

Repair is non-destructive: downgraded checks become `llm_check` and every change is annotated in the
constraint description, so a later re-compilation can undo it. New gate G25 asserts no prose or
identifier sits in a regex slot: it FAILs on `mbg_cards_app` and PASSes on `mbg_cards_repaired`.

## 2d. Constraint kinds: measured cost of "card -> executable program" (P3 evidence)

Same 1,200 CC0 Tech-in-GOV pairs, same cached applicability judgements (4,196 decisions), zero new
model calls per re-measurement. Verdicts come only from machine constraints.

| card set | applicable checks | satisfied | violated | violation rate | verified labels | distinct rules |
|---|---|---|---|---|---|---|
| `mbg_cards_app` (as extracted) | 493 | 137 | 356 | 72.2% | 133 | 11 |
| `mbg_cards_repaired` (evidence-driven downgrade) | 325 | 144 | 181 | 55.7% | 140 | 11 |
| `mbg_cards_recompiled` (kind re-choice + admission tests) | 371 | 167 | 204 | **55.0%** | **162** | 11 |

Kind re-compilation changed 48 constraints:
27 -> `must_include_any_regex` (real disjunctions), 17 -> `must_include_regex`, 19 -> `llm_check`,
8 -> `optionality`, 3 -> `equivalence`, 5 -> `must_exclude_regex`; 11 got no proposal, 7 were rejected.

Two findings that only a machine check can produce:

1. **~13% of the rules that extraction turned into output constraints cannot be expressed as a
   containment test at all** (8 optionality + 3 equivalence out of the recompiled set). Examples:
   "nominative and accusative forms are identical", "the particle may be omitted". They are kept as
   explicit *deferred* checks that abstain, and deciding them requires a morphology engine - the
   concrete, measurable motivation for P3 (rules as code).
2. **Admission tests caught compilation errors without any human linguist**: 2 exclusions fired on
   their own card's example sentence (a rule contradicting its own illustration), 4 includes never
   fired on any own example. Those were rejected rather than shipped.

Accounting corrections made along the way (each one moved a reported number):
- undecidable constraints used to certify: a card whose constraints were all deferred / vacuous /
  un-routed llm_check was reported "satisfied". Now a verdict requires >=1 decidable constraint.
- exclusions are one-sided evidence: an absent forbidden shape carries no information, so it no
  longer yields "satisfied"; its presence still penalizes. This lowered `satisfied` (179 -> 144)
  while making the labels trustworthy.
- legacy kind names (`must_not_include_regex`) silently degraded to `llm_check` in the extractor's
  mapping, so a supposedly machine-checkable exclusion was never checked.

Residual problem, quantified: the single most productive card (`LB_0c9c0e28da81`, definite article
full vs contracted forms) is applicable 154 times and still wrong 43.5% of the time on professional
translations. Yield is capped by constraint precision, not by data volume: 162 verified labels out of
1,200 pairs (13.5%), concentrated in 11 rules.

## 2e. Precision vs coverage, measured (the P1 reward-design problem in one table)

`scripts/triage_constraints.py` demotes, by evidence only, any machine constraint that violated on
>= 30% of its judge-applicable professional pairs with >= 10 applicable pairs. The measured numbers
are written into the constraint description, so each demotion is auditable and reversible.

| card set | applicable | satisfied | violated | violation rate | verified labels | distinct rules |
|---|---|---|---|---|---|---|
| as extracted | 493 | 137 | 356 | 72.2% | 133 | 11 |
| repaired | 325 | 144 | 181 | 55.7% | 140 | 11 |
| recompiled | 371 | 167 | 204 | 55.0% | 162 | 11 |
| **triaged** | 61 | 40 | 21 | **34.4%** | **39** | **7** |

Diagnosis from the dossiers (`artifacts/audit/violations_recompiled.json`): the compiled constraints
were derived from the card's EXAMPLE MORPHEME, not from the rule -
`^(deen|déin|d'|den|dem|der)$` for "the article has a full and a reduced form",
`^dënn-en$` for "there is no strong/weak adjective distinction",
`(mäin|däin|säin|hir|eis|är)` for "possessive articles agree with the possessor",
`(datt|wéini)\s*s\b` for "the complementizer takes {-s} with a 2sg subject".
The rule can genuinely govern a sentence while that token is absent, so the check fails on
professional translations at a measured rate.

The tension is explicit: the 8 demoted cards supplied 123 of the 162 verified labels. The surviving
58 machine constraints certify only 39 pairs (3.3% of 1,200) but at ~66% agreement with human
translations. Consequences for P1:
1. a rule-compiled grammar reward is either high-coverage/noisy or low-coverage/precise;
2. the demoted majority needs phrase-level applicability (does this sentence contain an attributive
   adjective / a possessive NP / a subordinate clause with a 2sg subject), i.e. POS + morphology -
   the concrete P3 unlock, not a rhetorical one;
3. the alternative is per-rule reliability weighting instead of binary keep/drop: the audit already
   measures per-rule precision, so the reward can weight each verdict by it (next implementation step).

## 3. Reliability-weighted grammar reward: measured, not asserted

A professional human translation must not be punished by the grammar term, so any pair whose grammar
reward is negative is a *false penalty* - directly measurable with the judge cache replayed (no LLM).
`scripts/reward_sanity_check.py` compares the naive unweighted mean over verdicts with the
reliability-weighted mean, where each rule's weight = clamp(2p-1, 0, 1) and p is its agreement with
human translations, shrunk toward the coin flip by support: p_hat = (satisfied + 2)/(applicable + 2)
(`grammetarl/rewards/reliability.py`, built by `scripts/build_reliability_table.py`).

| card set | pairs scored | false-penalty rate (unweighted) | false-penalty rate (weighted) | mean reward unweighted | mean reward weighted |
|---|---|---|---|---|---|
| recompiled | 359 | **52.9%** | **25.1%** | -0.134 | +0.187 |
| triaged | 66 | 33.3% | 6.1% | 0.333 | 0.561 |

Sign flips on the recompiled set: weighting removed 102 false penalties and introduced 2.

This is the third option the triage section asked for, and it dominates the binary keep/drop choice:
recompiled+weights keeps 5.4x the coverage of the triaged set (359 vs 66 scored pairs) at a lower
false-penalty rate than triaged-unweighted (25.1% vs 33.3%). A hard triage was throwing away signal
that soft weighting keeps.

Table contents (`data/rules/reliability.json`): 35 rules measured, 11 with weight > 0, best 0.88
(29/29 agreement), and the refuted cards land at 0.24 / 0.14 / 0.00 - they can no longer move the
reward in either direction.

Reward invariants now enforced by tests:
- uniform weights reproduce `grammar_reward_verdicts` exactly (a generalization, not a new metric);
- an unmeasured or coin-flip rule has weight 0 and cannot change the reward;
- 2/2 observations do not certify a rule (shrinkage): weight(SMALL) < 0.5 < weight(BIG at 60/60).
Gate G26 requires the table to exist, weights in [0,1], >=5 rules with weight > 0, and the
reward-sanity artifact to show the weighted false-penalty rate strictly below the unweighted one.
Wired into `run_eval` (`--reliability`, default `data/rules/reliability.json`); `grammar_verify`
now returns `rule_ids` aligned 1:1 with `verdicts` so weights can be attached downstream.

## 4. Applicability predicates: measured precision, and the gated verifier

Applicability is the gate that decides whether a check runs, so a bad predicate is worse than a bad
constraint: a false positive re-imports the false penalty, a false negative silently deletes signal.
The judge cache is ground truth for this, so predicates are scored with zero model calls
(`scripts/validate_triggers.py`).

| predicate source | judge-applicable checks | predicate fires | agreement | precision | recall |
|---|---|---|---|---|---|
| bulk LLM-compiled (Strata, precise subset) | 17 | 341 | 14 | **0.041** | 0.824 |
| hand-authored from judge-labeled examples | 123 | 199 | 69 | **0.347** | 0.561 |
| same, held-out half (rows 601-1200) | 69 | 112 | 40 | **0.357** | 0.580 |

Per card (held-out P/R): possessive determiner 0.77/0.57, present perfect 0.67/0.55, default -en
plural 0.50/0.40 - kept; subordinate clause 0.19/0.65 and 1sg present 0.30/0.75 - rejected for
precision. Guided authoring beats bulk compilation by ~8.5x on precision at similar recall, and the
held-out half shows it is not fit to the pool.

Two structural findings:
1. **The judge labels applicability of the card's stated trigger, not of the whole rule
   configuration.** For LB_353e50a17df4 ("complementizer takes {-s} with a 2sg subject") the judge
   marks "subordinate clause present" applicable (31/369) while none of the positives contain a 2sg
   subject. A predicate fitted to the *rule* rather than the *trigger* cannot agree with it.
2. **For always-surface-present rules, applicability is near-degenerate**: LB_0c9c0e28da81 is judged
   applicable on 162 pairs and not applicable on 157 - a coin flip on a corpus where almost every
   sentence contains a definite article. Fitting a predicate to that label set would be fitting
   noise; these rules need phrase-level structure (which NP, is an adjective intervening), i.e. P3.

### Gated composition (`scripts/gated_pipeline_measurement.py`)
A card may emit a verdict only if (a) its predicate validated at precision >= 0.5 against the judge
and (b) its machine constraints have reliability weight > 0. Over the 1,200 judged CC0 pairs:

| stage | labeled pairs | decided checks | false-penalty rate | rules |
|---|---|---|---|---|
| unweighted, recompiled constraints | 162 | 371 | 55.0% violation rate | 11 |
| reliability-weighted, recompiled | 359 scored | 371 | 25.1% | 11 |
| **predicate-gated + reliability-weighted** | **440** | 558 | **8.9%** | 2 |

So the pipeline now produces 2.7x the verified labels of the best previous configuration at ~6x
lower error. Honest limitation: those 440 labels come from just **2 rules** (possessive determiners,
default -en plural), so `valid_rules_heldout` is empty for this split (0 rows) - rule-generalization
cannot be measured until more rules pass the gate. Getting there is a predicate-quality problem, not
a data-volume problem.

## 5. One denominator: reliability measured under the gating the pipeline actually uses

Correction of a real error in section 4. The 8.9% false-penalty figure was computed with weights
built from a JUDGE-gated audit while the pipeline gates on the card's own PREDICATE. Two different
meanings of "this rule applies here". Re-measuring with `scripts/predicate_gated_audit.py` (every
decision from the compiled predicates, no judge) and rebuilding weights from it:

| gate (predicate precision >= 0.5) | allowed cards | scored pairs | labeled pairs | decided checks | FP rate |
|---|---|---|---|---|---|
| agreement > 0 (any weight) | 3 | 632 | 510 | 838 | **19.9%** (was reported as 8.9% under the mismatched denominator) |
| agreement >= 0.6 | 2 | - | 440 | 558 | 8.9% |
| agreement >= 0.7 | 1 | - | 359 | 359 | **0.0%** |

So the honest statement is a frontier, not a single number: the verifier can certify 359 CC0 pairs
with ZERO false penalties (one rule: default -en plural), or 440 pairs at 8.9% (adding possessive
determiners), or 510 pairs at 19.9% (adding a present-perfect rule whose predicate validated but
whose constraint agrees only 51% of the time - a coin flip).

Predicate-gated, ungated-by-quality over the same 1,200 pairs: 6,046 decided checks but 63.2%
violation rate. Again: unvalidated predicates manufacture yield.

The high-agreement cards waiting on predicate validation (from `reliability_predicate_gated.json`):
LB_9789dfb15623 comparative particle 0.94 agreement on 146 checks, LB_e51077fe7961 pronoun forms
0.80 on 249 checks - both had ZERO judge coverage, which is why they could not pass the predicate
gate. The `--judge-all` run in progress targets exactly them.

## 6. Scale check: subset estimates were biased; the full professional pool is the honest base

Section 5's numbers came from the first 1,200 pairs of the pool. Measuring on ALL 8,972 CC0
professional pairs (`--source-filter techgov_lu_en --rows 0`) moves them materially:

| gate | allowed cards | labeled pairs | decided checks | FP rate |
|---|---|---|---|---|
| min-agreement 0.6 | 2 | 2,810 | 3,639 | 13.2% |
| min-agreement 0.65 | 1 | **2,041** | 2,041 | **0.0%** |
| (subset 1,200, same gate 0.6) | 2 | 440 | 558 | 8.9% <- optimistic |

Per-rule agreement, full pool vs the 1,200 subset, is the reason:
`LB_946a24dd25d6` (possessive determiners) 0.66 on the subset -> **0.51 on 1,826 observations**.
A coin flip once measured properly. Support matters: every rule now carries its n in the table.

The actual bottleneck is now isolated. These rules already have constraints that agree with
professional translations at full scale, and reliability weight > 0:

| card | n (predicate-gated) | agreement | weight | predicate validated? |
|---|---|---|---|---|
| LB_d121073f582e default -en plural | 2,041 | 1.00 | 1.00 | yes (0.50 held-out) |
| LB_27c64c1f7e4b 2nd person plural address forms | 1,993 | 1.00 | 1.00 | **never judged** |
| LB_374581869ba8 1sg present = infinitive | 676 | 1.00 | 0.99 | yes but P=0.30 (rejected) |
| LB_9789dfb15623 comparative particle | 680 | 0.92 | 0.84 | **never judged** |
| LB_e51077fe7961 1st/2nd person pronoun case | 1,940 | 0.80 | 0.59 | **never judged** |

So the P1 dataset size is bounded by predicate validation, not by constraint quality or data volume:
if those three unjudged rules validate at precision >= 0.5, the allowed set goes 1 -> 4 rules and the
certified CC0 pairs go from 2,041 to several thousand (their predicate-gated check counts total ~4.6k)
at essentially the same 0% false-penalty rate. That targeted judge pass is running on gpu7.

## 7. The verifier quality matrix: agreement x discrimination (this changes the P1 pick)

Agreement against correct translations is only half of verifier quality. A constraint that cannot
fail scores agreement 1.00 and 0% false penalties while contributing a CONSTANT reward term. Measured
directly (`scripts/verifier_discrimination.py`, `scripts/rule_quality_matrix.py`): take a check that
passed on a professional translation, apply a minimal perturbation designed to break THAT rule
(token-level: singularize, swap the plural marker, mangle the inflection), and require a `violated`
verdict.

| rule | agreement (n) | discrimination | verdict |
|---|---|---|---|
| LB_d121073f582e default -en plural | 1.00 (2,041) | **5.3%** (247 tests) | **vacuous** |
| LB_374581869ba8 1sg present = infinitive | 1.00 (676) | 8.4% (249) | vacuous |
| LB_e51077fe7961 1st/2nd person pronoun case | 0.80 (1,940) | 17.6% (125) | vacuous |
| LB_27c64c1f7e4b 2nd person plural address forms | 1.00 (1,993) | **70.2%** (104) | **usable** |
| LB_9789dfb15623 comparative particle | 0.92 (680) | **60.7%** (300) | **usable** |
| LB_f73c08e1e60c present perfect | 0.64 (1,598) | 66.0% (294) | weak (agreement) |
| LB_2ebda66afe75 French kinship borrowings | 0.59 (33) | 100% (60) | weak (agreement) |

Consequences, stated plainly:
1. The rule promoted in section 6 (2,041 CC0 labels, 0.0% false penalties) is **not usable dense
   supervision**: its constraint is `^.*en$` at token level - "some token ends in -en" - which holds
   for almost any sentence, so it catches 5% of targeted violations. Reward terms like this are
   constants dressed as signals.
2. Why it happened: the constraint was compiled from the rule's *surface vocabulary*, not from its
   *structure*. "The default plural marker is -en" requires knowing WHICH noun is plural, i.e.
   aligning the source plural to a target noun. Containment regexes cannot express that; this is the
   P3 argument with a number attached.
3. The two rules that are genuinely usable (comparative particle, 2nd-person-plural address) are
   exactly the ones blocked only by missing predicate validation - which is what the targeted
   `--judge-all` pass is producing. The right gate is therefore two-sided:
   agreement >= 0.65 AND discrimination >= 0.5 AND predicate precision >= 0.5.
