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

## 8. Third criterion, and the honest bottom line: 0 rules pass all three

The discrimination test has a blind spot: perturbations are built FROM the constraint, so a rule can
bite on its own span and still verify nothing about the translation. Inspection shows exactly that -
the two "usable" rules restate their own trigger:

| rule | applicability predicate | output constraint |
|---|---|---|
| LB_27c64c1f7e4b polite address | target token `^(?:dir|der|iech)$` | `^dir$` OR `^der$` OR `^iech$` |
| LB_9789dfb15623 comparative | target `\bméi\b` | `méi\s+\w+` |

Both fire exactly when the constraint is satisfied, so the "check" is the trigger wearing a hat.
`scripts/rule_quality_matrix.py` now flags this mechanically (predicate and constraint reduce to the
same literal surface material).

Result under the three-sided gate (agreement >= 0.65, discrimination >= 0.5, not self-restating):

**usable rules: 0 of 8 candidates.** Breakdown: 3 vacuous (discrimination 5-18%), 2 tautological,
3 weak (agreement 0.59-0.64, or n=1). G27 now fails with that breakdown instead of passing on a
weaker criterion.

What this demands, stated as a specification rather than a complaint: a machine-checkable grammar
rule over a parallel pair must be **conditional on the source**, i.e.
`if the SOURCE exhibits X then the TARGET must realize Y` - which is what a translation rule is.
Containment checks on the target alone are either vacuous (always true) or tautological (equal to
their own trigger). That is the single concrete unlock for P1: a `source_conditional` constraint kind
(source pattern => target pattern), plus an honest verdict of "undecidable from this language pair"
for rules the source cannot condition at all - e.g. polite address forms, because English marks no
clusivity/formality, so LB_27c64c1f7e4b should be dropped from the machine-checked set, not tuned.

## 9. Closing the loop: targeted judge pass gives the real applicability base rate

`label_with_judge.py --judge-all` judged 26 predicate-bearing cards x 292 pairs (4,672 new
decisions; cache now 8,868) on gpu7/Strata. This removes the coverage hole from section 4.

Applicability base rate under unbiased judging: **402 of 7,592 checks (5.3%)** - far below what the
retrieval-mode judge implied, because retrieval mode both pre-selected related cards and skipped
cards that already had predicates.

Predicate quality with full coverage (micro): precision **0.254**, recall 0.637; held-out half
0.262 / 0.615 - so the number generalizes, but the gate is far from usable.

| card | judge-applicable /292 | predicate fires | P (held-out) | R (held-out) |
|---|---|---|---|---|
| LB_34e781490e7b reduced article form | 93 | 204 | 0.31 (0.33) | 0.69 (0.70) |
| LB_9d86392ccf45 feminine article case | 63 | 144 | 0.31 (0.36) | 0.72 (0.72) |
| LB_473d56626cdb masculine article case | 67 | 94 | 0.34 (0.43) | 0.48 (0.49) |
| LB_9789dfb15623 comparative particle | 24 | 34 | 0.50 (0.42) | 0.71 (0.80) |
| LB_27c64c1f7e4b polite address | **0** | 81 | 0.00 | 0.00 |
| LB_e51077fe7961 pronoun case | 4 | 20 | 0.20 (0.38) | 1.00 |
| LB_2ebda66afe75 / LB_bf1449011785 / LB_55471d5dc4b4 loanwords | 0 | 0 | - | - |

Two conclusions:
1. **LB_27c64c1f7e4b is undecidable for en->lb**, not merely hard: the judge marked the rule
   applicable in 0 of 292 pairs (English marks no formality distinction), while its predicate fires
   on 81. It must be removed from the machine-checked set - a card being "correct about
   Luxembourgish" does not make it checkable from this source language.
2. Even the best hand-authored predicate (comparative) falls to held-out precision 0.42, and the
   strongest bulk-compiled ones sit at 0.31-0.43. Surface regexes on a single side cannot express
   "the clause contains an attributive adjective between article and noun".

Bottom line for P1, all three axes measured on real data: 0 of 8 candidate rules satisfy
agreement >= 0.65 AND discrimination >= 0.5 AND non-tautological, and the applicability predicates
that would gate even those sit at precision 0.25-0.43. The card-to-verifier compiler is the
bottleneck, and the required upgrade is structural: source-conditional constraints
(source pattern => target pattern) plus explicit `undecidable_for_pair` handling.

## 10. Source-conditional constraints: the first rule that passes all three axes

Implemented `source_conditional` (schema + validator): IF the source matches `source_pattern` the
target must realize one of `values`; if the source does not match, the constraint abstains (the rule
does not govern the pair). Applicability now comes from the source, so it no longer depends on a
target-side predicate - the axis where every compiled predicate scored 0.25-0.43 precision.

Hand-authored conditionals pass the same admission tests, and two of my three first attempts were
REJECTED for a real reason: the source pattern did not fire on the card's own illustration
(the comparative card's sources are bare comparatives `nicer`/`older`, not `X-er than`; the perfect
card's example is `they have built`, whose participle my `-ed`-based pattern missed). Fixing the
patterns to fit the card's own examples is what made them admissible - evidence that the admission
test is doing work, not just gating.

Agreement of the SAME rules, target-only vs source-conditional (full 8,972 professional pairs):

| rule | target-only | source-conditional |
|---|---|---|
| LB_946a24dd25d6 possessive articles | 0.51 (n=1,826) | **1.00 (n=1,433)** |
| LB_9789dfb15623 comparative | 0.92 (n=680) | **0.96 (n=507)** |
| LB_f73c08e1e60c present perfect | 0.64 (n=1,598) | 0.63 (n=162, condition rarely fires) |

Full three-axis matrix on the conditional set:

| rule | agreement (n) | discrimination | verdict |
|---|---|---|---|
| LB_9789dfb15623 comparative | 0.96 (507) | **59.3%** | **USABLE** |
| LB_f73c08e1e60c present perfect | 0.63 (162) | 63.7% | weak on agreement |
| LB_946a24dd25d6 possessive | 1.00 (1,433) | **1.7%** | vacuous: 11 alternatives cover nearly every possessive form, so breaking one token changes nothing |
| LB_d121073f582e / LB_374581869ba8 / LB_e51077fe7961 | 0.80-1.00 | 5-18% | vacuous |

So the possessive rule shows the failure mode from the other direction: perfect agreement, no biting.
Raising the alternatives' specificity is the next fix.

End-to-end with the source-conditional rules allowed by the gate (no predicate needed), over ALL
8,972 CC0 professional pairs: **3,580 pairs scored, 3,566 labeled pairs (39.7%), 3,981 decided
checks, false-penalty rate 0.39%** across 3 rules. Gate G27 now passes with a real rule behind it
(PASS=26 FAIL=2).

> **Superseded by section 11**: the 0.39% false-penalty rate came from a card set in which 2,041 of
> the 3,981 checks belonged to a rule with 5.3% discrimination - a rule that cannot bite, so it could
> not penalise anything. With discrimination measured, the honest number is 14.9%.

## 11. Held-out certification, two measurement artifacts that flattered us, and one that we flattered ourselves

**Source-derived applicability.** The validator now derives applicability from a
`source_conditional` constraint's own condition, and the audit/matrix accept such cards without a
target-side predicate. This is what made two rules measurable at all (they had n=0 before).

**The condition is measured with the same ruler as predicates** (`scripts/validate_triggers.py` now
scores source conditions against the judge cache):

| applicability signal | micro precision |
|---|---|
| bulk LLM-compiled predicates | 0.041 |
| hand-authored predicates | 0.347 |
| comparative source condition (tightened) | 0.50 (recall 0.71) |
| present-perfect source condition | 0.57 (recall 0.36) |

Two failed attempts recorded as evidence: `\b\w+er\b` matched *after/under/number*, and
`more [a-z]+` matched *more than / more or less* — idioms, not comparatives. The second failure
collapsed agreement 0.96 → 0.71, i.e. **a loose condition is as damaging as a loose predicate**.

**Two bugs in my own discrimination probe, found by the probe itself.**

(a) *Prefix-preserving corruption was a no-op.* `_break_word` used to keep the word stem and append
garbage, but the constraint patterns end in `\w*` on purpose (they must match an inflected paradigm),
so `hiren` -> `hirxx5` still satisfied `\bhir\w*\b`: **126/126** single-token perturbations failed to
flip for that reason alone. Corruption now replaces the whole matched span.

(b) *Per-occurrence rules need every instance broken.* A rule that must hold for each possessor
cannot be discriminated by breaking one of several (405 of 1,968 possessive pairs carry 2+ possessors).
`corrupt_all` breaks the span-union of all matches (overlapping alternatives count once).

With both fixed, full-pool discrimination: possessive **92.6%**, polite address 70.2%, comparative
75.3%, `een`-initial prohibition 100%, pronoun case 17.6%, 1sg present 8.4%, `-en` plural 5.3%.
The possessive and pronoun verdicts flipped: the possessive is not vacuous at all - it bites on 93% of
its targeted perturbations; it simply disagrees with professional translations often (0.61).

**Form lists over-matched.** `\bdäi\w*\b` matched `däischtere` ("those darker") and `\bsen\w*\b`
matched `sent` ("sand"), so the possessive check fired on non-possessives. Replaced with explicit
paradigms (`\bmäi(n|en|em|er)?\b` etc.); agreement moved 0.54 -> 0.61 and support stayed at 1,968.

**Held-out certification** (weights fit on pairs 0-3999; agreement measured on pairs 4000+, and
discrimination on those same pairs - the matrix now measures agreement in-run instead of re-printing
the reliability table, which previously made a `--skip` run look held-out while quoting fit numbers):

| rule | agreement (checks) | discrimination | verdict |
|---|---|---|---|
| LB_9789dfb15623 comparative | 0.66 (118) | 71.4% | **usable** |
| LB_363a21684c8f `een` may not open the sentence | 1.00 (13) | 100% | provisional (support 13 < 30) |
| LB_e51077fe7961 pronoun case | 0.79 (1,181) | 42.1% | weak discrimination |
| LB_374581869ba8 1sg present | 1.00 (392) | 34.5% | weak discrimination |
| LB_27c64c1f7e4b polite address | 1.00 (947) | 86.1% | tautological |

Same rules, full 8,972-pool run (so both halves, weights from the whole pool): comparative
**0.76 / n=310 / 75.3% usable**; possessive 0.61 / n=1,968 / 92.6% (weak agreement); `een`
prohibition 1.00 / n=29 / 100% provisional; `-en` plural 5.3% and 1sg present 8.4% stay vacuous.

**Discrimination is sample-sensitive and I do not hide it.** The `-en` plural rule measured 5.3%
(full-pool samples) but 31.5% (held-out samples) with the same code and card. Roughly 100 satisfied
samples x 3 perturbations gives a standard error of a few percent, and the satisfied-pair composition
changes between halves; a single number per rule over-claims. Reported as measured, no interval yet.

**Support floors.** `LB_55471d5dc4b4` passed all three axes on **n=1 check**. `usable` now requires
n>=30 decided checks and >=20 attempted perturbations; quality-passing-but-thin rules are reported as
`provisional`. Without the floor, gate G27 could be satisfied by a single lucky pair.

**The 0.39% false-penalty claim was Goodhart.** That run allowed only rules with agreement >= 0.65,
and the rule carrying 2,041 of its 3,981 checks was `LB_d121073f582e` (`^.*en$`, discrimination 5.3%)
- a rule that never bites, so of course it never penalised a professional translation. With
discrimination measured and the vacuous rules still admitted, the honest end-to-end number over all
8,972 CC0 pairs is: **3,836 pairs scored, 3,269 labeled pairs, 4,319 decided checks, 14.9% of
professional translations receive a negative grammar reward.** That 14.9% is the real disagreement
rate of rules that bite (comparative violates 29%, possessive 30%); part of it is my un-encoded
exception set (Luxembourgish uses the definite article for inalienable possession: `their schools` ->
`an d'Schoul`; `under her leadership` -> `an der Verantwortung`), part is genuine over-breadth.

Consequence for P1: `violated` must be treated as a **noisy** negative whose weight is the measured
agreement, never as ground truth. Labels on the `satisfied` side carry precision = agreement.

---

## 12. Support expansion, the possessive decomposition, and the first reward-side RL smoke (2026-11-26)

### 12.1 Support expansion on the second corpus (license split kept)

The `een`-prohibition rule (LB_363a21684c8f, "the indefinite pronoun `een` may not open the sentence")
had passed all three quality axes on n=1 check. Sourcing more support: 391 sentence-initial
`one/someone/somebody` pairs exist, but only 29 are CC0 (techgov); 362 come from luxalign (CC BY-NC
4.0). The measurement was therefore run as a **separate pool**, `artifacts/audit/condonly_luxalign.json`
and `artifacts/audit/rule_quality_condonly_luxalign.json`, not merged into the CC0 numbers. Result: the
prohibition moved from provisional to **USABLE (agreement 0.95, n=361, discrimination 100%)**, and
kinship borrowing (LB_2ebda66afe75) also certified. Two further rules were VACUOUS, one TAUTOLOGICAL —
adding data did not rescue them.

### 12.2 My first violation decomposition was wrong, and why the buckets are nameless

To fix the possessive rule's 30% disagreement I first bucketed its violations by "article + capitalized
token = proper-name recast". That bucket was **invalid**: Luxembourgish capitalizes every noun, so the
pattern also matched `den Attentater` ("the attackers") and `d'Nathalie Jacoby`. `scripts/
violation_decomposition.py` now takes buckets as `NAME=REGEX` arguments and adds no interpretation,
because interpretation is exactly what produced the false finding.

The real decomposition (rule LB_946a24dd25d6, 8,972 CC0 pairs, 1,209 satisfied / 759 violated):

| bucket (surface pattern, no claim attached) | n | share of violations |
|---|---|---|
| possessive article present, other possessor number | 396 | **52.2%** |
| residual | 363 | 47.8% |

Over half of the "violations" contained a possessive article at all — with the *other* number. The
cause is linguistic, not translator error: Luxembourgish "their" admits both `är-` and `hir-` variants
(`hirer Zäit` = their time), so **possessor number is undecidable for that person**, while "our" ->
`eis-` is decidable. Encoding the undecidable half manufactured false violations.

Fix: the rule is now two constraints with explicit claims —
- `our/ours` -> `\beis(e|en|em|er)?\b` (decidable: no homophonous singular),
- `their/theirs` -> *some* possessive (`är-|hir-|eis-`), description stating that possessor number is
  **not** claimed for this person.

Held-out agreement went 0.64 -> **0.74 at n=1,180**, discrimination **95.2%** -> certified **USABLE**
(full pool 0.70 / n=1,973). This is a data-driven correction of the form list, not a loosening of the
gate: the same thresholds (agree>=0.65, disc>=0.5, not self-restating, n>=30, >=20 perturbations) were
applied before and after.

### 12.3 The label gate now depends on the quality matrix

`gated_pipeline_measurement.py --quality` admits only rules the three-axis matrix certifies `usable`
(`--allow-provisional` opts into thin support). This dropped `LB_d121073f582e` (discrimination 5.3%: a
rule that cannot bite) and the then-provisional `een` rule. Honest yield on 8,972 CC0 pairs:

```
allowed rules           2 (comparative, possessive)
pairs_scored            2,228      labeled_pairs      1,585
decided_checks          2,283      false_penalty_rate   28.9% (weighted 29.2%)
```

Previously the same pipeline reported 3,367 labels and 12.8% false penalties; ~2,000 of those checks
belonged to a rule with 5.3% discrimination. A constant reward cannot generate a false penalty, so the
low rate was an artifact — recorded in §11 and now structurally impossible.

### 12.4 First P1 reward-side smoke (no training, verl deliberately absent)

`scripts/p1_reward_smoke.py` samples k rollouts per source from an OpenAI-compatible policy, scores each
with the reliability-weighted grammar reward over **only the certified rules**, and reports what decides
whether an RL launch is worth its cost. `gemma3:12b`, 24 groups x 3 rollouts,
`artifacts/p1/reward_smoke_gemma3_12b.json`:

| metric | value | reading |
|---|---|---|
| `nonzero_spread_rate` | **29.2%** | 71% of GRPO groups have zero advantage -> zero gradient, wasted compute |
| `mean_spread` / median | 0.583 / 0.0 | bimodal: groups are either saturated or informative |
| `reference_penalized_rate` | **12.5%** | measured noise floor: professionals losing reward on these 2 rules |
| `policy_minus_reference` | **-0.056** | a 12B generalist is nearly indistinguishable from professionals on this rule set |
| `abstain_rollout_rate` | 0.0 | both certified rules decide on every pair (source conditions fire) |

Capability is the first-order effect, found by running it: `qwen2.5-coder:7b`, `qwen2.5:14b` and
`gemma3:4b-it-fp16` do not produce Luxembourgish (e.g. `Ou sinn wëssen dëmmët dëmmen dëmmen?`), score
-1.0 on **every** rollout, and yield **zero** advantage spread. RL cannot learn from a policy with no
target-language competence; this is the empirical argument for the SFT warmup in P1, not a stylistic
preference. `qwen3:32b` and `gpt-oss:20b` did not answer within 400 s on this host.

### 12.5 A failure mode agreement and discrimination both miss: false-satisfy

One `qwen2.5-coder:7b` rollout earned **+1.0** while being gibberish, because `\bmeng(en)?\b` matched
the verb **mengen** ("to mean/think"), not a possessive article:

```
EN  How much does AI already meddle with our everyday life?
LB  Wéi vill mengt Dir, mengen AI schonn eis alleschdeemschttes Liewen?   -> r_grammar = +1.0
```

Agreement-vs-professionals and discrimination-vs-perturbations are both blind to this: the check never
penalized anything, it silently *rewarded* nothing-of-interest. `scripts/match_context_audit.py`
quantifies it by classifying the token after each match. In professional references (1,603 possessive
matches): **79.9% followed by an uppercase token, 19.2% lowercase, 0.9% digit/end**. The lowercase
minority is legitimate — an adjective sits between possessive and noun (`eis elektronesch ...`,
`sengem éischte`, `hire privaten`) — so the tempting fix (require a following capitalized noun) would
manufacture **19.2% false violations** on professional text. Verdict: **not fixable at surface level**;
it needs POS/morphology, i.e. the P3 argument.

Consequence for reward design, stated as a requirement: the grammar term must never be the whole reward.
`R = w_mt*R_mt + w_g*R_grammar` with `w_mt > 0` — the MT term is what stops a policy from farming
grammar checks with content-free output. An earlier draft of the P1 plan set `w_mt = 0`; that is now
ruled out by measurement.

## 13. README rewritten as a usage report (2026-11-26), and what that audit found

The README is now the operating manual for the measurement stack: environments, machine reachability,
license policy, every pipeline command, the three-axis certification protocol, the gate table, artifact
inventory, and the failure modes. Every number in it was re-read off the artifacts rather than from memory.
Writing it surfaced four things:

1. **A cited file did not exist.** I wrote `scripts/check.sh` (it belongs to a different project in this
   session). The trux env simply has no `ruff`. Removed.
2. **Numbers quoted from memory were wrong.** Possessive discrimination is **91.4%** in the tracked matrix
   artifact (92.6% came from a different perturbation sample), and my verdict tally mis-binned two rows.
   Artifact-exact tally on the CC0 pool (`artifacts/audit/rule_quality_conditional.json`, keys
   `n_agreement` / `perturbations_tested` / `predicate_restates_constraint`):
   usable 2 · provisional 2 · vacuous 3 · self-restating 1 (`LB_27c64c1f7e4b`, n=1,993, disc 70.2%) ·
   below-floor 1 (`LB_2ebda66afe75`, 0.595 at n=33).
   On the CC BY-NC pool a **different** pair certifies (`LB_363a21684c8f` 0.945/n=361,
   `LB_2ebda66afe75` 0.762/n=38) while the comparative rule **drops to 0.626 and fails**. Certification is
   pool-dependent; the README now states this instead of implying portability.
3. **Pool tags are `techgov_lu_en` (8,972 CC0) and `luxalign_v3_lb_en` (39,464 CC BY-NC).** Community
   Tatoeba-style rows were never ingested; the README previously implied a third pool that does not exist
   in the file.
4. **A real dead code path, found by running the README example.** `main.py verify` crashed with
   `AttributeError: 'VerificationReport' object has no attribute '__dict__'`: both `VerificationReport` and
   `RuleCheckResult` are `@dataclass(slots=True)`, and slots dataclasses have no `__dict__`. The subcommand
   had never worked since the slots conversion. Fixed with `dataclasses.asdict`, plus a CLI-level
   regression test that runs the subcommand and asserts both certified rules decide `satisfied` in the
   right order (so the weight-alignment contract is covered too). **Tests: 31 passed.**

Also rebuilt the training dataset on the current labels so the README's commands reproduce what is on disk:
`data/processed/gated_condonly/` = train 1,427 / valid 158 / valid_rules_heldout **0** (100% CC0-1.0).
The empty rule-held-out split is reported, not hidden: with 2 certified rules, a rule-level held-out split
is undefined, so cross-rule generalization remains unmeasurable until coverage grows.

New limitation recorded (measured, not inferred): `main.py retrieve --top-k 3` on
"Our children are taller than their parents." returns the possessive card plus two irrelevant
phonotactics cards, all tied at 0.336, and **misses the comparative card**. Measurement passes named
`--card-ids`, so no published number depends on retrieval; the P1 agent trajectory will.

## 14. Expanding certified rules 2 to 8, and the negative result that killed "coverage is the bottleneck" (2026-11-27)

The P1 plan named certified-rule count (2) as the binding constraint on reward signal density. This
section records the attempt to lift it, what was built, and the measurement that contradicted the
hypothesis in its strong form.

### 14.1 Why hand-authoring more conditionals did not work

`data/rules/source_conditionals_v2.json` fixed the present-perfect patterns (window 120 chars,
loan-verb participles `-eiert/-iert`, `gewiescht/gewuer`) and added four hand-authored conditionals
(past perfect, proper-name article, 2sg clitic `s du`, `eng` + cardinal). Applied to the card set
(10 constraints accepted, 0 rejected), audited on the fit half and certified on the held-out half:

| rule | held-out agreement |
|---|---|
| perfect (fixed) | 0.402 |
| past perfect | 0.30 |
| 2sg clitic `s du` | 0.23 |
| `eng` + cardinal | 0.08 |
| possessive (previous) | 0.74 |
| comparative (previous) | 1.00 |

Every new rule was refuted. The violation decomposition of the perfect rule explains why without any
appeal to intuition: of 388 "violations", 34.8% contained a `hu*` auxiliary, 26.3% a copula, 14.7% a
`ge-` word, so about 76% of them already had the correct morphology somewhere in the clause. Fixing
that took agreement from 0.215 to 0.402 - still far below the 0.65 floor. **The binding property is
target-form obligatoriness, not regex quality.** English tense/aspect at phrase level maps many-to-many
onto Luxembourgish, so a source cue does not license a single required target form. A rule can be
perfectly implemented and still be wrong as a reward.

One candidate died at admission for a harness reason worth recording: the proper-name article rule
depends on capitalisation, and `grammetarl/validator.py:82` `normalize_text` NFKC-normalises and
**casefolds** before any pattern runs. Capitalisation-dependent constraints are untestable in this
harness; the admission test caught it rather than a green number hiding it.

### 14.2 Screening instead of authoring

`scripts/screen_source_conditionals.py` screens candidate conditionals against the corpus directly
(screen on the fit half, certify on the held-out half, ship only if **both** halves clear the floor).
22 candidates in `data/rules/screened_conditionals.json`, mined from the corpus by reading which
English function words have stable Luxembourgish counterparts, then corrected from measured violations:

- `than` needed **`eweéi/ewéi`**: agreement 0.49 to 0.76 (LLB writes `ewéi` for `than` frequently).
- the `-est` superlative source condition fired on `interest`, `request`, `suggest` (they end in
  `-est`); requiring `the X-est` fixed the trigger, agreement still 0.43 because LLB often uses the
  analytic superlative.
- `as ... as` matched across unrelated `as` tokens in one sentence; restricted to an adjective list,
  agreement still 0.15, so the candidate was dropped.

The screening probe also had a bug, found because `tested` came out far below the number of samples:
`verifier_discrimination.corrupt_all` defaults to `min_instances=2` and refuses to perturb when the
target contains a single instance ("single-token corruption is enough"). For function-word cues that is
almost always the case, so 40 sampled targets produced 9 tested perturbations and the rules were
labelled "support too thin" by a probe limitation, not by the data. Screening now calls it with
`min_instances=1`, and `tested` reached 120 per candidate.

Certified on both halves (fit, held-out agreement, held-out n = pairs where the source condition fires):

| mined rule | fit | held-out | disc (matrix) |
|---|---|---|---|
| `negation_not` (not/n't -> net/keen/keng/...) | 0.92 / 581 | 0.89 / 177 | 96.7% |
| `cond_if` (if -> wa/wann/falls) | 0.82 / 281 | 0.81 / 115 | 100% |
| `comp_that` (that + subject -> datt/dass) | 0.75 / 326 | 0.75 / 103 | 100% |
| `every_each` (every/each -> all/jidd) | 0.76 / 170 | 0.77 / 86 | 100% |
| `than` (than -> weéi/ewéi) | 0.76 / 172 | 0.85 / 71 | 96.7% |
| `without` (without -> ouni) | 0.69 / 94 | 0.72 / 43 | 100% |

The screening probe's own discrimination column is **1.00 for all 22 candidates**, including the garbage
ones; the 96.7-100% above comes from the three-axis matrix (`rule_quality_condv3_heldout.json`), which
corrupts all instances. See 14.6 - this axis does not do the work people assume it does.

`also_too` scored 0.74 on the fit half and **0.64 on the held-out half, so it is refuted**: selecting on
the fit half alone over-fits, which is exactly what the two-half criterion is for. Rejected by genuine
divergence: `superlative` 0.42, `ability_can` 0.43 (fit 0.54), `oblig_must` 0.63, `reflexive_self` 0.27,
`while_durative` 0.12, `both` 0.06, `as_as` 0.08. LLB restructures the clause instead of using a
never/both-style adverb, so those source cues license no single target cue.

### 14.3 Mined rules are cards, with provenance that says they are mined

`scripts/make_mined_cards.py` emits `data/rules/mbg_cards_mined.jsonl` (6 cards). They carry
`source.source_id = CORPUS_MINED_TECHGOV`, an excerpt that says "no book evidence: derived from corpus
agreement, not extracted", `rule_status = "variable"` (the schema has no `statistical`), and the
measured agreement in the statement. Their examples are corpus instances taken from the **fit half**
with the licence and pair id in the note, so the card self-test does not leak the held-out certification.
Merged set: `data/rules/mbg_cards_condv3.jsonl` = 87 book cards + 6 mined = 93.

Three-axis matrix on the held-out half (`artifacts/audit/rule_quality_condv3_heldout.json`):
**8 rules USABLE** - `LB_9789dfb15623` comparative, `LB_946a24dd25d6` possessive, and
`LB_MIN01..06`. Discrimination 96.7-100%. Still 1 vacuous (`LB_55471d5dc4b4`), 1 tautological
(`LB_27c64c1f7e4b`), and several weak-on-discrimination suffix rules.

End-to-end over all 8,972 CC0 pairs with only the certified rules (`--quality` gating):

| | 2 rules | 8 rules |
|---|---|---|
| pairs scored | 2,228 | 3,270 |
| pairs labelled | 1,585 | **2,555** |
| decided checks | 2,283 | 3,907 |
| professional translations receiving a negative grammar reward | 28.9% | **22.3%** (weighted 23.9%) |

Coverage up a third, false penalty rate down. On the label side the expansion was an unqualified win.

### 14.4 The RL readout went the other way

Same protocol as the previous smoke (gemma3:12b, 24 prompts, 3 rollouts, first rollout greedy, rest at
temperature 0.9), 2 certified rules vs 8:

| | 2 rules | 8 rules |
|---|---|---|
| nonzero_spread_rate | 29.2% | **16.7%** |
| mean spread | 0.583 | 0.333 |
| reward mean | 0.694 | 0.833 |
| reference mean | 0.750 | 0.833 |
| policy minus reference | -0.056 | **0.000** |

More rules made the GRPO advantage **rarer**, and the policy now scores exactly like the professional
reference. Before interpreting that as a regression, the confound: `p1_reward_smoke.py` selects prompts
as "the first pairs whose reference has a decided check", so changing the rule set also changes the
prompt set, and the two runs were not on the same pairs.

`scripts/p1_signal_by_rule.py` removes the confound by re-scoring the **stored hypotheses** offline
(zero model calls) under different rule subsets. It first reproduced the online number exactly
(29.2% on the old log with the 2-rule set), which validates the offline path:

| hypothesis set | 2 rules | 8 rules |
|---|---|---|
| old prompt set (selected by 2 rules) | 29.2% | 29.2% |
| new prompt set (selected by 8 rules) | 12.5% | 16.7% |

So on identical hypotheses the 6 mined rules changed the spread rate by at most +4.2pp (one group,
`cond_if`), and the apparent 29.2% to 16.7% drop was **mostly the prompt set**.

Per-rule signal on the new prompt set (policy violated rate vs the professional reference's, same rules):

| rule | policy n | policy viol | ref viol | gap | groups where it creates spread |
|---|---|---|---|---|---|
| `LB_946a24dd25d6` possessive | 36 | 11.1% | 16.7% | **-5.6%** | 3/24 |
| `LB_MIN02_cond_if` | 3 | 66.7% | 0.0% | +66.7% | 1/24 |
| `LB_MIN04_negation_not` | 12 | 0.0% | 0.0% | 0 | 0/24 |
| `LB_MIN01_comp_that` | 12 | 0.0% | 0.0% | 0 | 0/24 |
| `LB_MIN05_than` | 9 | 0.0% | 0.0% | 0 | 0/24 |
| `LB_9789dfb15623` comparative | 9 | 0.0% | 0.0% | 0 | 0/24 |

This is the whole story in one table. A high-support certified rule can be **saturated**: `negation_not`
fires on 12 rollouts and the 12B satisfies it every time, exactly like the professionals, so it adds
decided checks and zero advantage. The only rule with a positive gap is `cond_if`, on n=3 checks. And
the possessive rule's gap is **negative** - it penalises professional translations (16.7%) more often
than the model (11.1%) - which is the same noisy-negative fact from section 11, now visible inside the
reward.

### 14.5 What the P1 plan now has to say

The claim "certified-rule count is the binding constraint on signal density" is **falsified in its strong
form**: going 2 to 8 certified rules did not raise advantage coverage. The measured levers are:

1. **Rule difficulty relative to the policy.** Select/derive rules where the policy violates and the
   professional reference satisfies (positive gap). Saturated function-word cues are useful for labels
   and for the eval, useless for GRPO advantage.
2. **Prompt sampling.** Advantage is a within-group property; which pairs are sampled dominates the
   measured rate. `p1_reward_smoke.py` now takes `--prompts-per-rule N` to stratify selection so every
   certified rule gets its own groups instead of whatever the first decidable pairs happen to contain.
3. **Statistical power.** 7/24 vs 4/24 is not a difference. The smoke now emits a Wilson interval
   (`nonzero_spread_rate_ci95`); at n=24 it is about +/-18pp, larger than every effect reported above.
   A stratified run (8 rules x 6 prompts) is the first experiment in this line with a defensible n.

Operational fact found while polling that run: on `/raid` a log file created by a background process is
not visible to a new exec for roughly 30-60s. Polling with `ls`/`grep` on the log looks like a failed
launch; `ps` shows the process immediately. Do not relaunch on the basis of a missing log - that is how
a job gets run twice.

### 14.6 The discrimination axis does no work on cue-type rules

`artifacts/audit/screen_conditionals_{fit,heldout}.json` report `discrimination = 1.00` for **all 22**
candidates, including the ones the corpus refutes outright: `both` 0.06 agreement, `as_as` 0.08,
`while_durative` 0.12. A rule with essentially zero agreement and perfect discrimination is not a paradox:
discrimination asks whether the check flips when the **target cue** is corrupted, and for a
`source_conditional` whose check is "target contains one of these cue tokens", corrupting that token flips
the check by construction.

What discrimination was built for is suffix/surface shape rules (`-en` plural at 5.3% = cannot bite). For a
cue rule it degenerates: a single-token cue list is self-certifying, and the only reason `negation_not` and
`than` read 96.7% instead of 100% in the matrix is that their cue lists contain synonyms, so corrupting one
leaves another valid.

Consequence for the measurement contract, stated plainly: **for cue-type `source_conditional` rules the
three axes collapse to two effective ones** - agreement against professional translations, and
non-tautology. The support floors (`n >= 30` decided checks, `>= 20` perturbations) and the two-half
protocol are therefore carrying the entire burden of proof for the 6 mined rules, and a future hardening
step is a discrimination probe that breaks the *source* condition instead of the target cue (i.e. test
whether removing `if` from the English side makes the constraint abstain - an applicability-sensitivity
axis, not a target-flip axis).

---

## 15. Mining rules from the policy's own errors: three candidate families, zero reward-ready rules (2026-11-27)

§14.5 concluded that rule *coverage* is not the binding constraint on GRPO signal and named the next
lever: mine constraints from the difference between stored policy rollouts and professional references.
This section reports executing that lever. It produced one new tool (`scripts/mine_policy_gaps.py`),
one new metric (gradient alignment), one new gate (G31), and a negative result on every candidate
family tried. The negative result is the finding.

### 15.1 Unbiased prompt sampling: the first number was an artifact, the corrected one is 6.2%

To mine gaps I first needed rollouts over prompts *not* selected to make a rule fire. Added
`--breadth N` to `scripts/p1_reward_smoke.py` (card-independent stride over the whole pool). 140 prompts
x **1** greedy rollout -> `artifacts/p1/rollouts_gemma3_12b_breadth.jsonl`, and it printed

    nonzero_spread_rate = 0.0

**That 0.0 was arithmetic, not data.** A GRPO group with one rollout has max(rewards) - min(rewards) = 0
by definition, so a corpus run (`--rollouts 1`, built to mine errors, not to measure signal) cannot
produce a coverage number at all. I quoted it in the commit message, README §0/§8.2 and §15.8 before
checking. The tooling now refuses to produce it again: `p1_reward_smoke.py` writes
`"nonzero_spread_rate": null` plus a `spread_undefined` string when `--rollouts < 2`, and
`p1_signal_by_rule.py` hard-stops on a log with one hypothesis per group
(`tests/test_p0_gates.py::test_spread_is_reported_as_undefined_not_zero_below_two_rollouts`,
`::test_signal_script_refuses_a_one_rollout_log`).

Re-measured properly: 80 stride-sampled prompts x 3 rollouts, same 8 certified rules
(`artifacts/p1/reward_smoke_gemma3_12b_breadth3.json`, `artifacts/p1/signal_condv3_breadth3.json`):

| | stratified (42 groups x 3) | unbiased (80 groups x 3) |
|---|---|---|
| groups with nonzero advantage | 5/42 = 11.9% [5.2, 25.0] | **5/80 = 6.2% [2.7, 13.8]** |
| rollouts reaching >=1 verdict | 126/126 = 100% (by construction: prompts were chosen where rules decide) | **75/240 = 31.2%** |
| prompts where no rule fires at all | 0 | **55/80 = 68.8%** |
| pairs tied on reward | 89/120 | 155/240 |
| reference penalized | 14.3% | 11.3% |

Two conclusions replace the retracted one. (a) Stratification inflates coverage (11.9% vs 6.2%) but the
intervals overlap, so prompt selection is a bias of maybe ~6pp, not the whole story. (b) **Before
saturation there is applicability**: on card-independent prompts the rule set simply does not apply to
two thirds of sentences, which is a different failure than "the rules are saturated", and it says the
reward term is a *narrow* term no matter how it is weighted.

### 15.2 Mining tool design, and the bug that motivated it

First version mined (cue -> form) pairs out of the rollout corpus itself. Two bugs, both caught by
looking at the numbers rather than the exit code:

- hypothesis token sets were stored as `{hyp_text: token_set}`, so `form in ht` tested dict *keys* and
  every policy realisation rate printed 0.00. The guard now in the script
  (`every policy realisation rate is 0.0: the hypothesis token sets are broken, not the policy`) turns
  that class of bug into a hard stop instead of a plausible-looking table.
- corpus-level statistics (n = 6-17 pairs per cue) cannot estimate obligatoriness. Mining was restructured
  into two stages with different estimators: **stage A on the pool** (fit half only, index < `--fit-boundary`,
  so the existing two-half certification still measures unseen data) requiring pool support >= 30,
  professional agreement >= 0.5 and cue specificity `lift = p(form|cue)/p(form|cue absent) >= 2`;
  **stage B on the rollout corpus**, used only to rank by the policy-vs-reference gap.

The lift axis is not decoration: an earlier corpus-only run promoted `has -> den` with a 0.50 gap, but
`den` is a dative article professionals write roughly half the time regardless of the cue (lift ~1).
Same lesson as the 0.39% false-penalty Goodhart correction in section 12: a check that fires regardless
of the condition is not a check.

### 15.3 Family 1 - single-token cue, positive form: 0 certified

155 stage-A candidates -> 4 with policy gap >= 0.10 -> screened on both halves
(`artifacts/audit/screen_policygap_{fit,heldout}.json`):

| candidate | fit n | fit agree | heldout n | heldout agree | verdict |
|---|---|---|---|---|---|
| `after -> no` | 187 | 0.52 | 128 | 0.51 | refuted (agreement) |
| `language -> fir` | 71 | 0.52 | 11 | 0.45 | no-support / refuted |
| `if -> wann` | 281 | 0.60 | 115 | 0.63 | refuted (agreement) |
| `years -> joer` | 168 | **0.68** | 78 | 0.64 | fit USABLE, **refuted held out** |

The gap table itself is the diagnosis: for the high-obligatoriness cues (`an, fir, et, net, och, méi,
wéi, ouni, wann`) the policy realisation rate already equals the professional rate. Cue-level
obligatoriness is **saturated** for this 12B, which is why 155 pool-quality candidates collapse to 4
teachable ones and then to 0 certified.

### 15.4 Family 2 - ordered pairs: structurally impossible under the agreement floor

`--ordered` mines patterns `t1 .* t2` (order-type constraints, where the policy genuinely differs from
professionals). Result: **0 candidates** at support >= 60, agreement >= 0.45, lift >= 2.5. The reason is
arithmetic, not tuning: a multi-element constraint is satisfied by professionals strictly less often than
each element alone, so `P(t1 and t2 in order) ~ 0.3` while the certification floor is 0.65. Raising
agreement to 0.65 requires both elements individually near-obligatory - which is family 1 again.

Two ways out, both unresolved and both honest to name:
- *conditional applicability* ("if the cue fires **and** both tokens appear, they must be ordered"):
  lifts agreement by construction, but applicability defined on the hypothesis is gameable - the policy
  can dodge the check by dropping a token. Any such rule needs an abstention penalty in the reward.
- *morphology/POS* (P3): order constraints in Luxembourgish are about finite-verb position and verb
  clusters, which cannot be expressed as token-order regexes over surface forms at all.

### 15.5 Family 3 - exclusions (professionals avoid it, policy uses it): mechanically works, semantically empty

`--negative` mines `negate: true` candidates with professional violation <= 5% under the cue and policy
violation >= 20%: 1,287 candidates, 739 with gap >= 0.10. Mechanically these are the ideal RL rules -
they fire on the policy and not on the reference, so they cannot be saturated or anti-professional.
Inspecting the shipped top-24 kills them as *grammar*:

    during -> während   so -> just      minister -> iwwer   every -> vum
    would -> wär        with -> vill    life -> iwwer       been -> all

These are near-synonyms and frequency variants professionals happen not to prefer. Penalising them is
reference imitation - exactly what a chrF/BLEU term already does - wearing a grammar label. Decision
recorded: **do not ship exclusion candidates mined this way**. They stay in
`data/rules/policy_gap_candidates.json` as evidence, not in a card set.

### 15.6 Excluding anti-professional rules lowers spread: spread was never the objective

Offline re-scoring of the identical 126 stratified hypotheses (`scripts/p1_signal_by_rule.py`, zero model
calls), 8 rules vs the 5 with non-negative policy-vs-reference gap:

| rule set | groups | nonzero spread |
|---|---|---|
| 8 certified rules | 42 | 5/42 = 11.9% [5.2, 25.0] |
| 5 rules (negative-gap rules dropped) | 42 | 3/42 = 7.1% [2.5, 19.0] |

Dropping the rules that penalise professionals harder than the policy *reduces* advantage coverage.
Spread counts any within-group reward difference, including differences that point the wrong way. So
"exclude anti-professional rules" is a correctness lever, not a signal-density lever.

### 15.7 Gradient alignment: the missing measurement, and it reads badly

New metric in `scripts/p1_signal_by_rule.py`: for every within-group rollout pair, compare the sign of
the grammar-reward difference with the sign of the lexical-similarity-to-reference difference (token F1
over normalised text). Concordant = the reward ranks rollouts the way a professional reference would.

    42 groups x 3 rollouts, 8 certified rules
    concordant = 2   discordant = 8   -> 20.0%  [95% CI 5.7, 51.0]
    ties: reward-only 89, similarity-only 0, both 27

Per-rule (only pairs where that rule's verdict differs): `cond_if` 1 concordant / 1 discordant;
`comp_that` 0/2; `every_each` 0/2; `possessive` 0/2; `without` 1/1.

Read it carefully. n = 10 decisive pairs, so the interval is huge and the discordance is *not*
significant. What is significant is the shape: 89 of 120 pairs tie on reward (the rules are blind), and
in the decisive remainder the reward more often opposes than supports the professional ranking. Combined
with 15.3 (saturated cues) this says the P0 certified set is **not reward-ready**. It is a label and
evaluation asset; as a training term it would mostly either do nothing or push against professional usage.

New gate G31 encodes that: PASS requires concordance >= 0.5 with >= 10 decisive pairs. Current state
**FAIL**.

**Superseded in part by 15.9:** "more often opposes than supports" was reading a 2-vs-8 split as a
direction. On the unbiased 80-prompt set the same rules score 6-vs-3 (66.7%), the opposite sign, and
neither rejects the 0.5 null. The defensible statement from 15.7 is the blindness (89/120 and 155/240
ties), not the direction. Gate tally with the documented command (which must include `--policy-endpoint`, otherwise two
policy-dependent gates silently change state): **PASS=29 FAIL=3 WARN=10** (the three FAILs: G21, G24,
G31). Tests: 35 passed.

### 15.8 What this changes in the plan

1. The next mining criterion is **concordance**, not gap: screen candidates by whether their verdict
   differences rank rollouts like the reference does. It is free offline once rollouts are stored
   (`scripts/p1_signal_by_rule.py`), and it is the axis that predicts useful RL signal.
2. The grammar weight band (G29) stops being belt-and-braces and becomes load-bearing: with an MT term
   present, a rule set that is blind on 89/120 pairs is harmless; a rule set that is discordant on the
   rest is only survivable because `w_grammar < w_mt + w_tool + w_trace`.
3. Surface cue mining is exhausted across three families. Teachable surface signal for a competent
   generalist is nearly gone; what remains needs morphology (P3) or a weaker policy (SFT warmup makes
   the policy's errors grammatical rather than stylistic, which is the only regime where these rules
   could bite).
4. Artifacts this section needs: `artifacts/p1/rollouts_gemma3_12b_breadth.jsonl` +
   `corpus_gemma3_12b_breadth.json`, `artifacts/p1/policy_gap_mining.json`,
   `data/rules/policy_gap_candidates.json`, `artifacts/audit/screen_policygap_{fit,heldout}.json`,
   `artifacts/p1/signal_condv3_stratified.json` (now with `alignment` + per-rule alignment),
   `artifacts/p1/signal_condv3_strat_pos5.json`.


---

## 16. Concordance screening, and why the null is the placebo band (2026-11-27, same session)

### 16.1 My alignment claim did not survive a second prompt set

`p1_signal_by_rule.py` re-run on the unbiased breadth corpus (80 groups x 3 rollouts,
`artifacts/p1/signal_condv3_breadth3.json`):

    stratified (42 groups): concordant 2, discordant 8 -> 20.0% [5.7, 51.0], p vs 0.5 = 0.109
    unbiased   (80 groups): concordant 6, discordant 3 -> 66.7% [35.4, 87.9], p vs 0.5 = 0.508

Same eight rules, opposite point estimates, neither significant. So the sentence I wrote in 15.7 - "the
reward more often opposes than supports the professional ranking" - is not supported by this data, and it
is retracted here and in README 12.18. What survives:

- the rules are **blind**: 89/120 (stratified) and 155/240 (unbiased) pairs tie on reward;
- on unbiased prompts **68.8% of prompts get no verdict at all** from the 8 rules, and only 31.2% of
  rollouts are judged - the reward term is narrow before it is ever saturated;
- the negative-gap rules replicate across prompt sets (`negation_not` -10pp stratified, -40pp unbiased),
  so excluding them from the reward is still right; `cond_if`'s +44pp did NOT replicate (0.0pp unbiased,
  it fires on 3 rollouts there).

### 16.2 `scripts/screen_concordance.py`: screening candidates by alignment, with the right null

Applying the alignment axis to the 46 **candidates** (22 screened source-conditionals + 24 policy-gap
exclusions, union in `data/rules/screened_conditionals_all.json`) exposed a second problem: the natural
null, concordance = 0.5, is wrong for a LEXICAL proxy. Token F1 to the reference rewards sharing surface
tokens, so a constraint that merely requires (or forbids) any frequent target token can score concordant
without encoding grammar.

The tool therefore runs a **placebo**: same source cue, same constraint kind, target forms replaced by
unrelated target tokens drawn from the same cue-applicable slice (present in both references and rollouts,
so the placebo actually fires). Reported per candidate: `decisive_pairs`, `concordance`, exact binomial
p vs 0.5, Wilson and cluster-over-groups CIs, `placebo_pooled_concordance`, `placebo_gap`, and a Fisher
exact p for candidate vs placebo.

Measured placebo behaviour (12 placebos/candidate, cue-targeted corpus): pooled placebo concordance is
nowhere near a fixed 0.5 - it ranges **0.14 to 0.83 across candidates** (`while_durative` placebos 0.83,
`pg_during_wär` 0.14). Any concordance number quoted against 0.5 is therefore uninterpretable; the
comparator is each candidate's own placebo band.

### 16.3 Cue-targeted prompts buy the power that stratified prompts could not

On the stratified log (built for the 8 certified rules), each of the 46 candidates had **0-2 decisive
pairs**, so alignment could not rank them at all. Selecting prompts by the CANDIDATE cue
(`--cue-specs ... --prompts-per-cue 6 --rollouts 6`, `artifacts/p1/rollouts_gemma3_12b_cue6.jsonl`)
raises applicable groups from 0-5 to 3-11 and, with 6 rollouts (15 pairs/group), decisive pairs per
candidate reach **up to 20**. Pairs scale as k(k-1)/2: the same prompts with 6 rollouts give 5x the
pairs of 3 rollouts. The screen also reports `groups_needed_for_decisive_floor` so the budget question
is explicit instead of guessed.

Results on the partial cue6 corpus (72 groups, 410 hypotheses; the full log is being regenerated):

| candidate | decisive pairs | concordant/discordant | placebo conc | placebo pairs | Fisher p | agreement fit / held-out | verdict |
|---|---|---|---|---|---|---|---|
| `than` | 6 | 6 / 0 (1.00) | 0.22 | 146 | 0.000 | 0.70 / 0.85 | right direction, **5-6 pairs is under the 10-pair floor** |
| `pg_government_gouf` | 5 | 5 / 0 (1.00) | 0.35 | 90 | 0.007 | 0.95 / 1.00 | right direction, too few pairs |
| `pg_during_während` | 5 | 4 / 1 (0.80) | 0.21 | 83 | 0.012 | 1.00 / 1.00 | right direction, too few pairs |
| `ability_can` | 19 | 18 / 1 (0.95) | 0.62 | 93 | 0.006 | 0.63 / **0.43** | plenty of pairs, but not what professionals do |
| `comp_that` | 17 | 9 / 8 (0.53) | 0.56 | 195 | 0.804 | 0.75 / 0.75 | indistinguishable from its placebo |
| `only` | 16 | 8 / 8 (0.50) | 0.50 | 180 | 1.000 | 0.54 / 0.55 | indistinguishable from its placebo |
| `cond_if` | 12 | 0 / 12 (0.00) | 0.30 | 93 | 0.033 | 0.83 / 0.81 | significantly **anti**-aligned (the rule whose gap was +44pp) |
| `pg_with_vill` | 9 | 0 / 9 (0.00) | 0.49 | 177 | 0.004 | 0.96 / 0.95 | significantly anti-aligned |
| `both` | 5 | 5 / 0 (1.00) | 0.44 | 68 | 0.021 | **0.07** / 0.06 | beats placebo while being nonsense: why agreement stays a gate |

No candidate clears every axis yet (agreement fit and held-out >= 0.65, >= 10 decisive pairs,
concordance >= 0.5, placebo gap >= 0.1, placebo Fisher p <= 0.05) - but the reasons differ, and that is
the useful part:

1. **Power-limited, not refuted (at 5-6 pairs)**: `than`, `pg_government_gouf` and `pg_during_während`
   beat their placebo band (p = 0.000 / 0.007 / 0.012) with agreement 0.70-1.00, and failed only the
   10-decisive-pair floor. **That reading was wrong, and 16.4 shows it.**
2. **The axes disagree.** `ability_can` has 19 decisive pairs and concordance 0.95 (beats placebo
   p=0.006) yet describes professional text poorly (agreement 0.63 fit, 0.43 held out) - the
   false-satisfy/lexical-shadow mode, caught only because agreement stays a separate gate. `both`
   (agreement 0.07) also beats its placebo.
3. **Gap and alignment are different axes, and gap mis-ranks.** The most teachable-looking candidate by
   gap (`cond_if`, +44pp) is the clearest anti-aligned one (0/12, p=0.033 against its own placebo band),
   and `pg_with_vill` (agreement 0.96) is anti-aligned too (0/9, p=0.004).

### 16.4 What is solid after this section

- spread/coverage: stratified 11.9% [5.2-25.0] vs unbiased 6.2% [2.7-13.8]; both low, difference inside
  the noise, so prompt selection biases coverage by a few points and no more;
- applicability on unbiased prompts: 68.8% of prompts, no rule fires; 31.2% of rollouts judged;
- blindness: 89/120 and 155/240 within-group pairs tie on reward;
- alignment direction of the certified set: **not established** (20% vs 67% across prompt sets);
- alignment of candidates against their own placebo band: one candidate beats placebo significantly
  (`ability_can`) while failing agreement, and two candidates are significantly anti-aligned;
- negative-gap rules replicate across prompt sets and are excluded from the reward term (G30).

Tooling added with this section: `scripts/screen_concordance.py` (candidate alignment + placebo +
Fisher), `--cue-specs/--prompts-per-cue` in `scripts/p1_reward_smoke.py`, `--breadth`,
`--exclude-rules`, the spread-undefined guards, exact binomial and Fisher tests, cluster CIs over
groups. Tests: 39 passed (added: spread guard x2, binomial null, concordance direction for both
constraint kinds, plus the earlier power/reader tests).

---

## 16.4. The powered re-test retracts the shortlist: three of five signals reversed (2026-11-28)

The shortlist corpus (`data/rules/shortlist_candidates.json`, 5 cues x 16 prompts x 6 rollouts = 78 groups,
446 hypotheses) multiplied the decisive pairs 2-10x. The result is a clean negative:

| candidate | pairs (was) | conc (was) | placebo conc (was) | Fisher p (was) | now |
|---|---|---|---|---|---|
| `pg_during_während` | 5 -> **51** | 0.80 -> 0.61 | 0.21 -> 0.49 | 0.012 -> **0.139** | no longer beats its placebo |
| `pg_government_gouf` | 5 -> **10** | 1.00 -> 0.20 | 0.35 -> 0.48 | 0.007 -> 0.108 | **reversed**: now discordant |
| `cond_if` | 12 -> **26** | 0.00 -> 0.38 | 0.30 -> 0.60 | 0.033 -> 0.038 | the **placebo** is now the aligned one |
| `pg_with_vill` | 9 -> **17** | 0.00 -> 0.71 | 0.49 -> 0.45 | 0.004 -> 0.048 | flipped from anti- to aligned: the only survivor, borderline |
| `than` | 6 -> 9 | 1.00 -> 0.67 | 0.22 -> 0.40 | 0.000 -> 0.168 | still thin (needs ~160 groups) |

So the partial-corpus screen produced four significant-looking effects and the powered screen kept none of
them at the same sign and strength. Two consequences:

1. **Never quote concordance from fewer than ~30 decisive pairs.** At 5-12 pairs the 95% interval is
   [0.3, 1.0], and placebo bands sit in 0.2-0.6, so every apparent win is inside the noise of the
   comparison it is judged against.
2. **Decisive pairs are capped by verdict diversity, not by prompt count.** The screen now reports
   `diversity_rate` (share of applicable groups whose verdicts differ across rollouts) and
   `groups_needed_for_decisive_floor`: `decisive ~= applicable_groups x diversity_rate x k(k-1)/2`.
   `pg_during_während` has divR 0.56 (18 groups suffice); `than` has 0.06 (160 groups); the exclusion
   candidates in the cue6 corpus have divR = 0.00 and produced **zero** decisive pairs at 161 groups.
   That is the real cost model for this axis: a rule that almost never distinguishes two rollouts cannot
   be alignment-tested at any reasonable budget.

On the full cue6 corpus (161 groups, 5.78 rollouts/group) exactly one candidate clears all axes:
`LB_MIN04_negation_not` - which is the rule with the **most negative** policy-vs-reference gap (-10pp
stratified, -40pp unbiased). The gap axis and the alignment axis disagree even about a certified rule,
which is the strongest form of the 16.3 point: single-axis screening is not a selection procedure.

---

## 16.5. Coverage by prompt targeting (positive), and the state of the Lean line (2026-11-28)

**Coverage is a curriculum variable.** The same eight rules give advantage coverage 6.2% on
card-independent prompts, 11.9% on stratified-per-rule prompts, and **18.0% [12.8-24.7]** when prompts are
selected so that a candidate cue is present (cue6: 161 prompts x 6 rollouts; shortlist set 17.9%
[11.0-27.9]). Nothing about the rules changed. This is a legitimate P4 lever - choose prompts the verifier
can judge - as long as the population number (6.2%) is reported next to the curriculum number, because the
latter is conditioning on applicability.

**The Lean line (branch `p6-lean-verified-grammar`, worktree `GramMetaRL_p6`) was re-checked against the
current card set and cannot ingest it.** Measured: `scripts/lean_pilot.py --cards
data/rules/mbg_cards_condv3.jsonl` dies with a pydantic `literal_error` on
`output_constraints.0.kind = 'source_conditional'`: the branch schema predates that kind. Its own numbers
(`artifacts/lean/summary.txt`, on the 87-card recompiled set) stand as a measurement:

- 87 cards / 97 constraints / alphabet 94: `ok` 63, `out_of_fragment` 29 (27 non-regex kinds, 2 use `(?!`),
  `dead` 4, `universal` 1;
- 59 cards carry a formalizable obligation; **9 cards are provably unsatisfiable** (`positives /\ not(exclusions) = empty`),
  which is an unavoidable penalty for every rollout that retrieves them;
- 596 pairwise card conflicts among 1,711 pairs tested;
- 12 automata claims emitted, **12/12 accepted by the Lean 4.15.0 kernel**, with minimal witnesses
  (`'s'` at depth 0, `'en'` at depth 1) - and getting there the kernel exposed 7 encoding bugs in the
  compiler, including one that made 24/59 look unsatisfiable when only 9 were.

Why this matters for the current bottleneck: the axes that sampling measures noisily are **decidable** in
that layer. `dead`/`universal` is exactly the vacuity that shows up as discrimination 5.3%; provable
emptiness of the region where two verdicts differ predicts `diversity_rate = 0` - the property that made
the exclusion candidates untestable at 161 groups - **without any GPU corpus**. Reconnecting the two lines
is a small, well-defined job: add `source_conditional` to the branch schema and lower it into the automata
layer (the obligation is `not(source_pattern) \/ target_form`, a regular language), then re-run
certificates on the 93-card condv3 set and gate on `dead`/`universal`/unsatisfiable there.
