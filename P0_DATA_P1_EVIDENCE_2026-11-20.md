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
