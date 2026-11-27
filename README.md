# GramMetaRL

Compile a descriptive grammar book into **executable grammar constraints**, measure how good each
constraint actually is against professional human translations, and use only the certified ones as a
reinforcement-learning reward for an `en → lb` (English → Luxembourgish) translation agent.

The project's contribution is not "we wrote rules". It is **a measurement contract for rules**: every
constraint that reaches the reward function arrives with a measured agreement rate, a measured
discrimination rate, a measured support count, and a reliability weight derived from them. Numbers that
cannot be measured are reported as undecidable and abstain instead of producing a verdict.

---

## 0. Status at a glance (measured, 2026-11-27)

| item | value | where it comes from |
|---|---|---|
| Gate table | **PASS=29 FAIL=3 WARN=10** | `scripts/smoke_test.py` |
| Unit tests | **35 passed** | `tests/` |
| Cards: 87 book-extracted + 6 corpus-mined | **93** | `data/rules/mbg_cards_condv3.jsonl` |
| Rules certified usable (3 axes + support floors, CC0 pool) | **8** (2 book + 6 mined) | `artifacts/audit/rule_quality_condv3_heldout.json` |
| Rules certified on the CC BY-NC pool (reported separately) | **2** (a *different* pair — see §5) | `artifacts/audit/rule_quality_condonly_luxalign.json` |
| Labeled pairs (certified rules only, 100% CC0) | **2,555** | `data/processed/labeled_condv3.jsonl` |
| SFT/RL dataset (train / valid / rule-level held-out) | **2,183 / 243 / 129** (1 of 8 rules held out) | `data/processed/gated_condv3/` |
| Honest false-penalty rate (professional translations penalized) | **22.3%** (weighted 23.9%) | `artifacts/audit/gated_condv3_techgov.json` |
| GRPO advantage coverage, stratified prompt set (the honest number) | **11.9%** of groups [CI 5.2–25.0] | `artifacts/p1/reward_smoke_gemma3_12b_strat.json` |
| Rules that actually separate policy from professionals (violation gap > 0.05) | **3 of 8**, one strongly (`cond_if` +44pp) | `artifacts/p1/signal_condv3_stratified.json` |
| Reward noise floor (reference translation penalized) | **14.3%** | `artifacts/p1/reward_smoke_gemma3_12b_strat.json` |
| GRPO advantage coverage on an **unbiased** prompt set (140 prompts, card-independent) | **0.0%** | `artifacts/p1/corpus_gemma3_12b_breadth.json` |
| Gradient alignment: reward ranks a rollout pair the way the reference does | **20%** of decisive pairs (2 vs 8) [CI 5.7–51.0] | `artifacts/p1/signal_condv3_stratified.json` |
| New rules certified by mining the policy's own errors (3 families tried) | **0** | `artifacts/p1/policy_gap_mining.json`, `artifacts/audit/screen_policygap_{fit,heldout}.json` |

Headline measured result of 2026-11-27 (two parts, both negative). (1) **Rule coverage is not the RL
bottleneck**: 2 → 8 certified rules raised label yield by a third and lowered the false-penalty rate, but
moved advantage coverage by at most +4.2pp on identical hypotheses (§8.1, evidence §14). (2) **The
certified set is not reward-ready**: on prompts sampled without respect to the rules the 8 rules produce
zero advantage, and where they do rank two rollouts differently the ranking usually opposes the
professional reference (20% concordance, gate G31 FAIL). They remain a label and evaluation asset.
See §8.2 and evidence §15.

The three FAIL gates: G21 and G24 are about *predicate* applicability (G21: 5/100 trigger conditions are
in executable form; G24: 26/87 cards carry an admitted predicate) and are bypassed in practice by
`source_conditional` constraints, which derive applicability from the **source** sentence instead of a
target-side predicate; G31 is the reward-alignment FAIL above. See §5, §8.2, §12.

### 中文快速上手

```bash
# trux 上的轻量环境（不装 verl，不装 torch）
PY=/raid/yewei/.venvs/gm-lite/bin/python
cd /raid/yewei/projects_yewei/GramMetaRL

$PY -m pytest tests/ -q                                   # 35 passed
$PY scripts/smoke_test.py --cards data/rules/mbg_cards_condv3.jsonl \
    --policy-endpoint http://127.0.0.1:11434/v1 --policy-model qwen2.5-coder:7b   # 门控表 PASS=29 FAIL=3 WARN=10
```

重新认证一条规则（留出协议，见 §10）不需要调用任何模型：判定缓存 `data/external/judge_cache.jsonl`
会直接复用。真正会烧时间的是 §4 的 LLM 卡片编译和 §8 的 rollout 采样。

---

## 1. The measurement contract (read this before trusting any number)

A constraint is allowed into the reward only if it passes **three independent axes plus support floors**:

1. **Agreement** — of the pairs where the constraint *decides*, how often does it agree with a
   professional human translation? Measured by `scripts/predicate_gated_audit.py`.
2. **Discrimination** — of the *minimal perturbations* applied to a satisfied check, how many flip to
   violated? A constraint that cannot fail contributes a **constant** reward and therefore a constant,
   useless (or worse: Goodhart-able) gradient. Measured by `scripts/verifier_discrimination.py`.
3. **Non-tautology** — does the applicability predicate restate the constraint itself? A rule whose
   trigger and requirement reduce to the same surface literals can only ever agree with itself.

Support floors: `>= 30` decided checks and `>= 20` attempted perturbations, or the rule is reported
`provisional` rather than certified. Without floors a single lucky pair satisfies the gates.

Three semantic rules that the code enforces, not just documents:

- **Undecidable means abstain.** `equivalence` and `optionality` constraints (they need a morphology
  engine) return no verdict and add no reward term. A silent false positive inside a reward is worse
  than a missing term.
- **Exclusions are one-sided evidence.** A `must_exclude` hit can penalize; the *absence* of a forbidden
  shape cannot certify correctness.
- **`violated` is a noisy negative.** Its weight is the measured agreement. It is never ground truth, and
  `satisfied` labels carry precision equal to that same agreement.

---

## 2. Machines, environments, model endpoints

| machine | role | path | notes |
|---|---|---|---|
| **trux** (`trux-tsubame.uni.lux`) | control plane, everything measured | repo `/raid/yewei/projects_yewei/GramMetaRL` | env `/raid/yewei/.venvs/gm-lite` (no torch, no verl) |
| **gpu7** (DGX Spark, `10.6.32.15`) | the only host that can reach Strata | repo `/home/yiqun/gm_scratch/GramMetaRL`, env `.venv-gm` | LLM compile + judge runs happen here |

| endpoint | URL | model | notes |
|---|---|---|---|
| Strata (gpu7 only) | `http://127.0.0.1:8081/v1` | `qwen3.8-flash-next-iq3_s` | **thinking model**: always send `max_tokens`; `content` comes back `null` when reasoning exhausts the budget. Key in `/home/yiqun/.local/state/strata/launch/iq3_s-256k.json`. Runs MoE experts from the disk tier: ~45 s per card, so batch 3–4 and `--timeout 850`. |
| Ollama (trux) | `http://127.0.0.1:11434/v1` | `gemma3:12b` (usable), `qwen2.5-coder:7b` (not usable) | ~70 s per call for `gemma3:12b`; see §8 |

**Network reality (do not re-discover this):** trux cannot reach gpu7 on 22 or 8081. gpu7 **can** reach
trux, so artifacts move one way only:

```bash
# run on gpu7
rsync -e "ssh -i ~/.ssh/id_ed25519_gm_transfer" -av <dir> \
  yewei@trux-tsubame.uni.lux:/raid/yewei/projects_yewei/GramMetaRL/artifacts/from_gpu7/
```

Code moves between machines via the git remote (`DobricLilujun/GramMetaRL`), never rsync.

---

## 3. Data inventory and the license policy

`data/external/pairs_en_lb.jsonl` — **48,436** sentence pairs, one license tag per row
(`MANIFEST.json` beside it, ingested by `grammetarl/data_sources/external_pairs.py`):

| pool tag | n | license | usage rule |
|---|---|---|---|
| `techgov_lu_en` (Tech-in-GOV "Meisproochegen Iwwersetzungskorpus", LU-EN) | **8,972** | **CC0 1.0** | publishable, and it is the default `--source-filter` of every measurement |
| `luxalign_v3_lb_en` (LuxAlign v3, used reversed `en→lb`) | 39,464 | CC BY-NC 4.0 | usable, but **always reported as a separate pool**; its own card warns pairs are only semantically similar |

Community `Tatoeba`-style pairs (42.8k rows, targets like `"Luxembourgish"`) are **deliberately not
ingested** — probe-only garbage. Per-row provenance lives in `data/external/MANIFEST.json`.

The grammar book itself yields only **14 sentence-grade pairs** out of 164 examples, which is why
rule labels come from compiled verifiers rather than from the book.

Rule: **never merge CC0 and CC BY-NC numbers into one table.** Support expansion on the second pool is
reported in its own artifact (`condonly_luxalign.json`), and the paper quotes both.

---

## 4. Card pipeline (grammar book → executable constraints)

All commands run from the repo root. `grammetarl/training/*` need `PYTHONPATH=.` unless the package is
installed. `main.py` is subcommand-based (`build-mbg`, `retrieve`, `verify`, `init-workspace`,
`migrate-workspace`); the global flags `--workspace-root` / `--namespace` go **before** the subcommand. The
grammar PDF lives at `assets/lb/luxembourgish_grammar.pdf` (not tracked).

```bash
PY=/raid/yewei/.venvs/gm-lite/bin/python

# 1. OCR the book, page-level JSON
$PY scripts/extract_pdf_page_ocr.py --pdf assets/lb/luxembourgish_grammar.pdf \
    --start-page 1 --end-page 40 --dpi 200

# 2. Extract MBG cards (run on gpu7, the only host that can reach Strata)
$PY main.py build-mbg --language lb --source-id LLB_GRAMMAR_2020 \
    --pdf assets/lb/luxembourgish_grammar.pdf \
    --ocr-pages-dir artifacts/ocr/<run> \
    --llm-endpoint http://127.0.0.1:8081/v1 --llm-model qwen3.8-flash-next-iq3_s \
    --llm-timeout 850 --llm-max-tokens 4096
# -> data/rules/mbg_cards.jsonl  (87 cards)
# TRAP: build-mbg defaults --llm-endpoint to http://10.6.32.16:8000/v1 and --llm-model to
# nvidia/Qwen3.6-35B-A3B-NVFP4, i.e. a server on another machine. Always override both.

# 2b. single-sentence inspection, no model needed
$PY main.py retrieve --cards data/rules/mbg_cards_condonly.jsonl \
    --sentence "Our children are taller than their parents." --top-k 3
$PY main.py verify --cards data/rules/mbg_cards_condonly.jsonl \
    --sentence "Our children are taller than their parents." \
    --translation "Eis Kanner si méi grouss wéi hir Elteren." \
    --rule-ids LB_9789dfb15623,LB_946a24dd25d6
# -> both certified rules decide 'satisfied', with the source-derived applicability trace in `messages`

# 3. Compile applicability predicates, then repair/recompile constraints.
#    Admission is strict and local: a card is rejected if its own example violates it.
$PY scripts/compile_triggers.py     --cards data/rules/mbg_cards.jsonl --out data/rules/mbg_cards_app.jsonl \
    --endpoint http://127.0.0.1:8081/v1 --model qwen3.8-flash-next-iq3_s --batch-size 4 --timeout 850
$PY scripts/repair_constraints.py   --cards data/rules/mbg_cards_app.jsonl --audit <audit.json> --out data/rules/mbg_cards_repaired.jsonl
$PY scripts/recompile_constraints.py --cards data/rules/mbg_cards_repaired.jsonl --audit <audit.json> \
    --out data/rules/mbg_cards_recompiled.jsonl --endpoint ... --model ... --batch-size 4 --timeout 850
$PY scripts/triage_constraints.py   --cards data/rules/mbg_cards_recompiled.jsonl --audit <audit.json> \
    --min-support 10 --max-violation 0.3 --out data/rules/mbg_cards_triaged.jsonl
```

Measured acceptance, in order (these are the honest yield numbers, not a bug report):

| step | result |
|---|---|
| trigger compilation | 21/87 predicates accepted on first admission; 12 `llm_check` give-ups after the prompt was strengthened (was 46) |
| constraint recompilation | 48/87 constraint *kinds* changed (27 → `must_include_any_regex`, 8 → `optionality`, 3 → `equivalence`, 19 → `llm_check`) |
| predicate precision, bulk LLM-compiled | **0.041** micro — useless |
| predicate precision, hand-authored | **0.347** — 8.5× better, still weak |
| `source_conditional` condition precision | **0.50–0.57** — the only applicability route that works |

Tracked card sets (`data/rules/`, all 87 rows unless noted): `mbg_cards{,_app,_repaired,_recompiled,
_recompiled_manual,_triaged}.jsonl`; the precise 50-card subsets `mbg_cards_precise{,_app,_manual}.jsonl`;
`mbg_cards_conditional.jsonl` (conditions added) and **`mbg_cards_condonly.jsonl`** (conditions *replace*
the predicate-bearing constraints — this is the set every current number is measured on; 5 cards carry
`source_conditional` constraints, 6 constraints total, one of them `negate: true`).

Adding a source-conditional constraint is gated, never free:

```bash
$PY scripts/apply_source_conditionals.py --cards data/rules/mbg_cards_recompiled_manual.jsonl \
    --spec data/rules/source_conditionals.json --replace --out data/rules/mbg_cards_condonly.jsonl
```

Admission requires that both regexes compile, the **source** pattern fires on the card's own examples,
and the **target** values hold on those examples (inverted for `negate`). It has rejected 2 of my own
attempts: `\b[a-z]{3,}er\b` matched `after/under/number`, and a perfect-tense pattern missed the
irregular `built`.

---

## 5. The measurement stack

Every script below is zero-LLM by default: it replays the judge cache (`data/external/judge_cache.jsonl`,
8,868 lines, keyed by `(pair, card)`, first-wins). So re-measuring after any card change is free.

| script | what it answers | key flags |
|---|---|---|
| `scripts/label_with_judge.py` | "is this rule *applicable* to this pair?" (model-judged, cached) | `--judge-all --card-ids` (retrieval mode deliberately skips predicate-bearing cards, which is why their precision was unmeasurable before) |
| `scripts/audit_constraint_precision.py` | violation rate + verified-label count per card set | `--emit-pairs` makes one code path the source of truth for both precision **and** yield |
| `scripts/validate_triggers.py` | predicate/applicability precision & recall against the judge cache | `--split-rows` for fit/held-out |
| `scripts/predicate_gated_audit.py` | constraint agreement under a consistent denominator | `--rows 4000` (fit), `--skip 4000` (measure) |
| `scripts/build_reliability_table.py` | weight = `clamp(2p-1, 0, 1)` on support-shrunk agreement | unmeasured or coin-flip rules get weight **0** and cannot move the reward |
| `scripts/verifier_discrimination.py` | does the check ever flip? (token-level minimal perturbations) | `--perturbations 3` |
| `scripts/rule_quality_matrix.py` | the three axes + `usable` / `provisional` verdicts | `--skip`, `--min-support 30`, `--min-tested 20`, `--reliability <fit table>` |
| `scripts/violation_decomposition.py` | **why** a rule disagrees with professionals | `--bucket 'NAME=REGEX'` (repeatable, first match wins) |
| `scripts/match_context_audit.py` | what token actually *satisfies* a check (false-satisfy audit) | `--rollouts` from a P1 smoke log |
| `scripts/gated_pipeline_measurement.py` | end-to-end: which rules may label pairs, and the honest yield | **`--quality`** = admit only matrix-certified rules |

Current verdicts (CC0 pool, `artifacts/audit/rule_quality_conditional.json`; `n` = decided checks that
agree, `pert` = perturbations attempted):

```
usable        2   LB_9789dfb15623  comparative      agree 0.755 / n=310   / disc 75.3% (pert 300)
                  LB_946a24dd25d6  possessives      agree 0.697 / n=1,973 / disc 91.4% (pert 162)
provisional   2   LB_363a21684c8f  een-prohibition  agree 0.939 / n=29    / disc 100%  (support < 30)
                  LB_55471d5dc4b4  -                agree 0.600 / n=1     / disc 100%  (the reason floors exist)
vacuous       3   LB_d121073f582e  disc  5.3% (n=2,041 - a constant reward), LB_374581869ba8 8.4%,
                  LB_e51077fe7961  disc 17.6%
self-restating 1  LB_27c64c1f7e4b  polite address: predicate == its own constraint (n=1,993, disc 70.2%);
                                     judged applicable in 0/292 pairs because English marks no formality
below floor    1  LB_2ebda66afe75  kinship borrowing: agree 0.595 on CC0 (n=33) - certifies on the other pool
```

The same cards measured on the **CC BY-NC pool** (`artifacts/audit/rule_quality_condonly_luxalign.json`
rebuild it; `rule_quality_condonly_luxalign.json` is the matrix view) certify a *different* pair:
`LB_363a21684c8f` (agree 0.945 / n=361 / disc 100%) and `LB_2ebda66afe75` (0.762 / n=38 / disc 100%), while
the comparative rule **drops to 0.626 and fails there**. Certification is pool-dependent, which is exactly
why the two pools are never merged into one number.

Discrimination is **sample-sensitive**: the `-en` plural rule measured 5.3% on the full pool and 31.5% on
the held-out slice with identical code, and the possessive rule read 92.6% in one sample and 91.4% in
another. Single percentages over-claim; no confidence intervals yet.

### 5.1 Corpus-mined `source_conditional` rules (2026-11-27)

Hand-authored conditionals mostly failed: past perfect 0.30, 2sg clitic 0.23, `eng`+cardinal 0.08 held out
(evidence §14.1). The property that predicts survival is **how obligatory the target form is**, not regex
care. So new rules are mined from the corpus and screened, not invented:

```bash
# candidates (22) live in data/rules/screened_conditionals.json; screen on the FIT half only
$PY scripts/screen_source_conditionals.py --candidates data/rules/screened_conditionals.json \
    --rows 4000 --out artifacts/audit/screen_conditionals_fit.json
$PY scripts/screen_source_conditionals.py --candidates data/rules/screened_conditionals.json \
    --skip 4000 --out artifacts/audit/screen_conditionals_heldout.json
# a candidate ships only if BOTH halves clear agree>=0.65, disc>=0.5, n>=30, tested>=20

# certified candidates -> real MBG cards with honest provenance
$PY scripts/make_mined_cards.py            # -> data/rules/mbg_cards_mined.jsonl (6 cards)
$PY scripts/apply_source_conditionals.py --cards data/rules/mbg_cards_condv2.jsonl \
    --specs data/rules/source_conditionals_v2.json --out data/rules/mbg_cards_condv3.jsonl

# then the normal held-out certification on the merged set
$PY scripts/predicate_gated_audit.py --cards data/rules/mbg_cards_condv3.jsonl --rows 4000 \
    --emit-reliability data/rules/reliability_condv3_fit.json --out artifacts/audit/condv3_fit.json
$PY scripts/rule_quality_matrix.py --cards data/rules/mbg_cards_condv3.jsonl --skip 4000 \
    --reliability data/rules/reliability_condv3_fit.json \
    --out artifacts/audit/rule_quality_condv3_heldout.json
```

Mined cards carry `source.source_id = CORPUS_MINED_TECHGOV`, `rule_status = "variable"` (there is no
`statistical` in the schema), an excerpt that states "no book evidence", and examples drawn from the **fit
half** so the card self-test does not leak the held-out certification. Two protocol traps found while
building this: `verifier_discrimination.corrupt_all` defaults to `min_instances=2` and silently returns
nothing for single-instance cues (screening passes `min_instances=1`, otherwise `tested` collapses and a
rule is called "too thin" because of the probe); and a loose **source** condition is as damaging as a loose
predicate — `-est` fired on `interest`/`request`/`suggest`, and `as … as` matched across unrelated `as`
tokens.

---

## 6. Labels and the training dataset

```bash
# labels: only matrix-certified rules, reliability-weighted, applicability required
$PY scripts/gated_pipeline_measurement.py \
    --cards data/rules/mbg_cards_condv3.jsonl \
    --reliability data/rules/reliability_condv3_fit.json \
    --quality artifacts/audit/rule_quality_condv3_heldout.json --min-agreement 0.65 \
    --rows 0 --source-filter techgov_lu_en \
    --labeled-out data/processed/labeled_condv3.jsonl \
    --out artifacts/audit/gated_condv3_techgov.json
# -> pairs_scored 3,270 / labeled_pairs 2,555 / decided_checks 3,907 / false_penalty 22.3% (weighted 23.9%)

# dataset: pair-level splits + rule-level held-out
PYTHONPATH=. $PY grammetarl/training/build_dataset.py \
    --from-cards data/rules/mbg_cards_condv3.jsonl \
    --external-tagged data/processed/labeled_condv3.jsonl \
    --out-dir gated_condv3 --no-lexicon
# -> train 2,183 / valid 243 / valid_rules_heldout 129  (1 of 8 rules held out, overlap 0, 100% CC0-1.0)
```

**Rule-level held-out became measurable only after the rule set reached 8.** At 2 certified rules it was
undefined (`valid_rules_heldout.jsonl` was empty, which §13 of the evidence doc reports rather than
hides); now one rule (`LB_MIN03_every_each`) is held out entirely and 129 pairs score cross-rule
generalization. `data/processed/labeled_condonly.jsonl` (1,585 rows, 2 rules) and
`labeled_conditional.jsonl` (3,269 rows, pre-discrimination-gate, ~2,000 checks from a rule that cannot
bite) are both superseded by `labeled_condv3.jsonl`.

Reward-side sanity check (no policy involved):

```bash
$PY scripts/reward_sanity_check.py --cards data/rules/mbg_cards_condonly.jsonl \
    --reliability data/rules/reliability_condonly.json --require-applicability
```

---

## 7. Reward: weighting and the enforced weight band

- `grammetarl/rewards/reliability.py` — `ReliabilityTable.load(...)`, `.weights(rule_ids)`.
- `grammetarl/rewards/grammar_reward.py::grammar_reward_weighted(verdicts, weights)` — weighted mean over
  the checks that *earned* a weight. All weights equal ⇒ identical to the unweighted mean; weight 0 ⇒
  the check disappears.
- `grammetarl/configs/reward/grammar_reward.yaml` — **grammar 0.45 / mt 0.30 / tool 0.10 / trace 0.10 /
  efficiency 0.05**, and gate **G29** enforces the two measured constraints behind it:
  `w_grammar > w_mt` (the contribution must not be outvotable by a reference metric) **and**
  `w_mt + w_tool > 0` (a grammar-only reward is farmable — see §12).

Measured effect of weighting on professional translations: false penalties drop **52.9% → 25.1%**
(recompiled cards) and **33.3% → 6.1%** (triaged cards).

---

## 8. P1 reward smoke: is the reward a usable RL signal?

`scripts/p1_reward_smoke.py` runs the reward half of GRPO on real rollouts. It does **not** train, and
`verl` is deliberately not installed in this environment (it would conflict with the extraction stack).

```bash
# the honest protocol: stratify prompt selection so every certified rule gets its own groups
$PY scripts/p1_reward_smoke.py --prompts-per-rule 6 --rollouts 3 --temperature 0.9 \
    --cards data/rules/mbg_cards_condv3.jsonl \
    --reliability data/rules/reliability_condv3_fit.json \
    --quality artifacts/audit/rule_quality_condv3_heldout.json \
    --policy-endpoint http://127.0.0.1:11434/v1 --policy-model gemma3:12b \
    --timeout 400 --max-tokens 220 \
    --out artifacts/p1/reward_smoke_gemma3_12b_strat.json \
    --rollout-log artifacts/p1/rollouts_gemma3_12b_strat.jsonl   # log is not tracked, JSON is
# rollout 0 of every group is greedy (temperature 0), the rest at --temperature; recorded in the artifact
```

Measured per policy:

| policy | Luxembourgish? | `nonzero_spread_rate` | reading |
|---|---|---|---|
| `gemma3:12b`, 8 rules, **stratified** | real, imperfect | **0.119** [CI 0.052–0.25] | the defensible number: 42 groups, 6 per rule; `reference_penalized_rate` 14.3%; `policy_minus_reference` −0.092 |
| `gemma3:12b`, 2 rules, first-decided prompts | real, imperfect | 0.292 | superseded as a headline: that prompt set was selected by the 2-rule set, so it flattered it |
| `qwen2.5-coder:7b` | gibberish | 0.0 | every rollout scores −1.0 → zero advantage |
| `qwen2.5:14b` | gibberish | 0.0 | same |
| `gemma3:4b-it-fp16` | weak/gibberish | 0.0 | same |
| `qwen3:32b`, `gpt-oss:20b` | — | — | never answered within 400 s on trux |

**Capability is the first-order effect.** A policy with no target-language competence cannot be trained by
this reward at all, because a group of identical rewards has zero advantage. That is the empirical
argument for the SFT warmup step in `paper/research_plans/P1_card_to_verifier_reward.md`.

### 8.1 Coverage is not the bottleneck (measured 2026-11-27, evidence §14)

Going 2 → 8 certified rules raised label yield (+61% labelled pairs) and *lowered* the false-penalty rate
(28.9% → 22.3%), but the RL readout went the other way: 29.2% → 16.7% spread rate. That comparison was
confounded: `--prompts N` selects "the first pairs whose reference decides a rule", so the rule set
changes the prompt set. `scripts/p1_signal_by_rule.py` re-scores the **stored hypotheses** offline (zero
model calls) and reproduced the online 29.2% exactly, then held hypotheses fixed:

| hypothesis set | 2 rules | 8 rules |
|---|---|---|
| prompts selected by the 2-rule set | 29.2% | 29.2% |
| prompts selected by the 8-rule set | 12.5% | 16.7% |

Per-rule readout on the stratified run (18 policy checks per rule) is what actually tells you which rules
belong in a reward — policy violation rate vs the professional reference's:

| rule | policy viol | ref viol | gap | creates group spread |
|---|---|---|---|---|
| `LB_MIN02_cond_if` | 61.1% | 16.7% | **+44.4pp** | 1/42 |
| `LB_MIN03_every_each` | 38.9% | 33.3% | +5.6 | 1/42 |
| `LB_MIN01_comp_that` | 5.6% | 0.0% | +5.6 | 1/42 |
| `LB_MIN05_than` | 16.7% | 16.7% | 0.0 | 0/42 |
| `LB_9789dfb15623` comparative | 0.0% | 0.0% | 0.0 | 0/42 |
| `LB_946a24dd25d6` possessive | 7.4% | 11.1% | −3.7 | 1/42 |
| `LB_MIN06_without` | 27.8% | 33.3% | −5.6 | 1/42 |
| `LB_MIN04_negation_not` | 0.0% | 10.0% | **−10.0** | 0/42 |

Three readings, all of them measurements:

1. A rule can be **saturated**: `negation_not` is the highest-agreement mined rule (0.89 held-out), fires
   on 30 rollouts, and the 12B satisfies it 100% of the time → zero advantage.
2. A certified rule can have a **negative gap**: it penalises professional translations more often than
   the policy, so in RL it pulls the policy away from professional usage. Gate **G30** names these.
3. Rollouts are not degenerate (mean distinct hypotheses per group 2.93 of 3; greedy-vs-sampled text
   similarity 0.812), so the missing signal is in **what the rules look at**, not in sampling.

```bash
# reproduce the table above from a rollout log, offline, no model calls
$PY scripts/p1_signal_by_rule.py --rollout-log artifacts/p1/rollouts_gemma3_12b_strat.jsonl \
    --out artifacts/p1/signal_condv3_stratified.json
# compare rule subsets on identical hypotheses (the paired test that removed the confound):
$PY scripts/p1_signal_by_rule.py --rollout-log artifacts/p1/rollouts_gemma3_12b.jsonl \
    --rule-ids LB_946a24dd25d6,LB_9789dfb15623 --out /tmp/sig.json
```

### 8.2 Reward readiness is alignment, not coverage (measured 2026-11-27, evidence §15)

Two measurements say the current certified set is not a usable training reward, and both are cheap to
re-run offline.

**An unbiased prompt set gives zero advantage.** `scripts/p1_reward_smoke.py --breadth 140` samples
prompts by stride over the whole pool, without regard to whether any rule fires:

    140 prompts x 1 greedy rollout (gemma3:12b) -> nonzero_spread_rate = 0.0

The 11.9% figure is conditional on per-rule prompt selection. Plan against the unbiased number.

**Where the reward does rank two rollouts, it usually ranks them against the reference.** For every
within-group rollout pair, `p1_signal_by_rule.py` compares the sign of the grammar-reward difference with
the sign of the token-F1-to-reference difference:

    concordant 2, discordant 8 -> 20.0% [95% CI 5.7, 51.0]
    ties: reward-only 89, similarity-only 0, both 27

n = 10 decisive pairs, so discordance is not statistically established — what is established is the shape:
the rules are blind on 89 of 120 pairs, and in the remainder they more often oppose than support the
professional ranking. Gate **G31** requires concordance ≥ 0.5 with ≥ 10 decisive pairs and therefore
**FAILs**. Consequence: `w_grammar` (gate G29) is not a safety margin, it is the reason a blind-to-opposing
rule term is survivable at all.

**Dropping the anti-professional rules lowers coverage**, which proves spread was never the objective:

| rule set (identical 126 hypotheses) | nonzero spread |
|---|---|
| 8 certified rules | 5/42 = 11.9% [5.2, 25.0] |
| 5 rules, negative-gap rules dropped | 3/42 = 7.1% [2.5, 19.0] |

Excluding them is a **correctness** lever (the gradient stops pointing away from professionals), not a
signal-density lever.

```bash
# unbiased prompt set (rollout corpus for mining and for the honest coverage number)
$PY scripts/p1_reward_smoke.py --model gemma3:12b --breadth 140 --rollouts 1 --temp 0 \
    --cards data/rules/mbg_cards_condv3.jsonl --reliability data/rules/reliability_condv3_fit.json \
    --log-out artifacts/p1/rollouts_gemma3_12b_breadth.jsonl \
    --out artifacts/p1/reward_smoke_gemma3_12b_breadth.json
# alignment + per-rule alignment + spread for any rule subset, offline
$PY scripts/p1_signal_by_rule.py --rollout-log artifacts/p1/rollouts_gemma3_12b_strat.jsonl \
    --rule-ids LB_MIN01_comp_that,LB_MIN02_cond_if,LB_MIN03_every_each,LB_9789dfb15623,LB_MIN05_than \
    --out artifacts/p1/signal_condv3_strat_pos5.json
```

### 8.3 Mining the policy's own errors: three families, zero new certified rules (evidence §15)

`scripts/mine_policy_gaps.py` mines candidates in two stages — pool statistics on the **fit half only**
(support ≥ 30, professional agreement ≥ 0.5, cue specificity `lift = p(form|cue)/p(form|cue absent) ≥ 2`),
then ranking by the policy-vs-reference gap measured on stored rollouts — and emits specs in exactly the
`data/rules/screened_conditionals.json` shape so the existing two-half screener consumes them.

| family | stage A | teachable after stage B | certified (two halves) | why it failed |
|---|---|---|---|---|
| single-token cue, required form | 155 | 4 | **0** | cues professionals obey are already obeyed by the 12B (saturated) |
| ordered pairs `t1 .* t2` | 0 | 0 | **0** | arithmetic: a multi-element constraint cannot clear an unconditional agreement floor of 0.65 |
| exclusion (`negate: true`) | 1,287 | 739 | **not shipped** | survivors are near-synonyms/frequency variants (`during→während`, `so→just`): reference imitation wearing a grammar label |

Guard added by a real bug: the first version stored hypotheses as `{text: token_set}`, so every policy
rate printed 0.00; the script now hard-stops when all policy rates are zero. The lift axis is load-bearing
too — an earlier corpus-only pass promoted `has → den` (a dative article with lift ≈ 1).

**The next mining criterion is concordance, not gap**: score candidates by whether their verdict
differences rank rollouts like the reference does (free offline once rollouts are stored).

---

## 9. Gates

```bash
$PY scripts/smoke_test.py --cards data/rules/mbg_cards_condv3.jsonl \
    --policy-endpoint http://127.0.0.1:11434/v1 --policy-model qwen2.5-coder:7b
```

Current: **PASS=29 FAIL=3 WARN=10.** `P0 gate is GREEN only when every FAIL is cleared (WARN = measured
risk, report it).` Pass `--policy-endpoint`/`--policy-model`: without them the policy-dependent gates
change state and the tally reads PASS=27 FAIL=4 for the same card set.

Gate families:

- G1–G19 schema, retrieval, env, tools, dataset integrity, chrF discriminativeness
- G20/G23 split hygiene (no pair overlap; rule-level held-out excludes those rules)
- G21 trigger executability — **FAIL** (5/100 in regex form)
- G24 admitted applicability predicates — **FAIL** (26/87)
- G25 no prose/identifier in a regex slot
- G26 reward is reliability-weighted, not oracle-assumed
- G27 ≥1 rule with agreement ≥ 0.65 **and** discrimination ≥ 0.5 **and** not its own trigger
- G28 P1 reward signal measured on rollouts; **G28b WARN** advantage coverage < 50% (11.9% stratified)
- G29 reward weights inside the measured band
- G30 ≥1 reward rule with positive policy-vs-reference gap; **G30b WARN** advantage coverage < 30%
  (11.9% stratified, 0.0% unbiased); **G31 FAIL** reward-vs-reference gradient concordance < 0.5
- G30 ≥1 reward rule has a positive policy-vs-reference violation gap (**PASS**: `cond_if` +44pp);
  **G30b WARN** advantage coverage on the stratified prompt set, with rollout diversity printed next to it

Note: the trux env has no `ruff`, so lint runs on a machine that has it. Tests, the gate table and the
re-measurement scripts all run on trux.

---

## 10. Reproducing a certification from scratch (held-out protocol)

```bash
PY=/raid/yewei/.venvs/gm-lite/bin/python
CARDS=data/rules/mbg_cards_condonly.jsonl

# fit half -> reliability weights
$PY scripts/predicate_gated_audit.py  --cards $CARDS --rows 4000 --source-filter techgov_lu_en \
    --sample 0 --out artifacts/audit/condonly_fit.json
$PY scripts/build_reliability_table.py --audit artifacts/audit/condonly_fit.json \
    --out data/rules/reliability_condonly_fit.json

# measure on the other half, with the FIT weights only
$PY scripts/rule_quality_matrix.py --cards $CARDS \
    --reliability data/rules/reliability_condonly_fit.json --skip 4000 \
    --samples-per-rule 100 --perturbations 3 \
    --out artifacts/audit/rule_quality_conditional_heldout.json
```

Why it is written this way: an earlier `--skip` run silently re-printed the fit-half numbers from the
reliability table, which made a held-out run look better than it was. The matrix now **measures agreement
in-run**; the table is only a weight source. Compare fit vs held-out before calling a rule certified —
`LB_946a24dd25d6` moved 0.64 → 0.74 and the `-en` rule's discrimination moved 5.3% → 31.5%.

---

## 11. What is in git, and what is not

Tracked: code, `tests/`, card sets `data/rules/mbg_cards*.jsonl` (all generations, including
`mbg_cards_mined.jsonl` and the merged `mbg_cards_condv3.jsonl`), rule specs
(`data/rules/{source_conditionals*,screened_conditionals,policy_gap_candidates,manual_triggers,reliability*}.json`),
CC0 labels `data/processed/labeled_*.jsonl` (verified: 100% of rows in every one of them are CC0-1.0), and
the evidence JSON: `artifacts/audit/{precision_*,rule_quality_*,condonly*,condv*_fit,screen_*,
gated_cond*,violations_*,trigger_quality*}.json` plus `artifacts/p1/*.json` (every smoke, alignment and
mining report, including the negative ones from evidence §15).

Not tracked: raw OCR output, rollout logs (`*.jsonl` under `artifacts/p1/`), the built dataset splits
(`data/processed/gated_condv3/{train,valid,valid_rules_heldout}.jsonl` = 2,183 / 243 / 129 rows — rebuild
with the §6 command, they are derived from tracked labels + tracked cards), model caches, the Overleaf
clone in `paper/` (it has its own git remote; only push it when explicitly authorized), and `data/external/`
(48k rows with per-row licenses — regenerate with `scripts/fetch_external_pairs.py`, provenance in
`MANIFEST.json`). Because rollout logs are not tracked, any measurement that re-scores stored hypotheses
(§8.1, §8.2, alignment) must be re-run after regenerating rollouts with §8; the summaries that those runs
produced are tracked, so every number quoted in this file and in the evidence document is checkable.

Not tracked: raw OCR output, rollout logs (`*.jsonl` under `artifacts/p1/`), model caches, the Overleaf
clone in `paper/` (it has its own git remote; only push it when explicitly authorized), and `data/external/`
(48k rows with per-row licenses — regenerate with `scripts/fetch_external_pairs.py`, provenance in
`MANIFEST.json`).

Deeper documents:

| file | content |
|---|---|
| `P0_DATA_P1_EVIDENCE_2026-11-20.md` | the full session record, §1–15, including every wrong claim and its correction (§10 is explicitly annotated "superseded by §11"; §14.4 supersedes the 29.2% headline; §15 supersedes "mine the policy's errors and the reward problem is solved") |
| `SMOKE_P0_BASELINE_2026-11-19.md` | the original P0 baseline gate run |
| `STATUS_REPORT.md` | earlier milestone status |
| `PRD_VERL_GRAMMAR_AGENT.md` | product spec for the verl agent side |
| `paper/research_plans/P1…P5*.md` | P1 card-to-verifier reward (main contribution), P2 turn-level credit, P3 rules-as-code, P4 curriculum/self-play, P5 benchmark transfer |

---

## 12. Known limitations and measured failure modes

1. **A rule that cannot bite is not a free win.** `LB_d121073f582e` (`^.*en$`-style plural check,
   discrimination 5.3%) supplied 2,041 of 3,981 checks and produced a beautiful **0.39% false-penalty
   rate**. With discrimination measured, the honest number for that card set is 14.9%, and with the
   quality gate on it is 28.9% over rules that actually constrain output.
2. **Possessor number is undecidable for 3rd-person plural.** LLB `their` admits both `är-` and `hir-`
   (`hirer Zäit`). Encoding the stricter claim manufactured false violations: 52.2% of the possessive
   rule's violations contain a possessive article of the *other* number. Fix: split into a decidable
   constraint (`our → eis-`) and a weakened one whose description states that number is **not** claimed.
   Held-out agreement 0.64 → 0.74 at n=1,180.
3. **False-satisfy.** A gibberish rollout earned **+1.0** because `\bmeng(en)?\b` matched the verb
   *mengen* ("to mean"), not a possessive article. `scripts/match_context_audit.py` quantifies why the
   obvious fix fails: of 1,603 professional possessive matches, 79.9% are followed by an uppercase token
   but **19.2% by a lowercase one**, because an adjective sits between possessive and noun
   (`eis elektronesch`, `sengem éischte`, `hire privaten`). Requiring a following capitalized noun would
   manufacture 19.2% false violations. Not fixable at surface level — this is the P3 morphology/POS
   argument, and the reason the MT term must stay in the composite reward.
4. **Surface patterns ≠ noun classes.** Two form-list defects found by the probe: `\bdäi\w*\b` matched
   `däischtere`, `\bsen\w*\b` matched `sent` ("sand"). Explicit paradigms
   (`\bmäi(n|en|em|er)?\b`) raised agreement 0.54 → 0.61 at n=1,968. A loose **condition** is just as
   damaging as a loose predicate: `\b\w+er\b` matched `after/under/number`, then `more [a-z]+` matched
   `more than` / `more or less` and collapsed comparative agreement 0.96 → 0.71.
5. **The perturbation probe had two bugs of its own** (found by running it): `_break_word` kept the stem
   while patterns end in `\w*` (so `hiren → hirxx5` still matched: 126/126 no-ops), and per-occurrence
   rules need **all** instances broken (405 of 1,968 possessive pairs carry 2+ possessors → `corrupt_all`).
   After fixing, the possessive rule's discrimination is 91.4% (`92.6%` in a different perturbation sample)
   — it was never vacuous, it simply disagrees with professional translations ~30% of the time.
6. **Retrieval is weak, and it is not on the current measurement path.** For
   `"Our children are taller than their parents."`, `main.py retrieve --top-k 3` returns the possessive
   rule plus two irrelevant phonotactics cards, all tied at score 0.336 — the comparative rule is not in
   the top 3. Measurement iterates named `--card-ids`, so nothing measured today depends on retrieval,
   but the agent's retrieve-tool path in a P1 rollout will hit this wall.
7. **Rule-level held-out was undefined at n=2 rules and is now defined.** With 8 certified rules,
   `build_dataset.py` holds out one whole rule (`LB_MIN03_every_each`, 129 pairs) — cross-rule
   generalization is measurable for the first time. At n=2 it silently produced an empty file, which is
   reported in evidence §13 rather than hidden.
8. **Judge cache is first-wins.** Appending a re-ask cannot overwrite an earlier judgment; delete the
   keyed line if a judgment must change.
9. **Interpretation attached to a regex bucket produces false findings.** Luxembourgish capitalizes every
   noun, so "article + capitalized token" is *not* a proper-name test (it matched
   `den Attentater` = the attackers). `violation_decomposition.py` therefore takes buckets as
   `NAME=REGEX` and adds no meaning.
10. **Coverage is not RL signal.** 2 → 8 certified rules: labels +61%, false-penalty −6.6pp, advantage
    coverage at most +4.2pp on identical hypotheses (§8.1). Saturated cues are good labels and useless
    advantages.
11. **A certified rule can be anti-professional in a reward.** `LB_MIN04_negation_not` (held-out agreement
    0.89) has a **−10pp** policy-vs-reference violation gap: professionals get penalised more than the
    policy. In GRPO that drags the policy away from professional usage. Gate G30 names these rules; the
    fix is to exclude them from the reward term while keeping them for labels/eval.
12. **Smoke comparisons are confounded by prompt selection.** `--prompts N` takes the first pairs whose
    reference decides a rule, so a new rule set changes the prompt set and a "regression" can be pure
    selection (§8.1). Use `--prompts-per-rule` and `scripts/p1_signal_by_rule.py` (paired, offline).
13. **Rates at n=24 need intervals.** 7/24 vs 4/24 is not a difference; both smoke scripts now emit
    `nonzero_spread_rate_ci95` (Wilson). At 42 groups the stratified rate is 0.119 [0.052, 0.25].
14. **Capitalization-dependent constraints are untestable in this harness.** `normalize_text`
    (`grammetarl/validator.py:82`) NFKC-normalises **and casefolds** before any pattern runs, so a rule
    that depends on a capital noun cannot be admitted; the admission test rejects it rather than letting a
    green number hide it.
15. **Orthographic variation is not modelled (scope note, not measured).** Constraints match the corpus
    spelling after normalisation; a policy that uses a different accepted spelling of the same word is
    scored as violating. No agreement number in this repo is spelling-corrected, and no number here quantifies
    how much of the disagreement that is — it is a known unmeasured source of noise, sitting next to the
    false-satisfy mode in item 4.
16. **Discrimination is 1.00 for every cue-type rule, so it certifies nothing there.** Screening reported
    `discrimination = 1.00` for all 22 mined candidates, including `both` (agreement 0.06) and `as_as`
    (0.08): corrupting a single-token target cue flips the check by construction. For
    `source_conditional` cue rules the certification therefore rests on agreement + non-tautology + support
    floors only; the missing axis is **applicability sensitivity** (does the constraint abstain when the
    *source* cue is removed?), not a target-flip test. Evidence §14.6.
17. **Every advantage-coverage number is conditional on prompt selection, and the unbiased number is 0.**
    On 140 prompts sampled by stride over the pool (card-independent, `--breadth`), the 8 certified rules
    produce `nonzero_spread_rate = 0.0`; on the per-rule-stratified set they produce 11.9%. Neither number
    is "the" coverage. Evidence §15.1.
18. **The reward gradient is not aligned with professional usage (measured, small n).** Of the within-group
    rollout pairs where both the reward and the lexical similarity to the reference differ, 2 are concordant
    and 8 discordant (20.0% [CI 5.7–51.0]); 89 of 120 pairs tie on reward. So the certified cue rules are
    mostly blind and, when they fire, more often oppose than support the professional ranking. Gate **G31**
    FAILs by design. This does not prove the rules are wrong — n = 10 decisive pairs — it proves the axis was
    never measured and the current rule set cannot carry a training gradient. Evidence §15.7.
19. **Surface-cue mining is exhausted; the certification regime also excludes a whole constraint family.**
    Three mined families produced zero new certified rules (§8.3, evidence §15.3–15.5): required-form
    candidates are saturated on this policy, ordered-pair candidates cannot exist under an *unconditional*
    agreement floor (a multi-element constraint is satisfied by professionals less often than each element
    alone, so `t1 .* t2` cannot reach 0.65), and exclusion candidates degenerate into reference imitation
    (penalising `während` because professionals prefer `wärend` is a chrF term, not grammar). Fixing the
    second requires target-side conditional applicability, which is gameable by omission and would need an
    abstention penalty — deliberately not done here.

---

## 13. Troubleshooting

| symptom | cause / fix |
|---|---|
| `AttributeError: 'VerificationReport' object has no attribute '__dict__'` in `main.py verify` | fixed here: the report dataclasses are `@dataclass(slots=True)` (no `__dict__`), so serialization uses `dataclasses.asdict`; guarded by `tests/test_p0_gates.py::test_main_verify_subcommand_emits_json` |
| `build-mbg` appears to hang | its defaults name a server on another machine (`http://10.6.32.16:8000/v1`); pass `--llm-endpoint` and `--llm-model` |
| `ModuleNotFoundError: grammetarl` from `grammetarl/training/*.py` | run with `PYTHONPATH=.` |
| Strata returns `content: null` | thinking model with too small a budget — always pass `max_tokens`, raise it |
| Trigger batches silently missing cards | ~45 s/card on the disk-backed MoE host after the machine move; use `--batch-size 3–4 --timeout 850` (400 s killed 3/13 batches silently once) |
| `0/292 applicable`, or a rule judging nothing | applicability genuinely absent (English marks no formality) → the rule is undecidable for `en→lb`; drop it from the machine-checked set rather than loosening the pattern |
| A gate passes on one pair | support floors: `--min-support 30 --min-tested 20` |
| Held-out numbers look identical to fit numbers | confirm the matrix is measuring in-run and `--reliability` points at the **fit** table |
| `pkill -f spark_worker.sh` kills the ssh session | pattern matches the ssh command line; use the bracket form `spark_[w]orker.sh` |
| a background run's log file "does not exist" for the first minute | `/raid` NFS visibility lag (~30–60 s) for files created by a detached process. `ps` shows the process immediately — do **not** relaunch because `ls`/`grep` found no log, that is how a job runs twice. Start with `setsid … < /dev/null & disown` and poll in a separate exec |
| a smoke metric "drops" after changing the rule set | probably prompt selection, not the rules (§12 item 12); re-score the stored rollout log with `scripts/p1_signal_by_rule.py` before believing it |
| `nonzero_spread_rate` jumps between runs of the same config | n is small: read `nonzero_spread_rate_ci95`, and use `--prompts-per-rule` so each rule gets its own groups |
| the gate tally differs from the README (PASS=27 FAIL=4 instead of 29/3) | the policy-dependent gates need the endpoint: run `scripts/smoke_test.py --cards … --policy-endpoint http://127.0.0.1:11434/v1 --policy-model qwen2.5-coder:7b` |
| a mined candidate looks great at stage B | check *which* sample estimated what: stage A (obligatoriness, lift) must be pool statistics on the fit half only, otherwise the held-out certification is measuring candidates that were selected on it. Mining from the ~100-pair rollout corpus alone gives lift≈1 junk (`has → den`) |

---

## 14. Roadmap as currently written

| plan | one-line status |
|---|---|
| **P1** card-to-verifier reward | reward side measured end-to-end and found **not reward-ready**: advantage coverage 11.9% [5.2–25.0] stratified / **0.0%** unbiased, and gradient concordance with the professional reference 20% (G31 FAIL). The certified set is a working **label and evaluation** asset (2,555 CC0 pairs, rule-level held-out now defined), not yet a training term |
| **P2** turn-level credit | blocked on P1 having a non-degenerate signal |
| **P3** rules as code (morphology/POS engine) | the strongest case made by data: undecidable `equivalence`/`optionality` (~13% of constraints), the false-satisfy failure mode, the 19.2% possessive+adjective construction, and now the saturated-cue result (surface cues cannot see the differences between rollouts) |
| **P4** curriculum / self-play | not started |
| **P5** benchmark transfer | not started |

The previously stated next step — "card coverage, not reward tuning" — is **falsified**: coverage went 2 → 8
and advantage coverage did not follow. Its successor ("mine rules from the policy's own errors") is now
also **executed and largely falsified**: three candidate families mined with
`scripts/mine_policy_gaps.py` produced zero new certified rules (§8.3, evidence §15). The next concrete
steps, in order of measured payoff:

1. **Change the mining criterion from gap to concordance.** `gap > 0` says the policy differs from the
   reference; it does not say the rule ranks rollouts the way the reference does. `scripts/p1_signal_by_rule.py`
   now emits per-rule alignment offline, so thousands of candidates can be screened against stored rollouts
   with zero model calls, and G31 is the gate that the winner must clear.
2. **Exclude negative-gap rules from the reward term** (keep them as labels/eval features) — a correctness
   lever, not a density lever: it lowered spread 11.9% → 7.1% on identical hypotheses while removing the
   pull away from professional usage (§8.2, G30).
3. **SFT warmup** on `gated_condv3` (train 2,183 / valid 243 / rule-held-out 129). Surface cues are
   saturated for a competent 12B generalist; the regime where grammar checks bite is one whose errors are
   grammatical rather than stylistic. This is now the highest-value unstarted experiment.
4. **P3 (morphology/POS)** moved up: it is the only remaining route to teachable surface rules — verb
   position, agreement, and the possessive+adjective construction are invisible to token-pattern checks, and
   order-type constraints cannot be certified at all under the current unconditional agreement floor.
5. Statistical power: ≥6 prompts per rule and ≥3 rollouts, and read the CI, not the point. For alignment
   specifically, n = 10 decisive pairs means the next alignment run needs ~10× more groups (or more
   rollouts per group) before a 20%-vs-80% difference can be called a result.
