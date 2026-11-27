# GramMetaRL

Compile a descriptive grammar book into **executable grammar constraints**, measure how good each
constraint actually is against professional human translations, and use only the certified ones as a
reinforcement-learning reward for an `en → lb` (English → Luxembourgish) translation agent.

The project's contribution is not "we wrote rules". It is **a measurement contract for rules**: every
constraint that reaches the reward function arrives with a measured agreement rate, a measured
discrimination rate, a measured support count, and a reliability weight derived from them. Numbers that
cannot be measured are reported as undecidable and abstain instead of producing a verdict.

---

## 0. Status at a glance (measured, 2026-11-26)

| item | value | where it comes from |
|---|---|---|
| Gate table | **PASS=28 FAIL=2 WARN=9** | `scripts/smoke_test.py` |
| Unit tests | **31 passed** | `tests/` |
| Cards extracted from the grammar book | **87** | `data/rules/mbg_cards_condonly.jsonl` |
| Rules certified usable (3 axes + support floors, CC0 pool) | **2** | `artifacts/audit/rule_quality_conditional.json` |
| Rules certified on the CC BY-NC pool (reported separately) | **2** (a *different* pair — see §5) | `artifacts/audit/rule_quality_condonly_luxalign.json` |
| Labeled pairs (certified rules only, 100% CC0) | **1,585** | `data/processed/labeled_condonly.jsonl` |
| SFT/RL dataset (train / valid / rule-held-out) | **1,427 / 158 / 0** | `data/processed/gated_condonly/` |
| Honest false-penalty rate (professional translations penalized) | **28.9%** (weighted 29.2%) | `artifacts/audit/gated_condonly_techgov.json` |
| GRPO advantage coverage on policy rollouts | **29.2%** of groups have nonzero spread | `artifacts/p1/reward_smoke_gemma3_12b.json` |
| Reward noise floor (reference translation penalized) | **12.5%** | same artifact |

The two FAIL gates are both about *predicate* applicability (G21: 5/100 trigger conditions are in
executable form; G24: 26/87 cards carry an admitted predicate). They are currently bypassed in practice by
`source_conditional` constraints, which derive applicability from the **source** sentence instead of a
target-side predicate — see §5 and §12.

### 中文快速上手

```bash
# trux 上的轻量环境（不装 verl，不装 torch）
PY=/raid/yewei/.venvs/gm-lite/bin/python
cd /raid/yewei/projects_yewei/GramMetaRL

$PY -m pytest tests/ -q                                   # 30 passed
$PY scripts/smoke_test.py --cards data/rules/mbg_cards_condonly.jsonl \
    --policy-endpoint http://127.0.0.1:11434/v1 --policy-model qwen2.5-coder:7b   # 门控表
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

---

## 6. Labels and the training dataset

```bash
# labels: only matrix-certified rules, reliability-weighted, applicability required
$PY scripts/gated_pipeline_measurement.py \
    --cards data/rules/mbg_cards_condonly.jsonl \
    --reliability data/rules/reliability_condonly.json \
    --quality artifacts/audit/rule_quality_conditional.json --min-agreement 0.65 \
    --rows 0 --source-filter techgov_lu_en \
    --labeled-out data/processed/labeled_condonly.jsonl \
    --out artifacts/audit/gated_condonly_techgov.json
# -> pairs_scored 2,228 / labeled_pairs 1,585 / decided_checks 2,283 / false_penalty 28.9%

# dataset: pair-level splits; rule-level held-out needs >= ~7 rules to be meaningful
PYTHONPATH=. $PY grammetarl/training/build_dataset.py \
    --from-cards data/rules/mbg_cards_condonly.jsonl \
    --external-tagged data/processed/labeled_condonly.jsonl \
    --out-dir gated_condonly --no-lexicon
# -> train 1,427 / valid 158 / valid_rules_heldout 0  (100% CC0-1.0)
```

`data/processed/labeled_condonly.jsonl` is the current label file. **`labeled_conditional.jsonl`
(3,269 rows) is superseded** — it was produced before the discrimination gate and contains ~2,000 checks
from a rule that cannot bite. Do not train on it.

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
$PY scripts/p1_reward_smoke.py --prompts 24 --rollouts 3 --temperature 0.9 \
    --cards data/rules/mbg_cards_condonly.jsonl \
    --reliability data/rules/reliability_condonly.json \
    --quality artifacts/audit/rule_quality_conditional.json \
    --policy-endpoint http://127.0.0.1:11434/v1 --policy-model gemma3:12b \
    --timeout 400 --max-tokens 220 \
    --out artifacts/p1/reward_smoke_gemma3_12b.json \
    --rollout-log artifacts/p1/rollouts_gemma3_12b.jsonl      # log is not tracked, JSON is
```

Measured per policy (this is the table that decides which policy to warm up):

| policy | Luxembourgish? | `nonzero_spread_rate` | reading |
|---|---|---|---|
| `gemma3:12b` | real, imperfect | **0.292** | 71% of groups give zero advantage; `reference_penalized_rate` 12.5%; `policy_minus_reference` **−0.056** (two certified rules barely separate a 12B generalist from professionals → signal near saturation) |
| `qwen2.5-coder:7b` | gibberish | 0.0 | every rollout scores −1.0 → zero advantage |
| `qwen2.5:14b` | gibberish | 0.0 | same |
| `gemma3:4b-it-fp16` | weak/gibberish | 0.0 | same |
| `qwen3:32b`, `gpt-oss:20b` | — | — | never answered within 400 s on trux |

**Capability is the first-order effect.** A policy with no target-language competence cannot be trained by
this reward at all, because a group of identical rewards has zero advantage. That is the empirical
argument for the SFT warmup step in `paper/research_plans/P1_card_to_verifier_reward.md`.

---

## 9. Gates

```bash
$PY scripts/smoke_test.py --cards data/rules/mbg_cards_condonly.jsonl \
    --policy-endpoint http://127.0.0.1:11434/v1 --policy-model qwen2.5-coder:7b
```

Current: **PASS=28 FAIL=2 WARN=9.** `P0 gate is GREEN only when every FAIL is cleared (WARN = measured
risk, report it).`

Gate families:

- G1–G19 schema, retrieval, env, tools, dataset integrity, chrF discriminativeness
- G20/G23 split hygiene (no pair overlap; rule-level held-out excludes those rules)
- G21 trigger executability — **FAIL** (5/100 in regex form)
- G24 admitted applicability predicates — **FAIL** (26/87)
- G25 no prose/identifier in a regex slot
- G26 reward is reliability-weighted, not oracle-assumed
- G27 ≥1 rule with agreement ≥ 0.65 **and** discrimination ≥ 0.5 **and** not its own trigger
- G28 P1 reward signal measured on rollouts; **G28b WARN** advantage coverage < 50% (29.2%)
- G29 reward weights inside the measured band

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

Tracked: code, `tests/`, card sets `data/rules/*.jsonl`, rule specs
(`data/rules/{source_conditionals,manual_triggers,reliability*}.json`), CC0 labels
`data/processed/labeled_*.jsonl`, and the evidence JSON:
`artifacts/audit/{precision_*,rule_quality_*,condonly*,gated_condonly*,violations_*,trigger_quality*}.json`
plus `artifacts/p1/*.json`.

Not tracked: raw OCR output, rollout logs (`*.jsonl` under `artifacts/p1/`), model caches, the Overleaf
clone in `paper/` (it has its own git remote; only push it when explicitly authorized), and `data/external/`
(48k rows with per-row licenses — regenerate with `scripts/fetch_external_pairs.py`, provenance in
`MANIFEST.json`).

Deeper documents:

| file | content |
|---|---|
| `P0_DATA_P1_EVIDENCE_2026-11-20.md` | the full session record, §1–12, including every wrong claim and its correction (§10 is explicitly annotated "superseded by §11") |
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
7. **Rule-level held-out is undefined at n=2 rules** (`valid_rules_heldout.jsonl` is empty). Generalization
   across rules cannot be measured until more rules are certified; use the fit/held-out *row* protocol of
   §10 meanwhile.
8. **Judge cache is first-wins.** Appending a re-ask cannot overwrite an earlier judgment; delete the
   keyed line if a judgment must change.
9. **Interpretation attached to a regex bucket produces false findings.** Luxembourgish capitalizes every
   noun, so "article + capitalized token" is *not* a proper-name test (it matched
   `den Attentater` = the attackers). `violation_decomposition.py` therefore takes buckets as
   `NAME=REGEX` and adds no meaning.

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

---

## 14. Roadmap as currently written

| plan | one-line status |
|---|---|
| **P1** card-to-verifier reward | reward side measured end-to-end; **29.2% advantage coverage** and **2 certified rules** are the binding constraints; SFT warmup now empirically required |
| **P2** turn-level credit | blocked on P1 having a non-degenerate signal |
| **P3** rules as code (morphology/POS engine) | the strongest case made by data: undecidable `equivalence`/`optionality` (~13% of constraints), the false-satisfy failure mode, and the 19.2% possessive+adjective construction |
| **P4** curriculum / self-play | not started |
| **P5** benchmark transfer | not started |

The next concrete step with the best measured payoff is **card coverage, not reward tuning**: compile more
`source_conditional` cards (the only applicability route with measured precision 0.50–0.57), certify them
through §10, then re-run §8 and check whether `nonzero_spread_rate` rises.
