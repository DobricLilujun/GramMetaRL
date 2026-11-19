# P0 Smoke Test Baseline — 2026-11-19 (trux-tsubame)

Harness: `scripts/smoke_test.py` (no GPU, no verl, no OCR, no external API needed).

```bash
# structural + reward + metric gates on a fixture card set
/raid/yewei/.venvs/gm-lite/bin/python scripts/smoke_test.py

# same gates auditing a real card file
/raid/yewei/.venvs/gm-lite/bin/python scripts/smoke_test.py --cards artifacts/smoke/cards_real_smoke.jsonl
```

Environment used: `uv venv /raid/yewei/.venvs/gm-lite` (pydantic, jinja2, pyyaml, requests, pymupdf,
pytest). The repo's `.venv/` is dead (symlink to a removed machine); `tests/` passes 2/2 in the new venv.

## Measured results

| Gate | Fixture cards | **Real cards (3, from PDF pp.11–16 + local Ollama qwen2.5-3b)** |
|---|---|---|
| G1 imports | PASS | PASS |
| G2 default cards file exists | FAIL (`data/rules/mbg_cards.jsonl` missing) | FAIL |
| G3 schema keeps `statement/confidence/rule_status/exceptions_and_variants` | FAIL (all 4 dropped) | FAIL |
| G4 no prose-as-regex constraints | FAIL (5 compilable / 1 prose) | **FAIL (0 compilable / 3 prose)** |
| G5 verifier separates good/bad on an executable card | PASS | – (fixture) |
| G5b each card passes its own example | PASS (after orientation fix) | – |
| G5c punctuation normalization | FAIL (`$`-anchored constraint breaks on a period) | – |
| G6 grammar reward non-degenerate | PASS (std 0.742) | – |
| G7a/b/c index ordering, override, retrieval | PASS | – |
| G8 env runs a grammar-first trajectory | PASS (4 steps, terminates) | – |
| G10 `grammar_apply` is not a placeholder | **FAIL (draft == source)** | FAIL |
| G12 train rows are sentences with `gold_rules` | **FAIL** (mean 1.60 src tokens; 0/200 with gold rules) | FAIL |
| G13 eval replay produces a hypothesis ≠ source | FAIL (hypothesis = English source) | FAIL |
| G14 reward ordering copy < wrong attempt < correct | PASS (−0.080 < 0.003 < 0.687) | – |
| G15 chrF discriminative | **FAIL: `chrF(ref,ref) = 2.015`** | – |
| G16 no `None/unspecified` sentinels in `scope` | FAIL | **FAIL 3/3 cards** |
| G17 examples are non-degenerate pairs | PASS | **FAIL: 0 usable, 5 degenerate** |
| G18 `operation_type` spread | – | WARN `{reorder: 3}` (phonotactics labelled as *reorder*) |
| G19 `phenomenon_tags` sanity | – | WARN (`word_order`/`agreement` noise on phonology) |
| G11a index rebuild | 0.28 ms/call | (scales with card count) |
| G11b dictionary linear scan | **29 ms/call → ~8 min of pure scanning per 8k-rollout RL pass** | |
| G9 dictionary on real LOD lexicon | works (2 hits for "Mann") | |

Fixture run: `PASS=10 FAIL=9 WARN=9`. Real-card run: same 8–9 hard FAILs.

## What was newly established (not in the 2026-11-18 audit)

1. **`run_eval._chr_f` can exceed 1.0** — it returns **2.015** for a string compared with itself, and
   0.646 for English-source vs Luxembourgish-reference. Any chrF number from this function is unusable.
   (`rewards/mt_reward.py`'s set-overlap `mt_reward` is separately order-blind.)
2. **Example orientation is inverted for en→lb.** `_normalize_card_obj` maps prompt
   `{sentence, translation}` → `{source_sentence=sentence, target_sentence=translation}`. The grammar-book
   sentence is the *target*, so verifiers currently check constraints against the **English** side. Measured
   on real cards: **5/5 examples degenerate** (empty or identical src/tgt).
3. **`scope` leaks the literal string `"None"`** — `str(scope_val.get(k))` on a `None` component; 3/3 real
   cards carry `... | None`.
4. **`GrammarFirstEnv` cannot be pointed at experiment cards**: `env.step` calls the tools without
   `cards_path`/`namespace`, so the env only ever reads the default `data/rules/mbg_cards.jsonl`, and
   *raises* if it is missing. Per-experiment isolation (namespaces) is impossible today.
5. **No normalization layer in `TranslationVerifier`**: anchors (`$`) fail on trailing punctuation.
6. **`AgentAction` has no `CALL_GRAMMAR_VERIFY`**, so the PRD's headline metric (grammar-verify pass
   rate) is unreachable inside the environment where the policy is trained.
7. **Dictionary tool costs 29 ms/call** (linear scan of 1.3 MB JSONL) — ~8 min of pure scanning per RL
   pass at 8k rollouts, before any GPU work.
8. The **PDF has a complete text layer** (122,565 chars, ~1,656/page), so Stage-0 extraction runs **without
   OCR**: `--ocr-provider none` works (57 s for a 6-page slice with a local 3B model). OCR was never the
   blocker for card extraction.

## Verdict on the reward (the thing that decides every RL plan)

On **real** extracted cards: `compilable=0, prose_as_regex=3`. Feeding this into
`rewards/grammar_reward.py` gives a **constant verdict per card**, i.e. `R_grammar` carries no signal and
GRPO would faithfully amplify it (cf. arXiv:2503.06639 on success amplification). **P1's self-test
admission gate is therefore a hard prerequisite, not an optimization.**

## Fix order that turns the gates green

| Fix | Files | Gates it turns green |
|---|---|---|
| add 4 schema fields; drop `str(None)` leakage | `schema.py`, `llm_extract.py` | G3, G16 |
| reject prose values (must compile **and** must not read as prose) → force `llm_check` | `llm_extract.py:180-200` | G4 |
| fix example orientation (`source=en`, `target=lb`) | `llm_extract.py` | G17 |
| normalization layer (NFKC, lower, strip punctuation) before matching | `validator.py` | G5c |
| replace `_chr_f` with a correct chrF (or `sacrebleu`/`unbabel-comet`) | `training/run_eval.py` | G15 |
| real draft generation instead of `draft = src` | `tools/grammar_apply.py` | G10, G13 |
| sentence-level dataset with `gold_rules` (cards + LuxBank) | `training/build_dataset.py` | G12 |
| thread `cards_path`/`namespace` through `GrammarFirstEnv` | `agent_env/env.py` | (removes G8c) |
| add `CALL_GRAMMAR_VERIFY` action | `agent_env/action_schema.py`, `env.py` | (removes G8b) |
| process-global lexicon dict + cached `RuleIndex` | `tools/*.py` | (G11a/b → ~0) |
