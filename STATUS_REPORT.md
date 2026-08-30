# GramMetaRL — Project Status Report

_Audit date: 2026-08-29. Remote server: `snt@gpu1.sedan.pro`. Project: `/home/snt/projects/GramMetaRL`. Branch: `main` (clean working tree, 4 commits)._

---

## 0. Verdict (TL;DR)

Your instinct is **correct**: you are at the **data-preparation stage (Stage 0)**, and Stage 0 is **~80% done**, but with a **significant data-quality problem**.

The **RL training core (Stage 1–4) — which is the actual goal of the project — is essentially not started.** Most critically, **`verl` is not a dependency and not installed**, so no training loop (SFT/GRPO/DrGRPO) can run today. All `training/` entrypoints are one-line stubs.

```
Stage 0  Data prep (OCR → MBG cards → index → verify)     ██████████  ~80%  (but data quality is weak)
Stage 1  SFT cold-start (action formatting / grammar-first) ░░░░░░░░░░    0%
Stage 2  GRPO with group sampling                          ░░░░░░░░░░    0%
Stage 3  DrGRPO (when length-hacking appears)             ░░░░░░░░░░    0%
Stage 4  Offline eval + ablation (chrF/BLEU/COMET)        ░░░░░░░░░░    0%
verl integration                                         ░░░░░░░░░░    0%  (not installed)
```

---

## 1. Project goal (as you described + PRD)

Use **verl** to:
1. Extract **meta grammar points (MBG cards)** from a grammar book.
2. Generate **reasoning chains** and **reasoning data**.
3. Use **verl RL (GRPO → DrGRPO)** to strengthen an LLM's **low-resource translation / communication / understanding**.
4. Principle: **grammar-first** tool-use (grammar tools before dictionary), SFT cold-start then RL.

Source book: `assets/lb/luxembourgish_grammar.pdf` (Luxembourgish, `lb`). Target language pair: `en→lb`.

---

## 2. Timeline (git history)

| Date   | Commit  | Message                          |
|--------|---------|----------------------------------|
| 07-14  | 0c7cc73 | Initial commit                   |
| 07-16  | c42e2a0 | first update for RL             |
| 07-16  | 9ac3b39 | update all for organizing the projects |
| 07-20  | 8aba3ae | update for grammar evaluation (latest) |

Last change was **~9 days ago**. Nothing committed since. Working tree is clean (no uncommitted work).

---

## 3. What IS DONE (Stage 0 — data pipeline)

| Component | Location | Status | Evidence |
|---|---|---|---|
| **OCR (full book)** | `scripts/extract_pdf_page_ocr.py`, `artifacts/deepseek_ocr_fullbook/` | ✅ Done | 74 pages, 426 blocks OCR'd to `pages_ocr.jsonl` / `blocks_ocr.jsonl` |
| **MBG extraction (OCR-first)** | `main.py` (`build-mbg`), `grammetarl/pipeline.py`, `grammetarl/llm_extract.py` | ✅ Done | Produces cards via `MBGExtractor` + Jinja prompts; 3 extraction modes (chapter / page_batch / block) |
| **MBG schema** | `grammetarl/schema.py` | ✅ Done | Rich pydantic `MBGCard` (constraints, dependencies, priority, examples, source, retrieval/verifier hints) |
| **Retrieval index** | `grammetarl/index.py` | ✅ Done (real) | Inverted-index BM25-style scoring + dependency topological ordering + override/incompatibility conflict resolution |
| **Verifier** | `grammetarl/validator.py` | ✅ Done (real) | Machine checks (regex / token-count) + optional LLM check |
| **Prompt system** | `grammetarl/prompts/jinja/`, `prompt_manager.py` | ✅ Done | Centralized Jinja (mbg / ocr / validator / example-agent templates) |
| **Workspace + namespace** | `grammetarl/workspace.py`, `configs/project/paths.yaml` | ✅ Done | Namespaced layout + `init-workspace` / `migrate-workspace` |
| **Example / reasoning-chain agent** | `grammetarl/example_agent.py` | ⚠️ Code exists, **0 output produced** | Sidecar "reasoning chain" generator — but all `*_examples.jsonl` are **empty (0 rows)** |
| **Dictionary conversion** | `scripts/convert_lod_xml_to_jsonl.py` | ✅ Done | `assets/lb/new_lod-art.xml` (100 MB) → `new_lod-art.jsonl` (196 MB) |
| **Configs (Hydra-style)** | `grammetarl/configs/` | ✅ Written (not wired) | model / data / tool / reward / algo / train / exp (see §6 for the gap) |
| **Tests** | `tests/test_tools_and_env.py` | ✅ 2 tests, **both pass** | Only tests the env state machine |
| **Notebooks** | `notebook/` | ✅ 5 notebooks | OCR / MBG / experiment-report (mostly exploratory) |

---

## 4. What is NOT DONE (the gaps vs. the goal)

### 4.1 The RL core — not started (the biggest gap)
- **`verl` is NOT in `pyproject.toml` and NOT installed** (0 references in `uv.lock`; `import verl` → `ModuleNotFoundError`). **No training can run today.**
- **`grammetarl/training/` is all stubs:**
  - `run_sft.py` → `print("[stub] SFT entrypoint. Integrate with verl supervised trainer here.")`
  - `run_grpo.py` → `print("[stub] GRPO entrypoint. Use verl trainer ...")`
  - `run_eval.py` → `print("[stub] Eval entrypoint ...")`
  - `build_dataset.py` → writes **exactly 1 hardcoded row** to `train.jsonl`.
- No SFT data, no GRPO group sampling, no DrGRPO, no rollout-backend wiring, no FSDP2/vLLM integration.

### 4.2 No canonical training/eval data — `data/` is EMPTY
- `data/raw/pdf`, `data/raw/ocr`, `data/rules`, `data/lexicon`, `data/processed` are **all empty**.
- Everything produced so far lives in **`artifacts/`** (experiment dirs), **not** in the canonical `data/` layout the configs reference.
- No `train.jsonl` / `valid.jsonl` / `mbg_cards.jsonl` / `dictionary.jsonl` in `data/`.
- **No real `TranslationExample` dataset** built from the extracted cards (src / tgt_ref / phenomena_tags / gold_rules / gold_lexicon).

### 4.3 No evaluation — metrics not wired
- `run_eval.py` is a stub; no chrF / BLEU / COMET integration.
- `grammetarl/eval/README.md` lists the ablation matrix but it is **not implemented**.

### 4.4 The "reasoning chain" part produced nothing
- `example_agent.py` exists, but every `*_examples.jsonl` output is **0 rows**. The reasoning-chain / example-generation path has not yielded usable data.

---

## 5. Data-quality findings (important — the cards may be unusable)

The **best** extraction run is `artifacts/mbg_fullbook_qwen_test_block/lb_mbg_cards_fullbook.jsonl` (**362 cards**). Distribution:

| Metric | Result | Concern |
|---|---|---|
| `operation_type` | **362 / 362 = `fallback`** | 🚩 Every single card collapsed to the fallback bucket — the extraction prompt / normalization is not producing real operations (reorder / negate / inflect / …) |
| `output_constraints` | **362 / 362 = empty** | 🚩 The verifier has **nothing machine-checkable** to check (it returns "No machine-checkable constraints") |
| `examples` | **361 / 362 = empty** | 🚩 Almost no example pairs |
| `scope` | **315 / 362 = "unspecified"** | 🚩 ~87% of cards have no usable scope |
| `phenomenon_tags` | mostly `["grammar_rule"]` | weak / generic |

**Interpretation:** the cards are being generated, but the LLM extractor is falling back to a degenerate schema, so the extracted "rules" carry **no constraints, no examples, and no real operation types**. As-is, these cards **cannot drive a meaningful grammar reward or a useful retrieval-then-verify loop**. This is the single most important thing to fix before any RL work — a training reward built on constraint-less cards will be near-signalless.

(Other runs: page_batch = 106 cards, chapter = 55; all show the same fallback collapse. The `p36_p40` ranges and `with_examples` files contain some content, but the `*_examples.jsonl` sidecars are all empty.)

---

## 6. Code-organization issues found (for the "整理/规范化" task)

1. **`verl` missing from dependencies** — the whole project cannot run its goal. Needs a `verl` (and `ray`, `hydra-core`, `accelerate`, `vllm/sglang`, `comet`) install plan.
2. **Config path mismatch (silent bug):** `grammetarl/workspace.py` reads `project_root/configs/project/paths.yaml`, but `configs/` actually lives at **`grammetarl/configs/`**. The code silently falls back to `{}` and uses **hardcoded default paths**, so the YAML configs are **effectively ignored** today.
3. **README is inaccurate (§2):** it claims "Legacy top-level folders `agent_env`, `tools`, `rewards`, `training` are now compatibility wrappers that forward to the canonical modules under `grammetarl`" — but **those top-level folders do not exist**. The README also points to `configs/` at the top level, but it is under `grammetarl/`.
4. **No top-level package `__init__`-style entry for `training/`** — the training module only exposes `build_min_dataset`; the stubs are not importable as a real interface.
5. **`__pycache__`** present at repo root (should be git-ignored — `.gitignore` ignores `**/__pycache__/**` but a top-level one exists).
6. **No `requirements`-style reproducibility** for the RL deps; `pyproject.toml` has `torch 2.6.0 / transformers 4.46.3` but no verl/ray/hydra.
7. **Extraction quality bug** (§5) is arguably the highest-impact "code" issue.

---

## 7. Recommended organization / standardization plan (proposed — for your confirmation)

I recommend a **safe, incremental** pass. Nothing destructive yet. Ordered by impact:

**A. Make the goal runnable (unblock Stage 1)**
1. Add a proper dependency set for the RL stack: `verl`, `ray`, `hydra-core`, `accelerate`, `vllm` (or `sglang`), `comet` — via `uv`, in a dedicated `rl` extra so the OCR-only path stays lightweight.
2. Fix the **config path** so `grammetarl/workspace.py` resolves `grammetarl/configs/...` (one-line path fix) so the Hydra YAML actually takes effect.

**B. Fix data quality before training (highest-value)**
3. Diagnose the **fallback collapse** in `llm_extract.py` / `mbg_system.j2`: why 100% of cards are `operation_type=fallback`, 0 constraints, 0 examples. Likely a prompt/normalization issue. This is the #1 thing to fix — the reward depends on it.
4. Implement **`build_dataset`** to build a real `TranslationExample` train/valid split from the extracted cards + LOD lexicon (src / tgt_ref / phenomena_tags / gold_rules / gold_lexicon), not 1 hardcoded row.

**C. Wire the RL loop**
5. Implement `run_sft.py` / `run_grpo.py` / `run_eval.py` against `verl` (SFT cold-start → GRPO → DrGRPO), using the existing `agent_env` / `rewards` / `tools` as the custom env, and the `configs/` Hydra stack.
6. Implement `grammar_apply` for real (currently returns the source sentence as the draft — a placeholder).

**D. Housekeeping / standardization**
7. Fix the **README** (remove the false "legacy wrapper" claim; correct the `configs/` location; add the RL install + run instructions).
8. Move canonical data into `data/` (or document that `artifacts/` is the canonical store), and add a `.gitignore` for top-level `__pycache__`.
9. Add real tests for `index.py`, `validator.py`, `pipeline.py` (only `agent_env` is tested today).
10. Standardize docstrings/comments to English (per your requirement) — most are already English; a few notebooks/report use Chinese.

**Suggested order of execution:** **B3 → B4 → A1/A2 → C5 → C6 → D**, so that usable data + a runnable loop come first.

---

## 8. Immediate priority (if you only fix 3 things)
1. **Fix the card-quality collapse** (§5) — otherwise the RL reward is signalless.
2. **Install `verl`** and fix the **config path bug** — otherwise nothing trains.
3. **Build a real training set** in `build_dataset` — the bridge from data-prep to Stage 1.

---

_This report is a snapshot of the current state (last commit 2026-07-20). Recommend re-running `build-mbg` after the §B3 fix and inspecting the card distribution before any training._