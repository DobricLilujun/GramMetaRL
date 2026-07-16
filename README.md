# GramMetaRL

Grammar-First Translation Toolkit focused on extracting Meta Bullet Grammar (MBG)
cards from grammar books and preparing them for retrieval, verification, and
future RL-based tool-using agents.

This README is implementation-oriented and intended as a learning guide.

## 1. What This Project Does

The project converts grammar resources into atomic MBG cards that are:

- Retrievable: find relevant rules for an input sentence.
- Composable: order and filter rules by dependency and override signals.
- Verifiable: check whether output translations satisfy constraints.

Current practical workflow is OCR-first:

1. Run OCR on the book and save page-level structured JSONL.
2. Build MBG cards directly from those OCR artifacts.
3. Use retrieval + verification + (future) RL environment.

## 2. Repository Map

- Core MBG extraction
	- [main.py](main.py)
	- [grammetarl/pipeline.py](grammetarl/pipeline.py)
	- [grammetarl/llm_extract.py](grammetarl/llm_extract.py)
	- [grammetarl/schema.py](grammetarl/schema.py)
	- [grammetarl/workspace.py](grammetarl/workspace.py)

- Prompt system (Jinja centralized management)
	- [grammetarl/prompt_manager.py](grammetarl/prompt_manager.py)
	- [grammetarl/prompts/jinja](grammetarl/prompts/jinja)

- OCR utilities
	- [scripts/extract_pdf_page_ocr.py](scripts/extract_pdf_page_ocr.py)
	- [grammetarl/ocr.py](grammetarl/ocr.py)

- RL scaffold
	- [grammetarl/agent_env](grammetarl/agent_env)
	- [grammetarl/tools](grammetarl/tools)
	- [grammetarl/rewards](grammetarl/rewards)
	- [grammetarl/training](grammetarl/training)
	- [configs](configs)

Legacy top-level folders [agent_env](agent_env), [tools](tools),
[rewards](rewards), and [training](training) are now compatibility wrappers
that forward to the canonical modules under [grammetarl](grammetarl).

## 3. Install and Environment

Install package:

```bash
pip install -e .
```

Key dependencies already in project config include:

- transformers / torch for model-side workflows
- pymupdf for PDF processing
- requests for OpenAI-compatible endpoints
- jinja2 for prompt templating

Project metadata is in [pyproject.toml](pyproject.toml).

### 3.1 Standardized Workspace Layout

Use one standardized layout for OCR, extraction, data preparation, and RL:

- Inputs
	- `data/raw/pdf`
	- `data/raw/ocr`

- Core data
	- `data/rules/mbg_cards.jsonl`
	- `data/lexicon/dictionary.jsonl`
	- `data/processed/train.jsonl`

- Artifacts
	- `artifacts/ocr`
	- `artifacts/extraction`
	- `artifacts/rl`
	- `artifacts/eval`
	- `artifacts/logs`

Initialize folders once:

```bash
python main.py init-workspace
```

Initialize a namespaced experiment layout:

```bash
python main.py --namespace exp_mbg_v1 init-workspace
```

Default namespace resolution order:

1. CLI `--namespace`
2. env `GRAMMETARL_NAMESPACE`
3. fallback `default`

Path templates are configurable in [configs/project/paths.yaml](configs/project/paths.yaml).

Migrate legacy outputs into the new layout:

```bash
# plan only
python main.py migrate-workspace --dry-run

# execute migration using symlinks
python main.py migrate-workspace --mode symlink
```

All default paths in CLI, tools, OCR script, and dataset builder now resolve
through [grammetarl/workspace.py](grammetarl/workspace.py).

## 4. MBG Extraction Architecture

### 4.1 Entry Point

Command entry is [main.py](main.py), subcommand build-mbg.

It wires:

- LLM extractor instance
- OCR input mode
- pipeline execution

### 4.2 Two Extraction Modes

Mode A: OCR-first (recommended)

- Input: precomputed OCR JSONL (or OCR directory)
- No online OCR call during MBG extraction
- Sectioning uses OCR block types, especially sub_title

Mode B: live OCR mode (legacy / optional)

- Input: PDF + OCR provider
- Pipeline renders pages and calls OCR during extraction

### 4.3 OCR-First Sectioning Logic (Important)

In [grammetarl/pipeline.py](grammetarl/pipeline.py), OCR-first mode now:

1. Reads pages_ocr.jsonl.
2. Uses block list per page.
3. Splits sections by block label sub_title.
4. Builds section text from blocks directly.
5. Sends each section once to LLM (no chunking in OCR-first mode).

This means the minimal extraction unit is section inferred from OCR sub_title,
not arbitrary text chunk boundaries.

### 4.4 Progress Logs

Pipeline prints real-time progress like:

- [MBG] section i/n title=... pages=x-y
- [MBG] section i/n extracted=a added=b
- [MBG] completed sections=... cards=...

This is visible both in terminal and notebook streaming cells.

## 5. Prompt System (Centralized Jinja)

All active prompts are moved into [grammetarl/prompts/jinja](grammetarl/prompts/jinja), rendered via
[grammetarl/prompt_manager.py](grammetarl/prompt_manager.py).

Key MBG templates:

- [grammetarl/prompts/jinja/mbg_system.j2](grammetarl/prompts/jinja/mbg_system.j2)
- [grammetarl/prompts/jinja/mbg_user.j2](grammetarl/prompts/jinja/mbg_user.j2)

Key behavior currently encoded in prompts:

- language-aware extraction context (language code + language name)
- allow multiple MBG cards from one section
- if section is not grammar-rule content, return []

Other template groups:

- OCR templates
	- [grammetarl/prompts/jinja/ocr_openai_system.j2](grammetarl/prompts/jinja/ocr_openai_system.j2)
	- [grammetarl/prompts/jinja/ocr_openai_user_text.j2](grammetarl/prompts/jinja/ocr_openai_user_text.j2)
	- [grammetarl/prompts/jinja/ocr_deepseek_local_user_text.j2](grammetarl/prompts/jinja/ocr_deepseek_local_user_text.j2)
	- [grammetarl/prompts/jinja/ocr_infer_markdown.j2](grammetarl/prompts/jinja/ocr_infer_markdown.j2)

- verifier templates
	- [grammetarl/prompts/jinja/validator_system.j2](grammetarl/prompts/jinja/validator_system.j2)
	- [grammetarl/prompts/jinja/validator_user.j2](grammetarl/prompts/jinja/validator_user.j2)

## 6. OCR Pipeline and Artifact Format

Use [scripts/extract_pdf_page_ocr.py](scripts/extract_pdf_page_ocr.py).

Main outputs under a chosen output dir:

- pages_ocr.jsonl: page-level records
- blocks_ocr.jsonl: block-level records
- per-page folder with image/result.mmd/raw stdout/structured JSON

Recommended command for full book:

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/extract_pdf_page_ocr.py \
	--namespace exp_mbg_v1 \
	--pdf data/raw/pdf/luxembourgish_grammar.pdf \
	--output-dir artifacts/experiments/exp_mbg_v1/ocr/deepseek_fullbook \
	--overwrite-jsonl
```

## 7. MBG Build Commands

### 7.1 OCR-First Build (No PDF Required)

```bash
python main.py build-mbg \
	--namespace exp_mbg_v1 \
	--output data/rules/mbg_cards_exp_mbg_v1.jsonl \
	--language lb \
	--source-id luxembourgish_grammar_book \
	--ocr-pages-dir artifacts/experiments/exp_mbg_v1/ocr/deepseek_fullbook \
	--llm-endpoint http://10.6.32.16:8000/v1 \
	--llm-model nvidia/Qwen3.6-35B-A3B-NVFP4 \
	--llm-timeout 600
```

### 7.2 OCR-First with Page Range (Recommended for debugging)

```bash
python main.py build-mbg \
	--namespace exp_mbg_v1 \
	--output artifacts/extraction/lb_mbg_cards_p36_p40.jsonl \
	--language lb \
	--source-id luxembourgish_grammar_book_p36_p40 \
	--ocr-pages-dir artifacts/experiments/exp_mbg_v1/ocr/deepseek_fullbook \
	--page-start 36 \
	--page-end 40 \
	--llm-endpoint http://10.6.32.16:8000/v1 \
	--llm-model nvidia/Qwen3.6-35B-A3B-NVFP4 \
	--llm-timeout 600
```

### 7.3 Optional Legacy PDF+OCR Build

Still supported if you pass PDF and OCR provider settings, but OCR-first mode is
the current preferred path.

## 8. Notebook for Quick Experiments

Use [notebook/mbg_5page_qwen_test.ipynb](notebook/mbg_5page_qwen_test.ipynb).

Current notebook behavior:

- selects middle 5 pages automatically from OCR artifacts
- runs build-mbg with OCR-only path
- streams progress in cell output
- prints card summary from output JSONL

## 9. MBG Schema and Normalization Notes

Schema is in [grammetarl/schema.py](grammetarl/schema.py).

Extractor-side normalization in [grammetarl/llm_extract.py](grammetarl/llm_extract.py) now includes:

- list/string coercions for many fields
- dependency normalization and invalid-entry filtering
- scope dict -> string fallback for schema compatibility
- unknown operation_type -> fallback mapping

These are practical guards against imperfect LLM JSON.

## 10. Retrieval and Verification

Retrieve rules:

```bash
python main.py retrieve \
	--namespace exp_mbg_v1 \
	--cards data/rules/mbg_cards_exp_mbg_v1.jsonl \
	--sentence "Ech hunn hien net gesinn." \
	--top-k 10
```

Verify translation:

```bash
python main.py verify \
	--namespace exp_mbg_v1 \
	--cards data/rules/mbg_cards_exp_mbg_v1.jsonl \
	--sentence "I did not see him." \
	--translation "Ech hunn hien net gesinn." \
	--rule-ids LB_NEG_0001,LB_WORDORDER_0003
```

## 11. RL Part: What Is Implemented vs Not Yet

Design and target architecture:

- [PRD_VERL_GRAMMAR_AGENT.md](PRD_VERL_GRAMMAR_AGENT.md)

Implemented scaffold:

- environment state/action loop
	- [grammetarl/agent_env/state.py](grammetarl/agent_env/state.py)
	- [grammetarl/agent_env/action_schema.py](grammetarl/agent_env/action_schema.py)
	- [grammetarl/agent_env/env.py](grammetarl/agent_env/env.py)

- tools
	- [grammetarl/tools/grammar_search.py](grammetarl/tools/grammar_search.py)
	- [grammetarl/tools/dictionary_lookup.py](grammetarl/tools/dictionary_lookup.py)
	- [grammetarl/tools/grammar_verify.py](grammetarl/tools/grammar_verify.py)
	- [grammetarl/tools/grammar_apply.py](grammetarl/tools/grammar_apply.py) (currently placeholder-style draft output)

- reward functions
	- [grammetarl/rewards/aggregate.py](grammetarl/rewards/aggregate.py)
	- [grammetarl/rewards/mt_reward.py](grammetarl/rewards/mt_reward.py)
	- [grammetarl/rewards/grammar_reward.py](grammetarl/rewards/grammar_reward.py)
	- [grammetarl/rewards/tool_reward.py](grammetarl/rewards/tool_reward.py)

- configs
	- [configs/algo/grpo.yaml](configs/algo/grpo.yaml)
	- [configs/algo/drgrpo.yaml](configs/algo/drgrpo.yaml)
	- [configs/train/fsdp2_vllm.yaml](configs/train/fsdp2_vllm.yaml)
	- [configs/reward/grammar_reward.yaml](configs/reward/grammar_reward.yaml)

Not fully implemented yet:

- true SFT/GRPO/DrGRPO training runner integration with verl
	- current training entrypoints are stubs in [grammetarl/training](grammetarl/training)

## 12. Quick Validation Commands

Build minimal dataset:

```bash
python main.py init-workspace
python -m grammetarl.training.build_dataset
```

Compile-check major modules:

```bash
python -m py_compile \
	main.py \
	grammetarl/*.py \
	agent_env/*.py \
	tools/*.py \
	rewards/*.py \
	training/*.py
```

Run tests:

```bash
pytest -q
```

## 13. Common Issues and Debug Tips

1. LLM timeout
- Symptom: ReadTimeout from endpoint.
- Mitigation: increase --llm-timeout, keep page range smaller first.

2. Empty output cards
- Often means prompt correctly judged sections as non-grammar.
- Try grammar-dense page ranges (sections with explicit paradigms/rules).

3. Schema mismatch from model output
- Normalization handles many cases, but if new mismatch appears, extend
	normalization in [grammetarl/llm_extract.py](grammetarl/llm_extract.py).

4. OCR quality variance
- Confirm page/block outputs in [artifacts/ocr/deepseek_fullbook](artifacts/ocr/deepseek_fullbook).
- Grammar extraction quality depends heavily on OCR block fidelity.

## 14. Suggested Learning Path

1. Read prompt templates in [grammetarl/prompts/jinja](grammetarl/prompts/jinja).
2. Run notebook [notebook/mbg_5page_qwen_test.ipynb](notebook/mbg_5page_qwen_test.ipynb).
3. Compare outputs from two page ranges: intro pages vs grammar-dense pages.
4. Inspect card JSONL and verify retrieval behavior.
5. Move to RL scaffold understanding in [grammetarl/agent_env](grammetarl/agent_env) and [grammetarl/rewards](grammetarl/rewards).
