# PRD: Built-on-verl Grammar-First Agentic RL

## 1. Objective
Train a translation agent that prioritizes grammar-book tools before dictionary tools, and learns rule retrieval, rule composition, rule application, and final translation generation via RL.

Core principle:
- Keep verl training loop.
- Implement custom environment, tools, and reward functions.
- Use GRPO first, switch to DrGRPO when length-hacking appears.

## 2. Scope
In scope:
- Multi-turn tool-calling environment.
- MBG rule cards and dictionary entries.
- Reward shaping for translation quality + grammar consistency + tool policy.
- SFT cold-start then GRPO/DrGRPO.

Out of scope:
- Re-implementing RL trainers from scratch.
- Replacing verl rollout backends.

## 3. Success Metrics
- Translation quality: chrF/BLEU/COMET improves over SFT baseline.
- Grammar compliance: grammar_verify pass rate improves.
- Tool efficiency: fewer invalid or redundant calls.
- Robustness: stable training under FSDP2 + vLLM/SGLang rollout.

## 4. System Modules
- configs/: Hydra configs for model/data/tool/reward/algo/train/exp.
- data/: raw/processed/rules/lexicon and build scripts.
- tools/: grammar search/apply/verify and dictionary lookup.
- agent_env/: multi-turn environment, state machine, action schema.
- rewards/: mt/grammar/tool/trace/efficiency and aggregator.
- prompts/: tool-calling system/planner/final answer schemas.
- training/: SFT, GRPO, DrGRPO, eval entrypoints.
- eval/: ablation, trajectory replay, failure analysis.

## 5. Data Contracts
### TranslationExample
- id
- src
- tgt_ref
- lang_pair
- difficulty
- phenomena_tags
- gold_rules (optional)
- gold_lexicon (optional)

### MBGCard
Use schema in grammetarl/schema.py.

### DictionaryEntry
- lemma
- pos
- senses
- morphology
- examples

### TrajectorySupervision (optional)
- action sequence
- tool arguments
- expected final translation

## 6. Agent Environment
Action space:
- CALL_GRAMMAR_SEARCH
- CALL_DICTIONARY
- CALL_APPLY_RULES
- RETURN_TRANSLATION

State includes:
- src sentence
- retrieved rules
- dictionary entries
- current draft
- action history
- step budget and done flag

Termination:
- RETURN_TRANSLATION
- step budget exhausted
- unrecoverable tool failure

## 7. Rewards
R = a*R_mt + b*R_grammar + c*R_tool + d*R_trace + e*R_efficiency

Default priority:
- b > c > a >= d > e
- grammar reward weight must exceed lexical/tool-only shortcuts.

## 8. Training Plan
- Stage 0: build MBG + lexicon + retrieval index.
- Stage 1: SFT for action formatting and grammar-first policy.
- Stage 2: GRPO with group sampling.
- Stage 3: DrGRPO when trajectory length bias appears.
- Stage 4: offline eval and ablation.

## 9. Ablation Matrix
- no tools
- dictionary only
- grammar only
- grammar-first no RL
- grammar-first + GRPO
- grammar-first + DrGRPO

## 10. Risks and Mitigations
- Risk: reward hacking via long trajectories.
  - Mitigation: DrGRPO + efficiency reward + step budget.
- Risk: retrieval bottleneck.
  - Mitigation: two-stage retrieval BM25/embedding + rerank.
- Risk: brittle tool outputs.
  - Mitigation: strict schemas + robust normalization + tests.
