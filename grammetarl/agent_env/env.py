from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .action_schema import AgentAction
from .state import AgentState
from grammetarl.tools import dictionary_lookup, grammar_rule_apply, grammar_rule_search


class GrammarFirstEnv:
    def __init__(self, step_budget: int = 6) -> None:
        self.step_budget = step_budget
        self.state: AgentState | None = None

    def reset(self, sample_id: str, src_sentence: str) -> dict[str, Any]:
        self.state = AgentState(sample_id=sample_id, src_sentence=src_sentence, step_budget=self.step_budget)
        return asdict(self.state)

    def step(self, action: str, payload: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        if self.state is None:
            raise RuntimeError("env not reset")
        if self.state.done:
            return asdict(self.state), True

        act = AgentAction(action)
        self.state.actions.append({"action": act.value, "payload": payload})

        if act == AgentAction.CALL_GRAMMAR_SEARCH:
            self.state.retrieved_rules = grammar_rule_search(
                query=payload.get("query", ""),
                src_sentence=self.state.src_sentence,
                draft=self.state.draft_translation or None,
                top_k=int(payload.get("top_k", 5)),
            )
        elif act == AgentAction.CALL_DICTIONARY:
            entry = dictionary_lookup(
                token=payload.get("token", ""),
                lemma=payload.get("lemma"),
                pos=payload.get("pos"),
                context=self.state.src_sentence,
            )
            self.state.dictionary_entries.extend(entry)
        elif act == AgentAction.CALL_APPLY_RULES:
            apply_result = grammar_rule_apply(
                rule_ids=payload.get("rule_ids", []),
                src_sentence=self.state.src_sentence,
                lexical_hints=payload.get("lexical_hints"),
            )
            self.state.draft_translation = str(apply_result.get("draft", ""))
        elif act == AgentAction.RETURN_TRANSLATION:
            self.state.final_translation = str(payload.get("translation", "")).strip()
            self.state.done = True

        self.state.step_budget -= 1
        if self.state.step_budget <= 0:
            self.state.done = True

        return asdict(self.state), self.state.done
