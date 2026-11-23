from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .action_schema import AgentAction
from .state import AgentState
from grammetarl.tools import dictionary_lookup, grammar_rule_apply, grammar_rule_search, grammar_verify


class GrammarFirstEnv:
    """Grammar-first agent environment.

    The tool layer resolves cards/lexicon from the workspace default unless the env is constructed
    with explicit paths, so an RL run on a namespaced card set actually sees those cards. The
    previous version never threaded cards_path/namespace, meaning a named experiment silently
    verified against the default rule set.
    """

    def __init__(
        self,
        step_budget: int = 6,
        cards_path: str | None = None,
        lexicon_path: str | None = None,
        namespace: str | None = None,
        policy_endpoint: str | None = None,
        policy_model: str | None = None,
        policy_api_key: str | None = None,
    ) -> None:
        self.step_budget = step_budget
        self.cards_path = cards_path
        self.lexicon_path = lexicon_path
        self.namespace = namespace
        self.policy_endpoint = policy_endpoint
        self.policy_model = policy_model
        self.policy_api_key = policy_api_key
        self.last_verification: dict[str, Any] | None = None
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
                cards_path=self.cards_path,
                namespace=self.namespace,
            )
        elif act == AgentAction.CALL_DICTIONARY:
            entry = dictionary_lookup(
                token=payload.get("token", ""),
                lemma=payload.get("lemma"),
                pos=payload.get("pos"),
                context=self.state.src_sentence,
                lexicon_path=self.lexicon_path,
                namespace=self.namespace,
            )
            self.state.dictionary_entries.extend(entry)
        elif act == AgentAction.CALL_APPLY_RULES:
            apply_result = grammar_rule_apply(
                rule_ids=payload.get("rule_ids", []),
                src_sentence=self.state.src_sentence,
                lexical_hints=payload.get("lexical_hints"),
                cards_path=self.cards_path,
                namespace=self.namespace,
                llm_endpoint=self.policy_endpoint,
                llm_model=self.policy_model,
                api_key=self.policy_api_key,
            )
            # None means "no draft yet": the policy must produce the translation. Never fabricate.
            draft = apply_result.get("draft")
            self.state.draft_translation = str(draft) if draft else ""
        elif act == AgentAction.CALL_GRAMMAR_VERIFY:
            hypothesis = str(
                payload.get("translation")
                or self.state.draft_translation
                or self.state.final_translation
                or ""
            ).strip()
            rule_ids = payload.get("rule_ids") or [
                r.get("rule_id") for r in self.state.retrieved_rules if isinstance(r, dict)
            ]
            if not hypothesis:
                self.last_verification = {"error": "no hypothesis to verify yet"}
            else:
                self.last_verification = grammar_verify(
                    src_sentence=self.state.src_sentence,
                    hypothesis=hypothesis,
                    applied_rule_ids=[r for r in rule_ids if r],
                    cards_path=self.cards_path,
                    namespace=self.namespace,
                )
        elif act == AgentAction.RETURN_TRANSLATION:
            self.state.final_translation = str(payload.get("translation", "")).strip()
            self.state.done = True

        self.state.step_budget -= 1
        if self.state.step_budget <= 0:
            self.state.done = True

        # Verification results are tool output, not agent state, but the policy must see them or
        # CALL_GRAMMAR_VERIFY cannot steer anything.
        return {**asdict(self.state), "verification": self.last_verification}, self.state.done
