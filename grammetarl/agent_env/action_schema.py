from __future__ import annotations

from enum import Enum


class AgentAction(str, Enum):
    CALL_GRAMMAR_SEARCH = "CALL_GRAMMAR_SEARCH"
    CALL_DICTIONARY = "CALL_DICTIONARY"
    CALL_APPLY_RULES = "CALL_APPLY_RULES"
    # Without this action the verifier is unreachable from the action space, so the grammar term
    # of the reward can only be computed after the episode and cannot steer the policy.
    CALL_GRAMMAR_VERIFY = "CALL_GRAMMAR_VERIFY"
    RETURN_TRANSLATION = "RETURN_TRANSLATION"
