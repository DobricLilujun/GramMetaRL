from __future__ import annotations

from enum import Enum


class AgentAction(str, Enum):
    CALL_GRAMMAR_SEARCH = "CALL_GRAMMAR_SEARCH"
    CALL_DICTIONARY = "CALL_DICTIONARY"
    CALL_APPLY_RULES = "CALL_APPLY_RULES"
    RETURN_TRANSLATION = "RETURN_TRANSLATION"
