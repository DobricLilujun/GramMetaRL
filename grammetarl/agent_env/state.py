from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentState:
    sample_id: str
    src_sentence: str
    step_budget: int = 6
    retrieved_rules: list[dict[str, Any]] = field(default_factory=list)
    dictionary_entries: list[dict[str, Any]] = field(default_factory=list)
    draft_translation: str = ""
    actions: list[dict[str, Any]] = field(default_factory=list)
    done: bool = False
    final_translation: str = ""
