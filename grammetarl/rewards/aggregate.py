from __future__ import annotations

from .grammar_reward import grammar_reward
from .mt_reward import mt_reward
from .tool_reward import tool_reward


def aggregate_reward(
    hypothesis: str,
    reference: str,
    grammar_passed: bool,
    checked_rules: int,
    action_names: list[str],
    w_mt: float = 0.25,
    w_grammar: float = 0.40,
    w_tool: float = 0.20,
    w_trace: float = 0.10,
    w_eff: float = 0.05,
) -> float:
    r_mt = mt_reward(hypothesis, reference)
    r_grammar = grammar_reward(grammar_passed, checked_rules)
    r_tool = tool_reward(action_names)
    # Placeholders for future detailed trace/efficiency metrics.
    r_trace = 0.0
    r_eff = -0.01 * max(0, len(action_names) - 4)
    return w_mt * r_mt + w_grammar * r_grammar + w_tool * r_tool + w_trace * r_trace + w_eff * r_eff
