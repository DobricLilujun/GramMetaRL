from __future__ import annotations

from .grammar_reward import grammar_reward, grammar_reward_verdicts, grammar_reward_weighted
from .mt_reward import mt_reward
from .tool_reward import tool_reward


def aggregate_reward(
    hypothesis: str,
    reference: str,
    grammar_passed: bool,
    checked_rules: int,
    verdicts: list[str] | None = None,
    rule_ids: list[str] | None = None,
    reliability=None,
    action_names: list[str] | None = None,
    w_mt: float = 0.25,
    w_grammar: float = 0.40,
    w_tool: float = 0.20,
    w_trace: float = 0.10,
    w_eff: float = 0.05,
) -> float:
    action_names = action_names or []
    r_mt = mt_reward(hypothesis, reference)
    # Prefer the three-valued form: rules that do not apply must not move the reward.
    if verdicts is not None and rule_ids is not None and reliability is not None:
        # Reliability-weighted: each rule's verdict counts in proportion to how well that rule
        # actually agrees with human translations (measured audit), never as an assumed oracle.
        r_grammar = grammar_reward_weighted(verdicts, reliability.weights(rule_ids))
    elif verdicts is not None:
        r_grammar = grammar_reward_verdicts(verdicts)
    else:
        r_grammar = grammar_reward(grammar_passed, checked_rules)
    r_tool = tool_reward(action_names)
    # Placeholders for future detailed trace/efficiency metrics.
    r_trace = 0.0
    r_eff = -0.01 * max(0, len(action_names) - 4)
    return w_mt * r_mt + w_grammar * r_grammar + w_tool * r_tool + w_trace * r_trace + w_eff * r_eff
