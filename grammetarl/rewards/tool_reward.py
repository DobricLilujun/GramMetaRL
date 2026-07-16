from __future__ import annotations


def tool_reward(action_names: list[str]) -> float:
    if not action_names:
        return -0.2
    # Encourage grammar-first then dictionary usage.
    reward = 0.0
    if action_names[0] == "CALL_GRAMMAR_SEARCH":
        reward += 0.4
    if "CALL_DICTIONARY" in action_names:
        reward += 0.2
    if action_names.count("CALL_GRAMMAR_SEARCH") > 3:
        reward -= 0.2
    return reward
