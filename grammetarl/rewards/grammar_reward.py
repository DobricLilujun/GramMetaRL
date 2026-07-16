from __future__ import annotations


def grammar_reward(grammar_passed: bool, checked_rules: int) -> float:
    if checked_rules == 0:
        return 0.0
    return 1.0 if grammar_passed else -0.5
