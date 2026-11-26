from __future__ import annotations

from collections.abc import Sequence

SATISFIED = "satisfied"
VIOLATED = "violated"
NOT_APPLICABLE = "not_applicable"
UNKNOWN = "unknown"


def grammar_reward_verdicts(verdicts: Sequence[str]) -> float:
    """Grammar reward over three-valued rule verdicts.

    `not_applicable` and `unknown` contribute nothing. The invariant this encodes: adding a rule
    that does not govern the sentence must not change the reward. Without it, retrieval alone
    guarantees noise — measured on 48,436 real en->lb pairs, 97.8% of sentences retrieved at least
    one executable card and 85.2% of the resulting checks failed, mostly because the retrieved
    rule never applied to that sentence.

    Returns 0.0 when nothing was decided, so a rollout that only triggers abstentions gets no
    grammar signal rather than a free positive or negative.
    """
    decided = [v for v in verdicts if v in (SATISFIED, VIOLATED)]
    if not decided:
        return 0.0
    pos = sum(1 for v in decided if v == SATISFIED)
    neg = len(decided) - pos
    return (pos - neg) / len(decided)


def grammar_reward_weighted(verdicts: Sequence[str], weights: Sequence[float] | None = None) -> float:
    """Reliability-weighted grammar reward.

    `weights` aligns with `verdicts` (see grammetarl.rewards.reliability). Measured fact behind it:
    the compiled constraint sets disagree with professional human translations on 34-72% of the
    judge-applicable pairs depending on the card set, and the agreement varies hugely per rule
    (one rule 0/15 wrong, another 44/44 wrong). Averaging verdicts as if every rule were a reliable
    oracle therefore injects rule-level error straight into the policy gradient.

    A weight of 0 (unmeasured or coin-flip rule) removes the check entirely, so the reward stays
    exactly the unweighted mean over the checks that earned a weight. With all weights equal this
    reduces to grammar_reward_verdicts, which is asserted in tests.
    """
    if weights is None:
        return grammar_reward_verdicts(verdicts)
    if len(weights) != len(verdicts):
        raise ValueError("weights must align 1:1 with verdicts")
    num = 0.0
    den = 0.0
    for v, w in zip(verdicts, weights):
        if v not in (SATISFIED, VIOLATED) or w <= 0.0:
            continue
        num += w * (1.0 if v == SATISFIED else -1.0)
        den += w
    if den <= 0.0:
        return 0.0
    return num / den


def decided_count(verdicts: Sequence[str]) -> int:
    return sum(1 for v in verdicts if v in (SATISFIED, VIOLATED))


def grammar_reward(grammar_passed: bool, checked_rules: int) -> float:
    """Deprecated binary form: it cannot express abstention.

    Kept so older call sites keep working; new code must use `grammar_reward_verdicts`, because a
    binary pass/fail over all retrieved rules is what made the grammar term mostly noise.
    """
    if checked_rules == 0:
        return 0.0
    return 1.0 if grammar_passed else -0.5
