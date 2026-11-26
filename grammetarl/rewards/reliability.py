"""Reliability weights for grammar-rule verdicts, measured rather than asserted.

The audit measures, per card, how often its machine check agrees with professional human
translations on judge-applicable pairs. That agreement is the only honest prior for how much a
verdict from that card should move an RL reward: a card that contradicts human translations half the
time carries no signal, whatever its statement says.

Weight = clamp(2p - 1, 0, 1), where p is the shrunk agreement probability. A coin-flip rule (p=0.5)
gets weight 0, so it cannot inject noise in either direction; a rule that always agrees gets 1.

`shrinkage` pulls a rule's estimate toward the coin flip in proportion to how little evidence backs
it: p_hat = (agreed + k/2) / (n + k). With k=4, a card observed twice must earn its weight instead of
getting 100% from 2/2.
"""
from __future__ import annotations

import json
from pathlib import Path

DEFAULT_SHRINKAGE = 4.0


def agreement_probability(
    agreed: int,
    total: int,
    shrinkage: float = DEFAULT_SHRINKAGE,
) -> float:
    """Posterior-style agreement rate; total==0 means no evidence -> abstain (0.5)."""
    if total <= 0:
        return 0.5
    return (agreed + shrinkage / 2.0) / (total + shrinkage)


def weight_from_agreement(p: float) -> float:
    """0 at chance, 1 when the check always agrees with the human reference."""
    return max(0.0, min(1.0, 2.0 * p - 1.0))


class ReliabilityTable:
    """rule_id -> measured agreement stats, from the cached audit."""

    def __init__(self, table: dict[str, dict], shrinkage: float = DEFAULT_SHRINKAGE) -> None:
        self.table = table
        self.shrinkage = shrinkage

    @classmethod
    def load(cls, path: str | Path, shrinkage: float = DEFAULT_SHRINKAGE) -> "ReliabilityTable":
        p = Path(path)
        if not p.exists():
            return cls({}, shrinkage)
        raw = json.loads(p.read_text("utf-8"))
        # accept either {"rules": {...}} or a bare mapping
        rules = raw.get("rules", raw) if isinstance(raw, dict) else {}
        return cls({str(k): dict(v) for k, v in rules.items() if isinstance(v, dict)}, shrinkage)

    def agreement(self, rule_id: str) -> float:
        st = self.table.get(rule_id)
        if not st:
            return 0.5  # unmeasured rule: no weight, never a free pass
        applicable = int(st.get("applicable", 0))
        violated = int(st.get("violated", 0))
        satisfied = st.get("satisfied")
        if satisfied is None:
            satisfied = max(0, applicable - violated)
        return agreement_probability(int(satisfied), applicable, self.shrinkage)

    def weight(self, rule_id: str) -> float:
        return weight_from_agreement(self.agreement(rule_id))

    def weights(self, rule_ids) -> list[float]:
        return [self.weight(r) for r in rule_ids]

    def __len__(self) -> int:
        return len(self.table)
