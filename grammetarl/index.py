from __future__ import annotations

import re
from collections import defaultdict

from .schema import MBGCard


_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class RuleIndex:
    def __init__(self, cards: list[MBGCard]) -> None:
        self.cards = cards
        self._id_to_card = {c.id: c for c in cards}
        self._inverted: dict[str, set[str]] = defaultdict(set)
        for card in cards:
            for tok in _tokenize(card.as_index_text()):
                self._inverted[tok].add(card.id)

    def retrieve(
        self,
        sentence: str,
        top_k: int = 8,
        required_tags: set[str] | None = None,
    ) -> list[tuple[MBGCard, float]]:
        q_tokens = _tokenize(sentence)
        candidates: set[str] = set()
        for tok in q_tokens:
            candidates.update(self._inverted.get(tok, set()))
        if not candidates:
            candidates = set(self._id_to_card.keys())

        scored: list[tuple[MBGCard, float]] = []
        q_set = set(q_tokens)
        for card_id in candidates:
            card = self._id_to_card[card_id]
            if required_tags and not required_tags.intersection(set(card.phenomenon_tags)):
                continue
            c_set = set(_tokenize(card.as_index_text()))
            overlap = len(q_set.intersection(c_set))
            denom = max(1, len(q_set))
            score = overlap / denom
            score += 0.1 * min(1.0, card.priority / 100.0)
            scored.append((card, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def order_for_application(self, card_ids: list[str]) -> list[MBGCard]:
        selected = [self._id_to_card[cid] for cid in card_ids if cid in self._id_to_card]
        id_set = {c.id for c in selected}

        edges: dict[str, set[str]] = defaultdict(set)
        indeg: dict[str, int] = {c.id: 0 for c in selected}

        for c in selected:
            for dep in c.dependencies:
                if dep.rule_id in id_set and dep.relation == "requires":
                    edges[dep.rule_id].add(c.id)

        for src, dsts in edges.items():
            for dst in dsts:
                indeg[dst] += 1

        queue = [cid for cid, v in indeg.items() if v == 0]
        queue.sort(key=lambda cid: self._id_to_card[cid].priority)
        out: list[str] = []

        while queue:
            cur = queue.pop(0)
            out.append(cur)
            for nxt in edges.get(cur, set()):
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    queue.append(nxt)
                    queue.sort(key=lambda cid: self._id_to_card[cid].priority)

        if len(out) < len(selected):
            remaining = [cid for cid in id_set if cid not in out]
            remaining.sort(key=lambda cid: self._id_to_card[cid].priority)
            out.extend(remaining)

        ordered = [self._id_to_card[cid] for cid in out]
        return self._apply_override_conflict_resolution(ordered)

    def _apply_override_conflict_resolution(self, ordered: list[MBGCard]) -> list[MBGCard]:
        removed: set[str] = set()
        by_id = {c.id: c for c in ordered}
        for card in ordered:
            for dep in card.dependencies:
                if dep.relation == "overrides" and dep.rule_id in by_id:
                    removed.add(dep.rule_id)
                if dep.relation == "incompatible_with" and dep.rule_id in by_id:
                    if card.priority >= by_id[dep.rule_id].priority:
                        removed.add(dep.rule_id)
                    else:
                        removed.add(card.id)
        return [c for c in ordered if c.id not in removed]
