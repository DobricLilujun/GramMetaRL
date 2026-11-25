r"""Static (automata-based) analysis of compiled grammar-card constraints.

Answers, per card and offline, the questions a string-level verifier cannot:
  * DEAD        the pattern can never match                -> guaranteed false penalty forever
  * VACUOUS     a must_exclude can never fire              -> the "pass" it grants proves nothing
  * UNIVERSAL   a must_include matches everything          -> the reward term is a constant
  * UNSAT       positives /\ not(exclusions) is empty      -> NO translation can satisfy the card
  * CONFLICT    two cards' obligations cannot hold together
Every "no match exists" answer is a bounded-model result over SIGMA*, and the alphabet is
recorded in the output so the claim's strength travels with the number.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from dataclasses import dataclass, field
from pathlib import Path

from .regex_dfa import (DFA, END, START, RegexCompiler, UnsupportedRegex, compile_pattern,
                     normalize_anchor)


@dataclass(slots=True)
class Alphabet:
    chars: tuple[str, ...]

    @property
    def with_sentinels(self) -> tuple[str, ...]:
        return tuple([START] + [c for c in self.chars if c not in (START, END)] + [END])

    def out_of_alphabet(self, text: str) -> set[str]:
        return {c for c in text if c not in self.chars}


def build_alphabet(pair_files: list[Path], max_chars: int = 70) -> Alphabet:
    """Corpus-derived finite alphabet: most frequent characters in the target language side."""
    counts: Counter[str] = Counter()
    for path in pair_files:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 4000:
                    break
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                counts.update(row.get("tgt_ref", ""))
    common = [c for c, _ in counts.most_common(max_chars)]
    for must in " .,;:!?-'\"()0123456789":
        if must not in common:
            common.append(must)
    return Alphabet(tuple(sorted(set(common))))


def _wrap_search(alpha: Alphabet) -> DFA:
    return DFA(compile_pattern("[^\\x00\\x01]*" if False else ".*", alpha.with_sentinels, search=False),
               alpha.with_sentinels)


@dataclass(slots=True)
class ConstraintFact:
    card_id: str
    index: int
    kind: str
    level: str
    pattern: str | None
    status: str                 # ok | dead | vacuous | universal | out_of_fragment | missing_value
    reason: str = ""
    witness: str | None = None
    n_states: int = 0


def analyze_constraint(card_id: str, index: int, cons, alpha: Alphabet,
                       dfa_cache: dict[str, DFA]) -> ConstraintFact:
    kind = getattr(cons, "kind", "?")
    level = getattr(cons, "level", None) or "sentence"
    pattern = getattr(cons, "value", None)
    if kind not in ("must_include_regex", "must_exclude_regex"):
        return ConstraintFact(card_id, index, kind, level, pattern,
                              "out_of_fragment", "not a regex constraint")
    if not pattern:
        return ConstraintFact(card_id, index, kind, level, pattern, "missing_value")
    try:
        # ONE alphabet for every level: mixing alpha.chars and alpha.with_sentinels makes the
        # conflict query (which intersects DFAs from different cards) ill-defined.
        compiler = RegexCompiler(alpha.with_sentinels)
        if level == "token":
            nfa = compiler.level_nfa(pattern, "token")
            dfa = DFA(nfa, alpha.with_sentinels)
        else:
            core, infix = normalize_anchor(pattern)
            nfa = compile_pattern(core, alpha.with_sentinels, search=infix)
            dfa = DFA(nfa, alpha.with_sentinels)
    except UnsupportedRegex as exc:
        return ConstraintFact(card_id, index, kind, level, pattern, "out_of_fragment", exc.reason)
    dfa_cache[f"{card_id}#{index}"] = dfa
    empty = dfa.is_empty()
    if kind == "must_exclude_regex" and empty:
        return ConstraintFact(card_id, index, kind, level, pattern, "vacuous",
                              "pattern matches nothing in SIGMA*", None, len(dfa.states))
    if kind == "must_include_regex" and empty:
        return ConstraintFact(card_id, index, kind, level, pattern, "dead",
                              "obligation unsatisfiable in SIGMA*", None, len(dfa.states))
    if not dfa.overflow and dfa.universal():
        return ConstraintFact(card_id, index, kind, level, pattern, "universal",
                              "matches every string in SIGMA*; carries no signal",
                              None, len(dfa.states))
    return ConstraintFact(card_id, index, kind, level, pattern, "ok",
                          "", dfa.witness(), len(dfa.states))


def _conjunction(dfas: list[DFA], cap_note: list[str]) -> DFA | None:
    if not dfas:
        return None
    acc = dfas[0]
    for d in dfas[1:]:
        acc = acc.intersect(d)
        if acc.overflow:
            cap_note.append("state-cap")
            return acc
    return acc


def obligation_dfa(card, alpha: Alphabet, cache: dict[str, DFA], note: list[str]) -> DFA | None:
    parts: list[DFA] = []
    for i, cons in enumerate(card.output_constraints):
        d = cache.get(f"{card.id}#{i}")
        if d is None:
            continue
        if getattr(cons, "kind", "") == "must_exclude_regex":
            d = d.complement()
        parts.append(d)
    return _conjunction(parts, note)


def analyze_cards(cards, alpha: Alphabet) -> dict:
    cache: dict[str, DFA] = {}
    facts: list[ConstraintFact] = []
    obligations: dict[str, DFA] = {}
    notes: list[str] = []
    for card in cards:
        for i, cons in enumerate(card.output_constraints):
            facts.append(analyze_constraint(card.id, i, cons, alpha, cache))
        # Applicability predicates get the same treatment: a DEAD predicate means the rule can
        # never be in play (silently inert card); a UNIVERSAL one means applicability was never
        # actually restricted, i.e. abstention is a no-op and the old noise returns.
        for i, pred in enumerate(getattr(card, "applicability", []) or []):
            if getattr(pred, "kind", "") != "regex" or not getattr(pred, "pattern", None):
                continue
            pseudo = type("C", (), {})()
            pseudo.kind = "must_include_regex"
            pseudo.value = pred.pattern
            pseudo.level = getattr(pred, "level", "sentence")
            f = analyze_constraint(card.id, 1000 + i, pseudo, alpha, cache)
            f.kind = "applicability_regex"
            facts.append(f)
        ob = obligation_dfa(card, alpha, cache, notes)
        if ob is not None:
            obligations[card.id] = ob

    by_status = Counter(f.status for f in facts)
    dead_cards = sorted({f.card_id for f in facts if f.status == "dead"})
    vacuous_cards = sorted({f.card_id for f in facts if f.status == "vacuous"})
    universal_cards = sorted({f.card_id for f in facts if f.status == "universal"})
    out_of_fragment = sorted({f.card_id for f in facts if f.status == "out_of_fragment"})

    unsat: list[dict] = []
    for cid, ob in obligations.items():
        if ob.is_empty():
            unsat.append({"card_id": cid, "why": "positives /\ not(exclusions) = empty in SIGMA*"})
    sat_witness = {cid: ob.witness() for cid, ob in obligations.items() if not ob.is_empty()}

    # pairwise conflicts, bounded to formalizable cards
    ids = sorted(obligations)
    conflicts: list[dict] = []
    budget = 4000
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if budget <= 0:
                break
            budget -= 1
            if obligations[ids[i]].is_empty() or obligations[ids[j]].is_empty():
                continue  # an unsatisfiable card conflicts with everything: reported separately
            inter = obligations[ids[i]].intersect(obligations[ids[j]])
            if not inter.overflow and inter.is_empty():
                conflicts.append({"cards": [ids[i], ids[j]],
                                  "why": "obligations cannot hold together in SIGMA*"})
        if budget <= 0:
            break

    reasons = Counter(f.reason for f in facts if f.status == "out_of_fragment")
    return {
        "alphabet_size": len(alpha.with_sentinels),
        "alphabet": "".join(alpha.with_sentinels).replace(START, "<^>").replace(END, "<$>"),
        "constraints": len(facts),
        "by_status": dict(by_status),
        "out_of_fragment_reasons": dict(reasons),
        "cards_dead": dead_cards,
        "cards_vacuous": vacuous_cards,
        "cards_universal": universal_cards,
        "cards_out_of_fragment": out_of_fragment,
        "cards_with_obligation": len(obligations),
        "unsatisfiable_cards": unsat,
        "satisfiable_witnesses": sat_witness,
        "pair_conflicts": conflicts,
        "pair_budget_used": 4000 - budget,
        "notes": notes,
        "facts": [asdict(f) for f in facts],
    }
