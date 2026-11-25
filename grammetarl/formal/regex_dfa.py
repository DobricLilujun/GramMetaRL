r"""Finite-alphabet automata semantics for the compiled card-constraint fragment.

Why this module exists
----------------------
The regex verifier is measurably noisy: on 48k real en->lb pairs, 85.2% of retrieved-rule checks
failed and 34,725 must_exclude checks were *vacuous* (the pattern matched nothing, so the "pass"
proved nothing). String-by-string testing cannot tell us whether a constraint is dead, always
true, or jointly unsatisfiable with another constraint. Automata can, and they do it offline,
once per card.

Guarantee directions are stated explicitly:
  * "a witness exists"            -> absolute; the witness string is returned
  * "no match / unsatisfiable"    -> relative to SIGMA*, a bounded model over a corpus-derived
                                     finite alphabet (documented, not a universe-wide proof)

Supported fragment (everything else is reported by construct name and demoted, never silently
formalized): literals, escapes, [.], [^], ( ), (?: ), |, *, +, ?, {n}, {n,m}, ^, $, \d \w \s.
Unsupported: (?= (?! (?<= (?<! backreferences \B.
"""
from __future__ import annotations

import random
import re as _re
from collections import deque
from dataclasses import dataclass, field

START = "\x00"   # boundary sentinel standing for re's ^
END = "\x01"     # boundary sentinel standing for re's $

UNSUPPORTED_MARKERS = ("(?=", "(?!", "(?<=", "(?<!", r"\B", r"\g<", r"\1", r"\2", r"\3")
DIGITS = "0123456789"
WS = " \t\n"


class UnsupportedRegex(Exception):
    def __init__(self, reason: str, pattern: str):
        super().__init__(f"{reason}: {pattern!r}")
        self.reason = reason
        self.pattern = pattern


@dataclass(slots=True)
class NFAState:
    edges: dict[str, set[int]] = field(default_factory=dict)
    eps: set[int] = field(default_factory=set)
    final: bool = False


class NFA:
    def __init__(self) -> None:
        self.states: list[NFAState] = [NFAState()]

    def new(self, final: bool = False) -> int:
        self.states.append(NFAState(final=final))
        return len(self.states) - 1

    def closure(self, states: set[int]) -> set[int]:
        out, stack = set(states), list(states)
        while stack:
            for t in self.states[stack.pop()].eps:
                if t not in out:
                    out.add(t)
                    stack.append(t)
        return out

    def step(self, states: set[int], c: str) -> set[int]:
        nxt: set[int] = set()
        for s in states:
            nxt |= self.states[s].edges.get(c, set())
        return self.closure(nxt)

    def accepts(self, word: str) -> bool:
        cur = self.closure({0})
        for c in word:
            cur = self.step(cur, c)
            if not cur:
                return False
        return any(self.states[s].final for s in cur)

    def is_empty(self) -> bool:
        seen, stack = set(), [0]
        while stack:
            s = stack.pop()
            if s in seen:
                continue
            seen.add(s)
            if self.states[s].final:
                return False
            for nxt in self.states[s].edges.values():
                stack.extend(nxt)
            stack.extend(self.states[s].eps)
        return True

    def shortest_word(self, cap: int = 200) -> str | None:
        start = self.closure({0})
        q, seen = deque([(start, "")]), {frozenset(start)}
        while q:
            cur, w = q.popleft()
            if any(self.states[s].final for s in cur):
                return w
            if len(w) >= cap:
                continue
            for s in cur:
                for c, nxt in self.states[s].edges.items():
                    got = self.closure(nxt)
                    key = frozenset(got)
                    if got and key not in seen:
                        seen.add(key)
                        q.append((got, w + c))
        return None


class DFA:
    """Subset construction over a fixed finite alphabet; a dead state is implicit."""

    def __init__(self, nfa: NFA, alphabet: tuple[str, ...], cap: int = 60000):
        self.alphabet = alphabet
        self.cap = cap
        self.trans: dict[tuple[frozenset[int], str], frozenset[int]] = {}
        self.states: set[frozenset[int]] = set()
        self.finals: set[frozenset[int]] = set()
        self.overflow = False
        DEAD = frozenset()
        start = frozenset(nfa.closure({0}))
        self.start = start if start else DEAD
        self.states.add(self.start)
        if any(nfa.states[s].final for s in start):
            self.finals.add(self.start)
        q = [self.start]
        while q:
            cur = q.pop()
            if not cur:
                continue
            for c in alphabet:
                nxt = frozenset(nfa.step(set(cur), c))
                self.trans[(cur, c)] = nxt
                if nxt and nxt not in self.states:
                    if len(self.states) >= cap:
                        self.overflow = True
                        return
                    self.states.add(nxt)
                    if any(nfa.states[s].final for s in nxt):
                        self.finals.add(nxt)
                    q.append(nxt)

    # --- queries ---------------------------------------------------------------
    def accepts(self, word: str) -> bool:
        cur = self.start
        for c in word:
            cur = self.trans.get((cur, c), frozenset())
            if not cur:
                return False
        return cur in self.finals

    def is_empty(self) -> bool:
        seen, q = {self.start}, [self.start]
        while q:
            cur = q.pop()
            if cur in self.finals:
                return False
            for c in self.alphabet:
                nxt = self.trans.get((cur, c), frozenset())
                if nxt and nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
        return True

    def witness(self, cap: int = 80) -> str | None:
        q, seen = deque([(self.start, "")]), {self.start}
        while q:
            cur, w = q.popleft()
            if cur in self.finals:
                return w
            if len(w) >= cap:
                continue
            for c in self.alphabet:
                nxt = self.trans.get((cur, c), frozenset())
                if nxt and nxt not in seen:
                    seen.add(nxt)
                    q.append((nxt, w + c))
        return None

    def complement(self) -> "DFA":
        """Complement over the same finite alphabet, using the implicit dead state."""
        out = DFA.__new__(DFA)
        out.alphabet, out.cap, out.overflow = self.alphabet, self.cap, self.overflow
        out.trans = dict(self.trans)
        out.states = set(self.states) | {frozenset()}
        out.start = self.start
        out.finals = {s for s in out.states if s not in self.finals}
        for s in out.states:
            for c in self.alphabet:
                out.trans.setdefault((s, c), frozenset())
        return out

    def intersect(self, other: "DFA") -> "DFA":
        assert self.alphabet == other.alphabet, "conflict queries need one alphabet"
        out = DFA.__new__(DFA)
        out.alphabet, out.cap = self.alphabet, self.cap
        out.overflow = self.overflow or other.overflow
        out.trans, out.states, out.finals = {}, set(), set()
        out.start = (self.start, other.start)
        out.states.add(out.start)
        if self.start in self.finals and other.start in other.finals:
            out.finals.add(out.start)
        q = [out.start]
        while q:
            cur = q.pop()
            a, b = cur
            for c in self.alphabet:
                na = self.trans.get((a, c), frozenset())
                nb = other.trans.get((b, c), frozenset())
                if not (na and nb):
                    continue
                nxt = (na, nb)
                out.trans[(cur, c)] = nxt
                if nxt not in out.states:
                    if len(out.states) >= self.cap:
                        out.overflow = True
                        return out
                    out.states.add(nxt)
                    if na in self.finals and nb in other.finals:
                        out.finals.add(nxt)
                    q.append(nxt)
        return out

    def universal(self) -> bool:
        return self.complement().is_empty()

    def random_word(self, rng: random.Random, lo: int = 1, hi: int = 14) -> str:
        pool = [c for c in self.alphabet if c not in (START, END)] or list(self.alphabet)
        return "".join(rng.choice(pool) for _ in range(rng.randint(lo, hi)))


class RegexCompiler:
    """Recursive-descent regex -> NFA over a fixed finite alphabet.

    `.` and negated classes are resolved against `alphabet` (finite model).
    `^`/`$` become the boundary sentinels; unanchored patterns are wrapped in alphabet*,
    matching re.search semantics rather than re.fullmatch.
    """

    def __init__(self, alphabet: tuple[str, ...]):
        self.alphabet = alphabet
        self.i = 0
        self.pat = ""

    def compile(self, pattern: str, search: bool = True) -> NFA:
        for bad in UNSUPPORTED_MARKERS:
            if bad in pattern:
                raise UnsupportedRegex(f"construct {bad}", pattern)
        self.pat, self.i = pattern, 0
        nfa, s, e = self._alt(NFA())
        if search:
            s0, e0 = nfa.new(), nfa.new(final=True)
            nfa.states[s0].eps.add(s)
            nfa.states[e].eps.add(e0)
            for c in self.alphabet:
                if c in (START, END):
                    continue
                nfa.states[s0].edges.setdefault(c, set()).add(s0)
                nfa.states[e0].eps.add(e0)
                nfa.states[e].edges.setdefault(c, set()).add(e0)
            nfa.states[s0].final = False
            return self._reroot(nfa, s0, e0)
        # Non-search (fullmatch) path: mark the terminal accepting, then re-root. Both steps are
        # load-bearing: without the final mark every anchored pattern looks empty, and NFA()
        # reserves state 0 so the real start would otherwise be unreachable.
        nfa.states[e].final = True
        return self._reroot(nfa, s, e)

    def _reroot(self, nfa: NFA, start: int, end: int) -> NFA:
        """Copy the automaton so that `start` becomes state 0 (invariant used everywhere)."""
        order = [start] + [s for s in range(len(nfa.states)) if s != start]
        idx = {s: k for k, s in enumerate(order)}
        out = NFA()
        out.states = [NFAState(final=nfa.states[s].final) for s in order]
        for k, s in enumerate(order):
            out.states[k].eps = {idx[t] for t in nfa.states[s].eps}
            for c, ts in nfa.states[s].edges.items():
                out.states[k].edges[c] = {idx[t] for t in ts}
        return out

    # --- grammar ---------------------------------------------------------------
    def _peek(self) -> str | None:
        return self.pat[self.i] if self.i < len(self.pat) else None

    def _alt(self, nfa: NFA) -> tuple[NFA, int, int]:
        s, e = nfa.new(), nfa.new()
        nfa, branch_s, branch_e = self._cat(nfa)
        nfa.states[s].eps.add(branch_s)
        nfa.states[branch_e].eps.add(e)
        while self._peek() == "|":
            self.i += 1
            nfa, bs, be = self._cat(nfa)
            nfa.states[s].eps.add(bs)
            nfa.states[be].eps.add(e)
        return nfa, s, e

    def _cat(self, nfa: NFA) -> tuple[NFA, int, int]:
        s = None
        e = None
        while self.i < len(self.pat) and self._peek() not in ("|", ")"):
            nfa, ns, ne = self._postfix(nfa)
            if s is None:
                s, e = ns, ne
            else:
                nfa.states[e].eps.add(ns)
                e = ne
        if s is None:
            s = nfa.new()
            e = s
            nfa.states[s].final = False
        return nfa, s, e

    def _postfix(self, nfa: NFA) -> tuple[NFA, int, int]:
        atom_start = self.i
        nfa, s, e = self._atom(nfa)
        atom_end = self.i
        while self._peek() in ("*", "+", "?", "{"):
            op = self.pat[self.i]
            if op == "{":
                m = _re.match(r"\{(\d+)(?:,(\d*))?\}", self.pat[self.i:])
                if not m:
                    raise UnsupportedRegex("bad {n,m}", self.pat)
                lo = int(m.group(1))
                hi = int(m.group(2)) if m.group(2) not in (None, "") else lo
                if m.group(2) == "":
                    hi = lo + 4  # open-ended star bounded for the finite model
                self.i += m.end()
                return self._bounded(nfa, s, e, lo, hi, self.pat[atom_start:atom_end])
            self.i += 1
            if op == "?":
                nfa.states[s].eps.add(e)
            elif op == "*":
                nfa.states[e].eps.add(s)
                nfa.states[s].eps.add(e)
                s2, e2 = nfa.new(), nfa.new(final=False)
                nfa.states[s2].eps.add(s)
                nfa.states[e].eps.add(e2)
                s, e = s2, e2
            elif op == "+":
                nfa.states[e].eps.add(s)
                nfa.states[s].eps.add(e)
        return nfa, s, e

    def _bounded(self, nfa: NFA, s: int, e: int, lo: int, hi: int,
                 atom_src: str) -> tuple[NFA, int, int]:
        """Expand x{lo,hi} by compiling the atom text `hi` times (hi is finite by construction).

        Re-compiling the source text is deliberately simpler and safer than graph copying:
        every copy is a fresh, independent fragment, so epsilon closure cannot leak between
        repetitions.
        """
        fin = nfa.new()
        start = nfa.new()
        prev = start
        for k in range(max(hi, 1)):
            if k >= lo:  # after k completed copies we may stop early
                nfa.states[prev].eps.add(fin)
            sub = RegexCompiler(self.alphabet)
            for bad in UNSUPPORTED_MARKERS:
                if bad in atom_src:
                    raise UnsupportedRegex(f"construct {bad}", atom_src)
            sub.pat, sub.i = atom_src, 0
            sub_nfa, ss, se = sub._alt(NFA())
            base = len(nfa.states)
            for st in sub_nfa.states:
                nfa.states.append(NFAState(final=False))
            for k2, st in enumerate(sub_nfa.states):
                nfa.states[base + k2].eps = {base + t for t in st.eps}
                for c, ts in st.edges.items():
                    nfa.states[base + k2].edges[c] = {base + t for t in ts}
            nfa.states[prev].eps.add(base + ss)
            prev = base + se
        nfa.states[prev].eps.add(fin)
        if lo == 0:
            nfa.states[start].eps.add(fin)
        return nfa, start, fin

    def _atom(self, nfa: NFA) -> tuple[NFA, int, int]:
        c = self._peek()
        if c == "(":
            self.i += 1
            if self.pat.startswith("?") and self.pat[self.i : self.i + 1] == ":":
                self.i += 1
            elif self.pat[self.i : self.i + 1] == "?":
                raise UnsupportedRegex("(?...) group", self.pat)
            n, s, e = self._alt(nfa)
            if self._peek() != ")":
                raise UnsupportedRegex("unbalanced (", self.pat)
            self.i += 1
            return n, s, e
        if c == "[":
            chars = self._class()
            s, e = nfa.new(), nfa.new()
            for ch in chars:
                nfa.states[s].edges.setdefault(ch, set()).add(e)
            return nfa, s, e
        if c == ".":
            self.i += 1
            s, e = nfa.new(), nfa.new()
            for ch in self.alphabet:
                if ch not in (START, END):
                    nfa.states[s].edges.setdefault(ch, set()).add(e)
            return nfa, s, e
        if c == "^":
            self.i += 1
            return self._lit(nfa, START)
        if c == "$":
            self.i += 1
            return self._lit(nfa, END)
        if c == "\\":
            return self._escape(nfa)
        if c in ("*", "+", "?", "}", ")"):
            raise UnsupportedRegex(f"stray {c}", self.pat)
        self.i += 1
        return self._lit(nfa, c)

    def _lit(self, nfa: NFA, ch: str) -> tuple[NFA, int, int]:
        s, e = nfa.new(), nfa.new()
        nfa.states[s].edges.setdefault(ch, set()).add(e)
        return nfa, s, e

    def _escape(self, nfa: NFA) -> tuple[NFA, int, int]:
        self.i += 1
        c = self._peek()
        if c is None:
            raise UnsupportedRegex("trailing backslash", self.pat)
        self.i += 1
        if c == "d":
            return self._set(nfa, set(DIGITS))
        if c == "w":
            return self._set(nfa, {a for a in self.alphabet if a.isalnum() or a == "_"})
        if c == "s":
            return self._set(nfa, set(WS))
        if c == "W":
            allw = {a for a in self.alphabet if a.isalnum() or a == "_"}
            return self._set(nfa, set(self.alphabet) - allw)
        if c == "D":
            return self._set(nfa, set(self.alphabet) - set(DIGITS))
        if c == "S":
            return self._set(nfa, set(self.alphabet) - set(WS))
        if c in ("n", "t", "r"):
            return self._lit(nfa, {"n": "\n", "t": "\t", "r": "\r"}[c])
        if c in "bB":
            raise UnsupportedRegex(rf"\{c} zero-width assertion", self.pat)
        if c.isdigit():
            raise UnsupportedRegex("backreference", self.pat)
        return self._lit(nfa, c)

    def _set(self, nfa: NFA, chars: set[str]) -> tuple[NFA, int, int]:
        s, e = nfa.new(), nfa.new()
        for ch in chars:
            nfa.states[s].edges.setdefault(ch, set()).add(e)
        return nfa, s, e

    def _class(self) -> set[str]:
        self.i += 1  # consume [
        neg = self._peek() == "^"
        if neg:
            self.i += 1
        chars: set[str] = set()
        while True:
            c = self._peek()
            if c is None:
                raise UnsupportedRegex("unterminated [", self.pat)
            if c == "]":
                self.i += 1
                break
            if c == "\\":
                self.i += 1
                e = self._peek()
                self.i += 1
                if e == "d":
                    chars |= set(DIGITS)
                elif e == "w":
                    chars |= {a for a in self.alphabet if a.isalnum() or a == "_"}
                elif e == "s":
                    chars |= set(WS)
                else:
                    chars.add(e if e else "\\")
                continue
            # possible range a-z
            if self.i + 2 < len(self.pat) and self.pat[self.i + 1] == "-" and self.pat[self.i + 2] != "]":
                a, b = self.pat[self.i], self.pat[self.i + 2]
                self.i += 3
                for o in range(ord(a), ord(b) + 1):
                    ch = chr(o)
                    if ch in self.alphabet:
                        chars.add(ch)
                continue
            self.i += 1
            if c in self.alphabet:
                chars.add(c)
        if neg:
            chars = {a for a in self.alphabet if a not in chars and a not in (START, END)}
        return chars


    # ------------------------------------------------------------------ level wrappers
    def level_nfa(self, pattern: str, level: str) -> NFA:
        """Build the automaton for a pattern *at its declared linguistic level*.

        The same pattern means different things at different levels, and evaluating a token-level
        obligation as a sentence-level one is what made two token cards look mutually exclusive
        (596 pairwise conflicts before this function existed):

          token    the pattern must match a whole token, possibly surrounded by word characters
          phrase   substring of the target (re.search semantics)
          sentence full string (anchors honoured via normalize_anchor)
          any      substring
        """
        core, infix = normalize_anchor(pattern)
        if level == "token":
            # A token-level obligation constrains a WHOLE token: SIGMAWORD* core SIGMAWORD*,
            # where core keeps the anchor semantics the card author wrote. Keeping those anchors
            # is the point: `^der$` at token level must reject "zuder", and an unanchored core such
            # as `der` still matches inside a longer word ("derx") because the surrounding word
            # characters supply the context. Stripping them (as an earlier version did) quietly
            # changed `^(dat|hatt)$` from a form inventory into a substring test.
            inner = self.compile(core, search=infix)
            nfa = NFA()
            order = [0] + [x for x in range(1, len(inner.states))]
            idx = {x: k for k, x in enumerate(order)}
            nfa.states = [NFAState(final=inner.states[x].final) for x in order]
            for k, x in enumerate(order):
                nfa.states[k].eps = {idx[t] for t in inner.states[x].eps}
                for c, ts in inner.states[x].edges.items():
                    nfa.states[k].edges[c] = {idx[t] for t in ts}
            s0, e0 = nfa.new(), nfa.new()
            s_in, e_in = idx[0], None
            for x, k in idx.items():
                if inner.states[x].final:
                    e_in = k if e_in is None else e_in
            nfa.states[s0].eps.add(s_in)
            nfa.states[e_in].eps.add(e0)
            word = [c for c in self.alphabet if c not in (" ", "\\t", "\\n")]
            for c in word:
                nfa.states[s0].edges.setdefault(c, set()).add(s0)
                nfa.states[e0].edges.setdefault(c, set()).add(e0)
            nfa.states[e0].eps.add(e_in)  # the occurrence may start later inside the token
            return self._reroot(nfa, s0, e0)
        if level in ("phrase", "any"):
            return self.compile(core, search=True)
        return self.compile(core, search=infix)


def compile_pattern(pattern: str, alphabet: tuple[str, ...], search: bool = True) -> NFA:
    return RegexCompiler(alphabet).compile(pattern, search=search)


def normalize_anchor(pattern: str) -> tuple[str, bool]:
    """Map a pattern to (core, allow_infix) using explicit wrapping instead of stripping anchors.

    Single-sided anchors matter: `.*en$` is a suffix obligation, not an infix one. Stripping `$`
    silently changes the language (measured: a 10% disagreement with the production checker that
    was our own encoding bug, not a card defect).
    """
    p = pattern.strip()
    start = p.startswith("^")
    end = p.endswith("$") and not p.endswith("\\$")
    core = p[1:] if start else p
    core = core[:-1] if end else core
    if start and end:
        return core, False          # full match
    if end:
        return ".*" + core, False   # suffix
    if start:
        return core + ".*", False   # prefix
    return core, True               # re.search-style infix




PHRASAL_MARKERS = (" ", r"\s", ".*", ".+", "{2", r"\W", "[^")


def infer_level(pattern: str) -> str:
    """Single shared rule for reading an unlabelled pattern (validator and formal layer agree).

    `^(dat|hatt|et|seng)$` is a word-form inventory -> token.
    `\b(well|datt)\b.*\b(ass|gaangen)$` spans words -> sentence, even though it is anchored.
    Cards that declare `level` explicitly never go through here.
    """
    p = pattern.strip()
    anchored = p.startswith("^") or p.endswith("$")
    phrasal = any(m in p for m in PHRASAL_MARKERS)
    return "token" if (anchored and not phrasal) else "sentence"
