# P6.1 — Proof-carrying grammar cards: pilot report (Lean 4.15.0, 87 canonical cards)

Branch `p6-lean-verified-grammar` (worktree `GramMetaRL_p6`, main worktree untouched).
Code: `grammetarl/formal/{regex_dfa,card_analysis,cards2lean}.py`, `scripts/lean_pilot.py`.
Artifacts (in git): `artifacts/lean/{analysis.json,certificates.lean,summary.txt}`.
Tests: `tests/test_p6_formal.py`.

## 1. What was built

1. **A finite-alphabet automata semantics for the card-constraint fragment** (`regex_dfa.py`):
   regex -> Thompson NFA -> subset-construction DFA, with emptiness, membership, shortest witness,
   complement, intersection, universality, and an explicit *unsupported-construct* report
   (`(?!`, `(?=`, lookbehind, backrefs, `\B`). Unsupported never means "guessed": the constraint is
   labelled `out_of_fragment` with the reason.
2. **Card-level static analysis** (`card_analysis.py`): per constraint
   `ok | dead | vacuous | universal | out_of_fragment | missing_value`; per card the obligation
   automaton `positives ∧ ¬exclusions`, its emptiness, and a satisfiability witness; pairwise
   obligation conflicts among formalizable cards.
3. **A Lean emitter producing genuine certificates** (`cards2lean.py`): the automaton itself is
   emitted (`step : Nat -> Nat -> Nat`, `fin`, `run = fin (w.foldl step init)`), plus per automaton
   sub-alphabet and a bounded string enumerator, then kernel-checked theorems:
   * `witness_<k>` — the kernel *re-executes* the automaton on the DFA's shortest word;
   * `minimal_<k>` — nothing shorter than that word is accepted (minimality of the witness);
   * `dead_bounded_<k>` — no string up to depth d over the card's own sub-alphabet is accepted.

## 2. Headline numbers (87 cards, 97 constraints, Σ = 94 corpus-derived chars)

| measurement | value |
|---|---|
| regex constraints / total | 68 / 97 |
| inside the formal fragment | 66 (2 rejected for `(?!`) |
| constraint status after analysis | ok 36, **dead 27**, vacuous exclusions 5, universal 0 |
| cards with a formalizable obligation | 59 / 87 |
| **provably unsatisfiable cards** (`positives ∧ ¬exclusions = ∅` in Σ*) | **24 / 59** |
| pairwise obligation conflicts (1,711 pairs probed) | **320** |
| automata + theorems emitted to Lean | 25 automata, 30 theorems (7 witness, 5 minimal, 18 dead_bounded) |
| Lean 4.15.0 kernel result | **exit 0, no errors — all 30 theorems checked** |
| T0 (`re.search`, production) vs T2 (automaton) on 1,500 real lb targets | **90.8% agreement** (91,455 / 100,708) |
| confusion (python, dfa) | (F,F) 90,247 · (T,F) 9,220 · (T,T) 1,208 · (F,T) 33 |

## 3. What the numbers mean for the paper

* **The strongest result: 24 of 59 formalizable cards can never be satisfied by any string in the
  model.** In the current reward those cards are pure noise: every rollout that retrieves one is
  charged a penalty it cannot avoid. This is a *proof-flavoured* claim (bounded model over Σ*), not
  an anecdote, and it is exactly what P1's "card self-test admission" is supposed to prevent.
* **Dead/vacuous counts are the same phenomenon measured offline.** The earlier run on 48k pairs
  counted 34,725 vacuous exclusion *checks*; here 27 dead patterns + 5 vacuous exclusions are
  counted once per card, i.e. the cause rather than the symptom.
* **320 pairwise conflicts must be read with its caveat.** Many obligations are anchored token-level
  patterns (`^(dat|hatt)$`) evaluated at sentence level; two such obligations are mutually exclusive
  as sentences. So the conflict count is itself evidence of the *level mis-typing* that
  `RuleConstraint.level` was added to fix — the algebra turns a design argument into a count.
* **T0/T2 agreement 90.8% is a conservative mismatch, not a coin flip.** Disagreement is one-sided:
  9,220 cases where `re.search` matches and the finite model does not, versus 33 the other way.
  The dominant cause is literals outside Σ (pattern symbols absent from the corpus alphabet) and
  class/`.` resolution against Σ. Direction matters: the model errs by *rejecting*, so as a gate it
  under-accepts rather than over-penalizes. Remaining causes are itemised in
  `artifacts/lean/analysis.json`.

## 4. Two bugs the kernel caught (these are the argument for kernel checking)

1. **Anchor encoding.** My first emitter stripped `^`/`$` and lost the suffix reading of `.*en$`
   (infix vs suffix), so Lean rejected 13 witness theorems ("decide proved the proposition false").
   The convention is now centralised in `normalize_anchor` (four cases: both/end/start/none) and
   used by both the analysis and the emitter; the fix moved the status distribution
   (ok 64 → 36, dead 4 → 27), i.e. the previous "ok" count had been inflated by mis-typed anchors.
2. **A card whose constraint value is the literal string `"null"`.** It compiles into a perfectly
   legal word obligation and the kernel happily certificates it. Certificates do not fix garbage
   input: this is why the schema/type layer plus self-test admission stays part of P1.

## 5. Rigor ladder, as actually instantiated

| tier | instantiated here | guarantee |
|---|---|---|
| T0 | `TranslationVerifier` regex | none |
| T1 | T0-vs-T2 agreement measurement over real pairs | empirical, 90.8% |
| T2 | DFA emptiness/universality/conflict over Σ* (unbounded length, finite Σ) | bounded-model proof, Python |
| T3 | 30 Lean theorems over 25 automata | **kernel-checked**, but witnesses unbounded and non-existence bounded to depth d |

The gap between T2 and T3 is explicit in the emitted file (`-- BOUNDED-MODEL BOUND ...`), and
closing it (a verified compiler from the card DSL to Lean terms, so a T2 verdict can be lifted into
an unbounded T3 theorem) is the next concrete step, together with:
* conflict-minimality (320 conflicts is not a minimal set),
* applying obligations at their declared `level` before counting conflicts,
* the verified derivation/CoT format (`apply <card> span reason`) and the first-failing-step signal,
* `tests/` growth: 5 regression tests now, anchor cases first.
