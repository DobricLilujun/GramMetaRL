# P6.1 — Proof-carrying grammar cards: pilot report

Branch `p6-lean-verified-grammar` (separate worktree `GramMetaRL_p6`; the `main` worktree and the
other session's work are untouched). Lean 4.15.0, x86_64-linux, installed from the GitHub release
(the `elan-lang.com` installer does not resolve from this cluster).

Code: `grammetarl/formal/{regex_dfa,card_analysis,cards2lean}.py`, `scripts/lean_pilot.py`.
Tests: `tests/test_p6_formal.py` (7 tests; whole suite 28 passed).
Artifacts in git: `artifacts/lean/{analysis.json,certificates.lean,summary.txt,cases/}`.

## 1. What was built

1. **Finite-alphabet automata semantics for the card-constraint fragment** (`regex_dfa.py`):
   regex → Thompson NFA → subset-construction DFA with emptiness, membership, shortest witness,
   complement, intersection, universality. Unsupported constructs (`(?!`, `(?=`, lookbehind,
   backrefs, `\B`) are *reported*, never silently approximated.
2. **Card-level static analysis** (`card_analysis.py`): per-constraint verdicts
   `ok | dead | vacuous | universal | out_of_fragment | missing_value`; per-card obligation
   automaton `positives ∧ ¬exclusions` with its emptiness and a satisfiability witness; pairwise
   obligation conflicts. `normalize_anchor` centralises the four anchor cases (both/end/start/none)
   so the analysis and the kernel encoding read a pattern the same way.
3. **Lean certificate emitter** (`cards2lean.py`): the automaton itself is emitted
   (`step : Nat → Nat → Nat`, `fin`, `run w = fin (w.foldl step init)`), an explicit per-automaton
   sub-alphabet, and kernel-checked claims:
   * `witness_k` — the kernel *re-executes* the automaton on the DFA's shortest word;
   * `minimal_k` — nothing shorter over that sub-alphabet is accepted (emitted only when the
     Python model already supports it, so Lean is an independent second opinion);
   * `dead_bounded_k` — no string up to depth d is accepted.

## 2. Final numbers (87 cards, 97 constraints, Σ = 94 corpus-derived chars)

| measurement | value |
|---|---|
| regex constraints / all | 68 / 97 |
| inside the formal fragment | 66 (2 rejected: `(?!`) |
| non-regex constraints (`llm_check`/`note`) | 27 |
| constraint verdicts | **ok 63 · dead 4 · universal 1 · out_of_fragment 29** |
| cards with a formalizable obligation | 59 / 87 |
| **provably unsatisfiable cards** (`positives ∧ ¬exclusions = ∅` in Σ*) | **9** |
| pairwise obligation conflicts (1,711 pairs probed) | 596 — see caveat §4 |
| Lean cases emitted / kernel-checked | **12 / 12 pass, 0 fail** (`lean_cases=12 kernel_pass=12`) |
| theorems in `certificates.lean` | 22 (witness + minimality + bounded deadness), over 12 automata |
| T0 (`re.search`, production verifier) vs T2 (automaton) on 1,500 real lb targets | **92.3%** agreement (92,925 / 100,708) |
| confusion (python, dfa) | (F,F) 90,247 · (T,F) 7,750 · (T,T) 2,678 · (F,T) 33 |

## 3. What this establishes for the paper

* **9 of 59 formalizable cards are provably unsatisfiable**: no string in the model can satisfy a
  card's own `positives ∧ ¬exclusions`. Under the current reward every rollout that retrieves one
  of them is charged an unavoidable penalty. This is a proof-shaped claim (bounded model over Σ*,
  unbounded length), not an anecdote, and it is precisely what card admission must reject.
* **`universal = 1` is the mirror pathology**: a `must_include` that can match ε is accepted by
  every string under `re.search` semantics, so it is a constant reward term. Both extremes
  (`dead` = always penalty, `universal` = always pass) are invisible to per-string testing and
  cheap to detect offline.
* **The finite model is conservative relative to the production checker**: disagreement is
  one-sided (7,750 python-only vs 33 dfa-only). As a gate it under-accepts rather than
  over-penalizes, which is the safe direction.
* **Conflict counts carry a caveat that is itself a result.** Many obligations are anchored
  token-level patterns (`^(dat|hatt)$`) evaluated at sentence level; two such obligations cannot
  both hold for one sentence. So the 596 conflicts quantify the *level mis-typing* that
  `RuleConstraint.level` was introduced to fix. Re-counting after obligations are applied at their
  declared level is the next measurement.

## 4. Bugs the kernel and tests caught (this is the argument for kernel checking)

Every one of these silently changed a constraint's *language*; none was visible from the regex
tests or from running the verifier:

1. **Anchor encoding**: stripping `^`/`$` turned the suffix obligation `.*en$` into an infix one.
   Lean rejected 13 witness theorems → fixed by `normalize_anchor` (four cases).
2. **Unreachable start on the fullmatch path**: `NFA()` reserves state 0, and the non-search branch
   was never re-rooted → *every* anchored pattern looked empty (dead count was inflated 4 → 27).
3. **Missing final-state mark** on that same branch → anchored alternations like `^(dat|hatt)$`
   accepted nothing.
4. **Dead-state aliasing**: `dead_id = len(sid)` collided with a real state id, so the kernel's
   automaton differed from Python's.
5. **Naive minimality claims** are false when the pattern matches ε (the depth-0 enumeration
   contains ε) → claims are now pre-validated by the Python model before being handed to Lean.
6. **Harness meta-bug**: `lean file | head` measured `head`'s exit code, so "kernel exit 0" had
   been meaningless. Kernel status is now asserted per case, from lean's own return code.
7. Block-splitting a single generated file by marker cut definitions in half
   (`invalid 'end', insufficient scopes`) → each case is emitted as a self-contained block.

Two data defects surfaced too: a card whose constraint value is the literal string `"null"`, and
one whose value is `nom_form == acc_form` — pseudo-code that passed the prose filter and compiles
into a perfectly legal literal obligation the kernel will happily certify. Certificates do not
substitute for admission: they make garbage *auditable*.

## 5. Rigor ladder as instantiated

| tier | instantiated | guarantee |
|---|---|---|
| T0 | `TranslationVerifier` regex | none |
| T1 | T0-vs-T2 agreement over real pairs | empirical (92.3%) |
| T2 | DFA emptiness / universality / conflicts over Σ* | bounded-model proof (finite Σ, unbounded length), Python |
| T3 | 12 automata, 22 theorems, checked one by one | **kernel-checked**; witnesses unbounded, non-existence bounded to depth d |

## 6. Next steps (concrete)

1. Apply obligations at their declared `level`, then re-count conflicts (expect a large drop).
2. Verified compiler from the card DSL to Lean terms, so a T2 verdict lifts into an unbounded T3
   theorem instead of a depth-bounded one.
3. Conflict minimality: 596 is not a minimal set; compute a small irreducible core.
4. The derivation/CoT format (`apply <card> span reason`), first-failing-step signal, and the
   verified-derivation kernel cases (T3 for traces, sampled).
5. Grow `tests/test_p6_formal.py` with one regression test per bug class listed in §4.
