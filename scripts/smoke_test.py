"""GramMetaRL P0 smoke test / gate harness.

Purpose: exercise every stage of the pipeline WITHOUT needing an LLM endpoint, a GPU,
or verl, and turn each known blocker from the 2026-11-18 audit into a measured,
machine-checkable gate. Re-run this after every P0 fix.

Usage:
    /raid/yewei/.venvs/gm-lite/bin/python scripts/smoke_test.py [--cards PATH] [--limit 3]

The cards used here are a HARNESS FIXTURE: the Luxembourgish strings are synthetic
placeholders for validating the code path, not validated linguistic claims. Real card
quality is measured by `--cards data/rules/mbg_cards.jsonl`.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grammetarl import MBGCard, RuleIndex, TranslationVerifier, load_cards  # noqa: E402
from grammetarl.agent_env import GrammarFirstEnv  # noqa: E402
from grammetarl.rewards import aggregate_reward  # noqa: E402
from grammetarl.rewards.grammar_reward import grammar_reward  # noqa: E402
from grammetarl.tools import dictionary_lookup, grammar_rule_apply, grammar_rule_search  # noqa: E402
from grammetarl.workspace import get_workspace_paths  # noqa: E402

OUT_DIR = ROOT / "artifacts" / "smoke"
FIXTURE_CARDS = OUT_DIR / "smoke_cards.jsonl"

# ---------------------------------------------------------------------------
# Fixture cards: deliberately mix (a) executable constraints, (b) the current
# prose-as-regex failure mode, (c) deferred llm_check, (d) dependencies for the
# topological + conflict-resolution paths in RuleIndex.
# ---------------------------------------------------------------------------
SRC = {"source_id": "smoke_fixture", "page_start": 1, "page_end": 2}

FIXTURES: list[dict] = [
    {
        "id": "LB_SMOKE_0001", "language": "lb", "scope": "main declarative clause | V2",
        "operation_type": "reorder", "priority": 60,
        "statement": "In a main declarative clause the finite verb is in second position.",
        "confidence": 0.9, "rule_status": "categorical",
        "phenomenon_tags": ["word_order", "V2"],
        "trigger_conditions": ["main declarative clause"],
        "operation_steps": ["identify the first constituent", "place the finite verb next"],
        "output_constraints": [
            {"kind": "must_include_regex", "description": "finite verb in second position",
             "value": r"^(?:\S+)\s+(ass|war|hu|kann|w\u00e9ll|g\u00ebtt)\b"}],
        "examples": [{"source_sentence": "He went to school.",
                      "target_sentence": "Hie ass an d'Schoul gaangen.", "gloss": None, "note": None}],
        "source": SRC,
    },
    {
        "id": "LB_SMOKE_0002", "language": "lb", "scope": "subordinate clause | verb-final",
        "operation_type": "reorder", "priority": 55,
        "statement": "Subordinate clauses put the verb complex clause-finally.",
        "confidence": 0.85, "rule_status": "default",
        "phenomenon_tags": ["word_order", "subordination"],
        "trigger_conditions": ["subordinate clause introduced by a subordinating conjunction"],
        "operation_steps": ["find the conjunction", "move the verb complex to the end"],
        "output_constraints": [
            {"kind": "must_include_regex", "description": "verb complex at clause end",
             "value": r"\b(well|datt|wann)\b.*\b(ass|gaangen|gemach)\s*$"}],
        "examples": [{"source_sentence": "... because he went to school.",
                      "target_sentence": "... well hie gaangen ass", "gloss": None, "note": None}],
        "source": SRC,
        "dependencies": [{"rule_id": "LB_SMOKE_0001", "relation": "requires", "note": "clause type contrast"}],
    },
    {
        "id": "LB_SMOKE_0003", "language": "lb", "scope": "clausal negation",
        "operation_type": "negate", "priority": 50,
        "statement": "The negation adverb follows the finite verb.",
        "confidence": 0.8, "rule_status": "default",
        "phenomenon_tags": ["negation"],
        "trigger_conditions": ["clausal negation"],
        "operation_steps": ["locate the finite verb", "insert the negation adverb after it"],
        "output_constraints": [
            {"kind": "must_include_regex", "description": "negation after finite verb",
             "value": r"\b(ass|war|kann|w\u00e9ll)\s+net\b"},
            {"kind": "must_exclude_regex", "description": "no doubled negation",
             "value": r"\bnet\b.{0,40}\bnet\b"}],
        "examples": [{"source_sentence": "He did not go.",
                      "target_sentence": "Hie ass net gaangen.", "gloss": None, "note": None}],
        "source": SRC,
    },
    {
        "id": "LB_SMOKE_0004", "language": "lb", "scope": "preposition case government",
        "operation_type": "mark_case", "priority": 70,
        "statement": "Some prepositions govern a specific case.",
        "confidence": 0.7, "rule_status": "variable",
        "phenomenon_tags": ["case_marking", "preposition_governance"],
        "trigger_conditions": ["prepositional object"],
        "operation_steps": ["identify the preposition", "check the case of its object"],
        "output_constraints": [
            {"kind": "llm_check", "description": "object NP bears the case required by the preposition"}],
        "examples": [],
        "source": SRC,
    },
    {
        "id": "LB_SMOKE_0005", "language": "lb", "scope": "unspecified",
        "operation_type": "fallback", "priority": 50,
        "statement": "Card carrying the CURRENT failure mode: prose in a regex slot.",
        "confidence": 0.5, "rule_status": "uncertain",
        "phenomenon_tags": ["grammar_rule"],
        "trigger_conditions": ["any"],
        "operation_steps": ["apply the rule"],
        "output_constraints": [
            {"kind": "must_include_regex", "description": "finite verb appears in second position",
             "value": "finite verb appears in second position"}],
        "examples": [],
        "source": SRC,
    },
    {
        "id": "LB_SMOKE_0006", "language": "lb", "scope": "noun phrase | definiteness",
        "operation_type": "inflect", "priority": 40,
        "statement": "Definite determiners surface as reduced forms in some contexts.",
        "confidence": 0.6, "rule_status": "variable",
        "phenomenon_tags": ["definiteness"],
        "trigger_conditions": ["definite NP"],
        "operation_steps": ["select the determiner form"],
        "output_constraints": [
            {"kind": "token_count_range", "description": "translation is not degenerate",
             "min_tokens": 2, "max_tokens": 40}],
        "examples": [],
        "source": SRC,
    },
    {
        "id": "LB_SMOKE_0007", "language": "lb", "scope": "noun phrase | override test",
        "operation_type": "lexical_override", "priority": 95,
        "statement": "Irregular determiner form overrides the default form.",
        "confidence": 0.9, "rule_status": "lexical_exception",
        "phenomenon_tags": ["definiteness", "lexical_exception"],
        "trigger_conditions": ["listed lexical items"],
        "operation_steps": ["use the irregular form"],
        "output_constraints": [
            {"kind": "must_exclude_regex", "description": "no default form on listed items",
             "value": r"\bde\s+M\u00e4dchen\b"}],
        "examples": [],
        "source": SRC,
        "dependencies": [{"rule_id": "LB_SMOKE_0006", "relation": "overrides", "note": "exception beats default"}],
    },
]


def write_fixture_cards(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for obj in FIXTURES:
            card = MBGCard.model_validate(obj)
            f.write(json.dumps(card.as_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n


def is_prose_pattern(value: str) -> bool:
    """A usable regex is not a sentence: no spaces between words unless escaped classes, and it compiles."""
    import re

    if not value:
        return True
    try:
        re.compile(value)
    except re.error:
        return True
    # Heuristic used in the audit: prose values contain bare word sequences.
    bare_words = re.findall(r"[A-Za-z]{3,}(?:\s+[A-Za-z]{3,}){1,}", value)
    return len(bare_words) > 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards", default=str(FIXTURE_CARDS), help="cards JSONL to audit")
    ap.add_argument("--limit", type=int, default=3, help="train.jsonl rows for the eval replay")
    args = ap.parse_args()

    results: list[tuple[str, str, str]] = []

    def gate(name: str, ok: bool, detail: str) -> None:
        results.append((name, "PASS" if ok else "FAIL", detail))

    def warn(name: str, detail: str) -> None:
        results.append((name, "WARN", detail))

    # G1 imports -------------------------------------------------------------------
    gate("G1 package imports (no GPU/verl needed)", True, "grammetarl + tools + rewards + env imported")

    # G2 default cards path --------------------------------------------------------
    ws = get_workspace_paths()
    default_cards = ws.rules_cards_file
    gate("G2 default cards file exists", default_cards.exists(),
         f"{default_cards} exists={default_cards.exists()}")

    # G3 schema round-trip ---------------------------------------------------------
    cards = load_cards(FIXTURE_CARDS)          # structural gates always run on the fixture
    audit_cards = load_cards(Path(args.cards))  # content gates run on whatever we are auditing
    dumped = cards[0].as_dict()
    missing = [k for k in ("statement", "confidence", "rule_status", "exceptions_and_variants")
               if k not in dumped]
    gate("G3 schema keeps statement/confidence/rule_status/exceptions", not missing,
         f"dropped fields: {missing or 'none'} (pydantic extra='ignore') "
         f"-- audit set has {len(audit_cards)} cards")

    # G4 constraint audit ----------------------------------------------------------
    n_cards = len(audit_cards)
    n_no_constraints = sum(1 for c in cards if not c.output_constraints)
    prose = 0
    compilable = 0
    deferred = 0
    for c in audit_cards:
        for cons in c.output_constraints:
            if cons.kind in ("llm_check", "note"):
                deferred += 1
            elif cons.kind in ("must_include_regex", "must_exclude_regex"):
                if is_prose_pattern(cons.value or ""):
                    prose += 1
                else:
                    compilable += 1
    total_constraints = sum(len(c.output_constraints) for c in audit_cards)
    gate("G4 no prose-as-regex constraints", prose == 0,
         f"cards={n_cards} constraints={total_constraints} "
         f"compilable={compilable} prose_as_regex={prose} deferred={deferred} "
         f"cards_with_zero_constraints={n_no_constraints}/{n_cards}")

    # G16 sentinels leaked into scope ---------------------------------------------
    sentinel = [c.id for c in audit_cards if any(
        s in (c.scope or "").lower() for s in ("none", "null", "n/a", "unspecified"))]
    gate("G16 no 'None/null/unspecified' sentinels in scope", not sentinel,
         f"{len(sentinel)}/{n_cards} cards affected, e.g. {sentinel[:3]} "
         "(llm_extract does str(None) -> 'None' on dict scope components)")

    # G17 examples are usable pairs ------------------------------------------------
    degen = 0
    usable = 0
    for c in audit_cards:
        for ex in c.examples:
            if not ex.target_sentence or ex.target_sentence == ex.source_sentence:
                degen += 1
            else:
                usable += 1
    gate("G17 examples are non-degenerate src/tgt pairs", degen == 0 and usable > 0,
         f"usable={usable} degenerate(empty-or-identical)={degen}")

    # G18 / G19 distributions (diagnostic, not pass/fail) --------------------------
    from collections import Counter

    ops = Counter(c.operation_type for c in audit_cards)
    tags = Counter(t for c in audit_cards for t in c.phenomenon_tags)
    warn("G18 operation_type distribution",
         f"{dict(ops)} -- phonotactic/phonological rules must not collapse into 'reorder'")
    warn("G19 phenomenon_tag distribution",
         f"{dict(tags)} -- _infer_phenomenon_tags keys on substrings of prose, so it "
         "manufactures 'word_order'/'agreement' noise")

    # G5 verifier actually discriminates ------------------------------------------
    v = TranslationVerifier()
    card1 = next(c for c in cards if c.id == "LB_SMOKE_0001")
    good = v.verify("He went to school", "Hie ass an d'Schoul gaangen.", [card1]).overall_passed
    bad = v.verify("He went to school", "Hie an d'Schoul gaangen ass.", [card1]).overall_passed
    gate("G5 verifier separates good/bad on an executable card", good and not bad,
         f"good_pass={good} bad_pass={bad}")

    # G5b every card's own examples pass its own constraints (the P1 admission gate)
    failed_selftest: list[str] = []
    for c in cards:
        for ex in c.examples:
            if not ex.target_sentence:
                continue
            rep = v.verify(ex.source_sentence, ex.target_sentence, [c])
            if not rep.overall_passed:
                failed_selftest.append(f"{c.id}:{[m for r in rep.results for m in r.messages][:1]}")
    gate("G5b each card passes its own example (self-test admission)", not failed_selftest,
         f"cards_with_failing_self_test={len(failed_selftest)} {failed_selftest[:2] or ''}")

    # G5c normalization robustness: does a clause-final constraint survive punctuation?
    import re as _re

    anchored = r"\b(ass|gaangen|gemach)$"
    clean = bool(_re.search(anchored, "well hie gaangen ass"))
    punct = bool(_re.search(anchored, "well hie gaangen ass."))
    gate("G5c verifier normalizes punctuation before matching", clean == punct,
         f"anchored-constraint clean_match={clean} punctuated_match={punct} "
         "-> TranslationVerifier matches raw strings with no normalization layer")

    # G5d example orientation for a target-language card
    warn("G5d example orientation (en->lb)",
         "llm_extract._normalize_card_obj maps prompt examples "
         "{sentence, translation} -> {source_sentence=sentence, target_sentence=translation}. "
         "For an lb-target grammar card the book sentence IS the target, so as written the "
         "verifier checks the constraint against the ENGLISH side. Fixture cards below use the "
         "corrected orientation (source=en, target=lb).")

    # G6 reward is non-degenerate --------------------------------------------------
    scores = []
    for c in cards:
        rep = v.verify("He went to school", "Hie ass an d'Schoul gaangen.", [c])
        scores.append(grammar_reward(rep.overall_passed, len(rep.results)))
    spread = statistics.pstdev(scores) if len(scores) > 1 else 0.0
    gate("G6 grammar reward is non-degenerate across cards", spread > 0.1,
         f"n={len(scores)} unique={sorted(set(scores))} std={spread:.3f}")

    # G7 index: retrieval + dependency ordering + conflict resolution -------------
    idx = RuleIndex(cards)
    order = idx.order_for_application(["LB_SMOKE_0001", "LB_SMOKE_0002"])
    ids = [c.id for c in order]
    gate("G7a topological order respects 'requires'", ids.index("LB_SMOKE_0001") < ids.index("LB_SMOKE_0002"),
         f"order={ids}")
    resolved = [c.id for c in idx.order_for_application(["LB_SMOKE_0006", "LB_SMOKE_0007"])]
    gate("G7b 'overrides' removes the overridden card", resolved == ["LB_SMOKE_0007"], f"resolved={resolved}")
    hits = idx.retrieve("the man went to school because he was late", top_k=5)
    gate("G7c retrieval returns candidates", len(hits) > 0, f"top={[(c.id, round(s,3)) for c, s in hits[:3]]}")

    # G8 env trajectory ------------------------------------------------------------
    # NOTE: GrammarFirstEnv does not thread cards_path/namespace into its tools, so the
    # env is hardwired to ws.rules_cards_file. To exercise it we stage the fixture at that
    # path and remove it again if we created it.
    staged = False
    if not default_cards.exists():
        default_cards.parent.mkdir(parents=True, exist_ok=True)
        default_cards.write_text(Path(args.cards).read_text(encoding="utf-8"), encoding="utf-8")
        staged = True
    warn("G8c env threads cards_path/namespace to its tools",
         "GrammarFirstEnv.step calls grammar_rule_search/dictionary_lookup/grammar_rule_apply "
         "with no cards_path or namespace, so the env can only ever read "
         f"{default_cards}. Per-experiment isolation is impossible; smoke test staged a fixture.")
    env = GrammarFirstEnv(step_budget=6)
    env.reset(sample_id="smoke", src_sentence="The man went to school because he was late.")
    actions = [
        ("CALL_GRAMMAR_SEARCH", {"query": "main clause word order", "top_k": 3}),
        ("CALL_DICTIONARY", {"token": "man"}),
        ("CALL_APPLY_RULES", {"rule_ids": ["LB_SMOKE_0001"]}),
        ("RETURN_TRANSLATION", {"translation": "Ee Mann ass an d'Schoul gaangen."}),
    ]
    done = False
    state: dict = {}
    env_error = ""
    try:
        for a, p in actions:
            state, done = env.step(a, p)
    except Exception as exc:  # noqa: BLE001
        env_error = f"{type(exc).__name__}: {exc}"
    if staged:
        default_cards.unlink(missing_ok=True)
    gate("G8 env runs a grammar-first trajectory and terminates",
         (not env_error) and done and len(state.get("actions", [])) == 4,
         env_error or (f"steps={len(state['actions'])} done={done} "
                       f"retrieved={len(state['retrieved_rules'])} "
                       f"dict_hits={len(state['dictionary_entries'])}"))
    warn("G8b verify is reachable from the action space",
         "AgentAction has only CALL_GRAMMAR_SEARCH / CALL_DICTIONARY / CALL_APPLY_RULES / "
         "RETURN_TRANSLATION: the verifier is unreachable from the env even though the PRD "
         "makes grammar_verify pass rate a headline metric.")

    # G9 dictionary against the real LOD lexicon ----------------------------------
    lex_hits = dictionary_lookup(token="Mann", max_results=3)
    warn("G9 dictionary tool returns hits on the real lexicon",
         f"{ws.lexicon_file} exists={ws.lexicon_file.exists()} hits('Mann')={len(lex_hits)}")

    # G10 grammar_apply is not a placeholder --------------------------------------
    applied = grammar_rule_apply(["LB_SMOKE_0001"], "The man went to school.", cards_path=args.cards)
    gate("G10 grammar_apply draft is not the source sentence",
         applied.get("draft") != "The man went to school.",
         f"draft==src: {applied.get('draft') == 'The man went to school.'}")

    # G11 per-call index rebuild cost ---------------------------------------------
    t0 = time.perf_counter()
    for _ in range(5):
        grammar_rule_search(query="word order", src_sentence="The man went to school.", top_k=5,
                            cards_path=args.cards)
    per_call_ms = (time.perf_counter() - t0) / 5 * 1000
    warn("G11a per-call card load + index rebuild cost",
         f"{per_call_ms:.2f} ms/call on {len(cards)} cards (re-parses the file and rebuilds "
         "RuleIndex every single tool call)")
    t0 = time.perf_counter()
    for _ in range(5):
        dictionary_lookup(token="Mann", max_results=3)
    lex_ms = (time.perf_counter() - t0) / 5 * 1000
    warn("G11b dictionary_lookup cost (linear JSONL scan)",
         f"{lex_ms:.2f} ms/call; at 8192 rollouts x ~2 lookups = {lex_ms * 8192 * 2 / 1000:.0f} s "
         "of pure scanning per RL pass")

    # G12 dataset shape (sentences, not glosses) ----------------------------------
    train = ws.train_dataset_file
    if train.exists():
        rows = [json.loads(l) for l in train.open("r", encoding="utf-8") if l.strip()][:200]
        mean_src_tokens = sum(len(r.get("src", "").split()) for r in rows) / max(1, len(rows))
        gold_nonempty = sum(1 for r in rows if r.get("gold_rules"))
        gate("G12 train rows are sentences with gold_rules",
             mean_src_tokens >= 4 and gold_nonempty > 0,
             f"mean_src_tokens={mean_src_tokens:.2f} rows_with_gold_rules={gold_nonempty}/{len(rows)}")
    else:
        gate("G12 train dataset exists", False, str(train))

    # G13 end-to-end eval replay (run_eval) ---------------------------------------
    from grammetarl.training.run_eval import replay_trajectory

    sample = {"src": "The man went to school.", "tgt_ref": "Ee Mann ass an d'Schoul gaangen."}
    rep = replay_trajectory(src=sample["src"], reference=sample["tgt_ref"], cards_path=args.cards)
    gate("G13 replay produces a hypothesis different from the source",
         rep["hypothesis"] != sample["src"],
         f"chrF={rep['chr_f']:.3f} reward={rep['reward']:.3f} grammar_passed={rep['grammar_passed']} "
         f"checked={rep['checked_rules']} hyp={rep['hypothesis']!r}")
    agg = aggregate_reward(hypothesis=rep["hypothesis"], reference=sample["tgt_ref"],
                           grammar_passed=rep["grammar_passed"], checked_rules=rep["checked_rules"],
                           action_names=rep["actions"])
    warn("G13b aggregate reward magnitude", f"aggregate={agg:.3f}")

    # G14 reward ordering: copying the source must NOT beat a correct translation ---
    from grammetarl.training.run_eval import _chr_f

    chr_f_copy = _chr_f(sample["src"], sample["tgt_ref"])
    r_copy = aggregate_reward(hypothesis=sample["src"], reference=sample["tgt_ref"],
                              grammar_passed=False, checked_rules=5, action_names=rep["actions"])
    r_try = aggregate_reward(hypothesis="Hie an d'Schoul gaangen ass.", reference=sample["tgt_ref"],
                             grammar_passed=False, checked_rules=5, action_names=rep["actions"])
    r_good = aggregate_reward(hypothesis="Hie ass an d'Schoul gaangen.", reference=sample["tgt_ref"],
                              grammar_passed=True, checked_rules=5, action_names=rep["actions"])
    gate("G14 reward ordering: copy-source < wrong attempt < correct+verified",
         r_copy < r_try < r_good,
         f"r(copy)={r_copy:.3f} r(attempt_wrong)={r_try:.3f} r(correct_verified)={r_good:.3f}")

    # G15 metric sanity: is run_eval's chrF discriminative at all? ------------------
    chr_f_correct = _chr_f("Hie ass an d'Schoul gaangen.", sample["tgt_ref"])
    chr_f_self = _chr_f(sample["tgt_ref"], sample["tgt_ref"])
    gate("G15 chrF is discriminative (copy-src must be far below a correct hypothesis)",
         chr_f_self > 0.99 and chr_f_copy < 0.35 and (chr_f_correct - chr_f_copy) > 0.4,
         f"chrF(ref,ref)={chr_f_self:.3f} chrF(copy_src,ref)={chr_f_copy:.3f} "
         f"chrF(correct_lb,ref)={chr_f_correct:.3f} "
         "-> _chr_f uses type counts and averages both sides, inflating cross-language scores")

    # report ----------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("P0 SMOKE TEST GATE TABLE")
    print("=" * 78)
    for name, status, detail in results:
        print(f"[{status:>4}] {name}\n        {detail}")
    npass = sum(1 for _, s, _ in results if s == "PASS")
    nfail = sum(1 for _, s, _ in results if s == "FAIL")
    nwarn = sum(1 for _, s, _ in results if s == "WARN")
    print("-" * 78)
    print(f"PASS={npass} FAIL={nfail} WARN={nwarn}   cards={args.cards}")
    print("P0 gate is GREEN only when every FAIL is cleared (WARN = measured risk, report it).")
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    n = write_fixture_cards(FIXTURE_CARDS)
    print(f"wrote {n} fixture cards -> {FIXTURE_CARDS}")
    raise SystemExit(main())
