"""Unit tests for the P0 extraction/verification fixes.

Run: /raid/yewei/.venvs/gm-lite/bin/python -m pytest tests -q
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from grammetarl import MBGCard, TranslationVerifier
from grammetarl.schema import RuleConstraint, RuleExample
from grammetarl.rewards.grammar_reward import grammar_reward_verdicts
from grammetarl.llm_extract import MBGExtractor, looks_like_prose
from grammetarl.training.run_eval import _chr_f

EXTRACTOR = MBGExtractor(endpoint="http://unused.invalid", model="unused")

SOURCE = {"source_id": "t", "page_start": 1, "page_end": 1}


def _normalize(obj: dict) -> dict:
    return EXTRACTOR._normalize_card_obj(dict(obj))


def _card(obj: dict) -> MBGCard:
    return MBGCard.model_validate(_normalize(obj))


def test_rich_prompt_fields_survive():
    card = _card({
        "id": "LB_T_1", "language": "lb", "scope": "main clause", "operation_type": "reorder",
        "statement": "The finite verb is in second position.",
        "rule_status": "categorical", "confidence": 0.9,
        "exceptions_and_variants": [{"type": "exception", "description": "verb-final in subordinate clauses"}],
        "source": SOURCE,
    })
    assert card.statement.startswith("The finite verb")
    assert card.rule_status == "categorical"
    assert card.confidence == 0.9
    assert card.exceptions_and_variants[0].description.startswith("verb-final")
    assert "second position" in card.as_index_text()


def test_scope_never_leaks_none_sentinel():
    card = _card({
        "id": "LB_T_2", "language": "lb", "operation_type": "agree",
        "scope": {"syntactic_domain": "noun phrase", "constituent_target": None,
                  "linear_domain": "null", "semantic_pragmatic_scope": None},
        "source": SOURCE,
    })
    assert card.scope == "noun phrase"
    assert "None" not in card.scope and "null" not in card.scope.lower()


def test_prose_in_regex_slot_is_downgraded_not_emitted():
    card = _card({
        "id": "LB_T_3", "language": "lb", "scope": "clause", "operation_type": "reorder",
        "output_constraints": [
            {"kind": "must_include_regex", "description": "finite verb appears in second position",
             "value": "finite verb appears in second position"},
            {"kind": "required_order", "description": "verb before subject in V1", "value": None},
            {"kind": "forbidden_form", "description": "no double negation",
             "value": r"\bnet\b.{0,40}\bnet\b"},
        ],
        "source": SOURCE,
    })
    kinds = [c.kind for c in card.output_constraints]
    assert kinds == ["llm_check", "llm_check", "must_exclude_regex"]
    assert looks_like_prose("finite verb appears in second position") is True
    assert looks_like_prose(r"\bnet\b.{0,40}\bnet\b") is False


def test_example_orientation_book_sentence_is_the_target():
    card = _card({
        "id": "LB_T_4", "language": "lb", "scope": "clause", "operation_type": "negate",
        "examples": [{"sentence": "Hie ass net gaangen.", "translation": "He did not go.",
                      "gloss": None, "grammaticality": "grammatical"}],
        "source": SOURCE,
    })
    ex = card.examples[0]
    assert ex.target_sentence == "Hie ass net gaangen."
    assert ex.source_sentence == "He did not go."


def test_example_without_translation_is_not_duplicated():
    card = _card({
        "id": "LB_T_5", "language": "lb", "scope": "word", "operation_type": "phonotactics",
        "examples": [{"sentence": "Strof [ʃtʀoːf]"}],
        "source": SOURCE,
    })
    ex = card.examples[0]
    assert ex.target_sentence == "Strof [ʃtʀoːf]"
    assert ex.source_sentence == ""
    assert card.operation_type == "phonotactics"


def test_example_copied_into_both_slots_is_not_a_pair():
    """Observed in real extraction: the model put the book sentence in sentence AND translation."""
    card = _card({
        "id": "LB_T_9", "language": "lb", "scope": "word", "operation_type": "phonotactics",
        "examples": [{"sentence": "Interieur [ˈɛ̃ːntɛʀjœːʀ]", "translation": "Interieur [ˈɛ̃ːntɛʀjœːʀ]"}],
        "source": SOURCE,
    })
    ex = card.examples[0]
    assert ex.target_sentence == "Interieur [ˈɛ̃ːntɛʀjœːʀ]"
    assert ex.source_sentence == ""  # not collapsed to fallback/reorder


def test_verifier_normalizes_punctuation_and_case():
    card = _card({
        "id": "LB_T_6", "language": "lb", "scope": "subordinate clause", "operation_type": "reorder",
        "output_constraints": [{"kind": "must_include_regex", "description": "verb final",
                                "value": r"\b(well|datt)\b.*\b(ass|gaangen)$"}],
        "source": SOURCE,
    })
    report = TranslationVerifier().verify(
        "He went because he was late.", "well hie gaangen ass.", [card])
    assert report.results[0].passed is True
    assert report.overall_passed is True


def _app_card(pattern, side="target", level="phrase"):
    """Card whose applicability predicate is admitted (fires on its own example target)."""
    return _card({
        "id": "LB_T_7", "language": "lb", "scope": "dative NP", "operation_type": "clitic",
        "trigger_conditions": ["a dative preposition is followed by the feminine article der"],
        "applicability": [{
            "source_text": "a dative preposition is followed by the feminine article der",
            "kind": "regex", "side": side, "pattern": pattern, "level": level, "accepted": True,
        }],
        "output_constraints": [{"kind": "must_exclude_regex", "description": "no zur+der",
                                "value": "zur der"}],
        "examples": [{"sentence": "On the gallery", "translation": "Op der Aeltere-Galerie"}],
        "source": SOURCE,
    })


def test_inapplicable_rule_cannot_change_the_reward():
    """The invariant that makes the grammar term usable in RL.

    Retrieval puts at least one executable card in front of the policy for ~98% of sentences and
    most of those rules do not govern the sentence. If an inapplicable rule contributes a term the
    reward is mostly noise (measured: 85.2% check failures on 48k real pairs).
    """
    from grammetarl.rewards.grammar_reward import grammar_reward_verdicts

    card = _app_card("op der")
    v = TranslationVerifier()
    applies = v.verify("On the gallery", "Op der Galerie ass.", [card])
    does_not = v.verify("He went to school", "Hie ass an d'Schoul gaangen.", [card])
    # Exclusions are ONE-SIDED evidence: they can penalize, but a clean sentence is not
    # certification. A never-firing exclusion contributes no verdict at all.
    assert applies.results[0].verdict == "unknown", applies.results[0].message
    assert does_not.results[0].verdict == "not_applicable"
    assert does_not.decided == 0
    base = grammar_reward_verdicts(applies.verdicts())
    with_noise = grammar_reward_verdicts(applies.verdicts() + does_not.verdicts())
    assert base == with_noise
    assert grammar_reward_verdicts(applies.verdicts()) == 0.0
    assert grammar_reward_verdicts(does_not.verdicts()) == 0.0


def test_exclusion_can_penalize_when_it_fires():
    """The other half of one-sided evidence: when the forbidden shape IS present, it must bite."""
    card = _app_card("op der")
    # needs both: the applicability trigger ("op der") and the forbidden shape ("zur der")
    rep = TranslationVerifier().verify("On the gallery", "Op der Galerie ass zur der Lamp.", [card])
    assert rep.results[0].verdict == "violated", rep.results[0].messages
    assert grammar_reward_verdicts(rep.verdicts()) < 0.0


def test_not_applicable_skips_constraint_evaluation():
    card = _app_card("op der")
    rep = TranslationVerifier().verify("He left.", "Hie ass fort.", [card])
    msgs = " ".join(rep.results[0].messages)
    assert "must_not_include_regex" not in msgs  # never evaluated for an inapplicable rule
    assert rep.results[0].verdict == "not_applicable"


def test_unknown_applicability_is_not_a_verdict():
    """Cards without admitted predicates abstain instead of guessing."""
    card = _card({
        "id": "LB_T_8", "language": "lb", "scope": "word", "operation_type": "inflect",
        "output_constraints": [{"kind": "must_include_regex", "description": "x",
                                "value": "bauen"}],
        "examples": [{"sentence": "I build", "translation": "ech bauen"}],
        "source": SOURCE,
    })
    rep = TranslationVerifier().verify("I build.", "ech bauen.", [card])
    assert rep.results[0].verdict == "unknown"
    assert rep.decided == 0


def test_chrf_is_bounded_and_discriminative():
    ref = "Hie ass an d'Schoul gaangen."
    assert abs(_chr_f(ref, ref) - 1.0) < 1e-9
    assert _chr_f("He went to school.", ref) < 0.5
    assert _chr_f("Hie ass an d'Schoul gaangen", ref) > _chr_f("He went to school.", ref)


def test_chat_text_sends_token_budget_and_falls_back_to_reasoning(monkeypatch):
    """Thinking servers (e.g. a local strata/llama.cpp Qwen build) return content=null while
    reasoning, and burn the whole context window when no max_tokens is sent."""
    import json

    from grammetarl import llm_http

    sent: list[dict] = []

    class FakeResp:
        def __init__(self, payload: dict) -> None:
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return self._payload

    def fake_post(url, headers=None, data=None, timeout=None):  # noqa: ANN001, ARG001
        sent.append(json.loads(data))
        return FakeResp({
            "choices": [{"message": {"role": "assistant", "content": None,
                                      "reasoning_content": "[{\"id\": \"X\"}]"},
                         "finish_reason": "stop"}],
            "usage": {"completion_tokens": 3},
        })

    monkeypatch.setattr(llm_http.requests, "post", fake_post)
    text, meta = llm_http.chat_text("http://x/v1", "m", [{"role": "user", "content": "hi"}],
                                    max_tokens=4096, reasoning_effort="minimal")
    assert sent[0]["max_tokens"] == 4096
    assert sent[0]["reasoning_effort"] == "minimal"
    assert text.startswith("[{")
    assert meta["finish_reason"] == "stop"
    assert meta["usage"]["completion_tokens"] == 3


def test_chat_text_raises_an_actionable_error_when_both_fields_are_null(monkeypatch):
    from grammetarl import llm_http

    class FakeResp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"choices": [{"message": {"role": "assistant", "content": None},
                                 "finish_reason": "length"}],
                    "usage": {"completion_tokens": 512}}

    monkeypatch.setattr(llm_http.requests, "post",
                        lambda url, headers=None, data=None, timeout=None: FakeResp())
    try:
        llm_http.chat_text("http://x/v1", "m", [{"role": "user", "content": "hi"}], max_tokens=32)
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert "max_tokens=32" in str(exc)


def test_extract_json_salvages_complete_cards_from_a_truncated_array():
    """A cut-off tail must not discard the cards that did complete."""
    extractor = MBGExtractor(endpoint="http://unused.invalid", model="unused")
    truncated = (
        '[{"id": "LB_A", "scope": "clause"}, '
        '{"id": "LB_B", "scope": "noun phrase"}, '
        '{"id": "LB_C", "scope": "unter'
    )
    parsed = extractor._extract_json(truncated)
    assert [p["id"] for p in parsed] == ["LB_A", "LB_B"]


def test_extract_json_handles_fenced_output():
    extractor = MBGExtractor(endpoint="http://unused.invalid", model="unused")
    parsed = extractor._extract_json('Here you go:\n```json\n[{"id": "LB_A"}]\n```')
    assert [p["id"] for p in parsed] == ["LB_A"]


def test_deferred_kinds_never_produce_a_reward_term():
    """"nominative = accusative" cannot be a containment check. It must abstain, not guess."""
    from grammetarl.schema import MBGCard, RuleConstraint, RuleExample

    card = MBGCard(
        id="LB_equiv",
        namespace="default",
        language="lb",
        scope="syntax",
        operation_type="inflect",
        source={"source_id": "test", "page_start": 1, "page_end": 1},
        statement="Nominative and accusative forms are identical for adjectives.",
        phenomenon_tags=["case"],
        trigger_conditions=[],
        output_constraints=[
            RuleConstraint(kind="equivalence", description="nom = acc", values=["nom", "acc"]),
            RuleConstraint(kind="optionality", description="particle may be omitted", values=["huesch"]),
        ],
        examples=[RuleExample(source_language="en", source_sentence="the big house",
                              target_language="lb", target_sentence="déi grouss Haus")],
    )
    report = TranslationVerifier(require_applicability=False).verify(
        "the big house", "déi grouss Haus", [card]
    )
    assert all(r.verdict == "unknown" for r in report.results), [r.verdict for r in report.results]
    assert report.decided == 0
    assert grammar_reward_verdicts(report.verdicts()) == 0.0


def test_disjunction_constraint_is_decidable():
    """A wrong allomorph set becomes a real disjunction over attested forms, and can pass."""
    from grammetarl.schema import MBGCard, RuleConstraint, RuleExample

    card = MBGCard(
        id="LB_disj",
        namespace="default",
        language="lb",
        scope="syntax",
        operation_type="inflect",
        source={"source_id": "test", "page_start": 1, "page_end": 1},
        statement="The neuter demonstrative appears as datt or dat.",
        phenomenon_tags=["demonstrative"],
        trigger_conditions=[],
        output_constraints=[
            RuleConstraint(kind="must_include_any_regex", description="datt or dat",
                           values=[r"\bdatt\b", r"\bdat\b"], level="token")
        ],
        examples=[RuleExample(source_language="en", source_sentence="that house",
                              target_language="lb", target_sentence="datt Haus")],
    )
    v = TranslationVerifier(require_applicability=False)
    ok = v.verify("that house", "dat Haus", [card])
    bad = v.verify("that house", "se Haus", [card])
    assert ok.results[0].verdict == "satisfied", ok.results[0].message
    assert bad.results[0].verdict == "violated", bad.results[0].message
    assert grammar_reward_verdicts(ok.verdicts()) > 0.0
    assert grammar_reward_verdicts(bad.verdicts()) < 0.0


def test_uniform_weights_reproduce_the_unweighted_reward():
    """The weighted form must be a strict generalization, not a different metric."""
    from grammetarl.rewards.grammar_reward import grammar_reward_verdicts, grammar_reward_weighted

    vs = ["satisfied", "violated", "unknown", "satisfied", "not_applicable"]
    assert grammar_reward_weighted(vs, [1.0] * len(vs)) == grammar_reward_verdicts(vs)
    assert grammar_reward_weighted(vs, [0.3] * len(vs)) == pytest.approx(grammar_reward_verdicts(vs))
    try:
        grammar_reward_weighted(vs, [1.0])
        raise AssertionError("misaligned weights must raise")
    except ValueError:
        pass


def test_unreliable_rules_cannot_move_the_reward():
    """An unmeasured or coin-flip rule has weight 0: it can neither help nor hurt.

    This is the soft version of the triage decision: instead of deleting a rule that contradicts
    professional human translations half the time, the reward stops believing it.
    """
    from grammetarl.rewards.grammar_reward import grammar_reward_weighted
    from grammetarl.rewards.reliability import ReliabilityTable

    table = ReliabilityTable(
        {
            "GOOD": {"applicable": 29, "satisfied": 29, "violated": 0},     # measured, reliable
            "COINFLIP": {"applicable": 40, "satisfied": 20, "violated": 20},  # no signal
            # UNMEASURED absent on purpose
        }
    )
    assert table.weight("GOOD") > 0.5
    assert table.weight("COINFLIP") == 0.0
    assert table.weight("UNMEASURED") == 0.0

    clean = grammar_reward_weighted(["satisfied", "satisfied"], table.weights(["GOOD", "COINFLIP"]))
    with_noise = grammar_reward_weighted(
        ["satisfied", "satisfied", "violated", "violated"],
        table.weights(["GOOD", "COINFLIP", "COINFLIP", "UNMEASURED"]),
    )
    assert clean == with_noise == 1.0


def test_reliability_shrinks_with_little_evidence():
    """2/2 observations must not certify a rule; the weight has to be earned."""
    from grammetarl.rewards.reliability import ReliabilityTable

    table = ReliabilityTable(
        {"SMALL": {"applicable": 2, "satisfied": 2}, "BIG": {"applicable": 60, "satisfied": 60}}
    )
    assert table.weight("SMALL") < 0.5
    assert table.weight("BIG") > 0.85


def _conditional_card() -> MBGCard:
    return MBGCard(
        id="LB_cond",
        namespace="default",
        language="lb",
        scope="adjective phrase",
        operation_type="inflect",
        statement="English comparatives require the analytic comparative with mees.",
        phenomenon_tags=["comparison"],
        trigger_conditions=["Forming the comparative degree"],
        source={"source_id": "test", "page_start": 1, "page_end": 1},
        output_constraints=[RuleConstraint(
            kind="source_conditional",
            description="en comparative => mees + adjective",
            source_pattern=r"\bmore [a-z]+\b",
            values=[r"\bméi\s+[a-z]+\b"],
            level="phrase",
        )],
        examples=[RuleExample(source_language="en", source_sentence="more expensive",
                              target_language="lb", target_sentence="méi deier")],
    )


def test_source_conditional_abstains_when_the_source_lacks_the_condition():
    """No source trigger => the rule does not govern the pair => no verdict, no reward term."""
    card = _conditional_card()
    rep = TranslationVerifier(require_applicability=False).verify(
        "the price is high", "d'Präis ass héich", [card]
    )
    assert rep.results[0].verdict == "unknown", rep.results[0].messages
    assert rep.decided == 0
    assert grammar_reward_verdicts(rep.verdicts()) == 0.0


def test_source_conditional_decides_both_ways_when_the_source_has_the_condition():
    card = _conditional_card()
    v = TranslationVerifier(require_applicability=False)
    ok = v.verify("more expensive", "méi deier", [card])
    bad = v.verify("more expensive", "deier", [card])
    assert ok.results[0].verdict == "satisfied", ok.results[0].messages
    assert bad.results[0].verdict == "violated", bad.results[0].messages
    assert grammar_reward_verdicts(ok.verdicts()) > 0.0
    assert grammar_reward_verdicts(bad.verdicts()) < 0.0


def test_source_conditional_is_not_a_target_only_check():
    """The condition lives on the source, so the check cannot equal its own applicability test."""
    card = _conditional_card()
    src_pat = card.output_constraints[0].source_pattern
    tgt_pats = card.output_constraints[0].values
    assert src_pat not in tgt_pats
    rep = TranslationVerifier(require_applicability=False).verify(
        "expensive", "méi deier", [card]  # target has the form, source has no comparative
    )
    assert rep.results[0].verdict == "unknown"


def test_negated_source_conditional_prohibits_a_position():
    card = MBGCard(
        id="LB_neg", namespace="default", language="lb", scope="sentence",
        operation_type="reorder",
        statement="een may not open the sentence.",
        phenomenon_tags=["word order"], trigger_conditions=["Sentence begins with One"],
        source={"source_id": "test", "page_start": 1, "page_end": 1},
        output_constraints=[RuleConstraint(
            kind="source_conditional", negate=True,
            description="English-initial One => target must not start with een",
            source_pattern=r"^\s*(one|someone)\b", values=[r"^\s*een\b"], level="phrase",
        )],
        examples=[],
    )
    v = TranslationVerifier(require_applicability=False)
    bad = v.verify("One cannot park here.", "Een kann hei net parken", [card])
    ok = v.verify("One cannot park here.", "Dat ee kann hei net parken", [card])
    assert bad.results[0].verdict == "violated", bad.results[0].messages
    assert ok.results[0].verdict == "satisfied", ok.results[0].messages
    # pipeline mode (applicability required): source condition absent => the rule does not govern
    vp = TranslationVerifier(require_applicability=True)
    off = vp.verify("Many people cannot park here.", "Vill Leit kënnen hei net parken", [card])
    assert off.decided == 0 and off.results[0].verdict == "not_applicable", off.results[0].messages
    # even when the caller skips applicability, an ungoverned constraint contributes no decision
    assert v.verify("Many people cannot park here.", "Vill Leit kënnen hei net parken",
                    [card]).decided == 0


def test_multi_instance_rule_needs_every_instance_broken():
    """405 of 1,968 possessive pairs carry 2+ possessors: breaking one legitimately leaves the check
    satisfied, so single-token corruption under-measures discrimination for per-occurrence rules."""
    from scripts.verifier_discrimination import corrupt_all, corrupt_token, inject_forbidden

    v = TranslationVerifier()
    pats = [r"\bmäi\w*\b", r"\bhir\w*\b"]
    one = "mäi Buch"
    two = "mäi Buch an hir Schoul"
    assert corrupt_all(v, one, pats, 1)[0] is None          # 1 instance: not this function's job
    bad_two, how = corrupt_all(v, two, pats, 1)
    assert bad_two is not None, how
    for pat in pats:
        assert not re.search(pat, bad_two, re.I), bad_two
    assert corrupt_token(v, two, pats, 1)[0] is not None    # single-token still available for 1-off rules


def test_positional_prohibition_is_discriminated_by_injection():
    from scripts.verifier_discrimination import inject_forbidden

    v = TranslationVerifier()
    out, how = inject_forbidden(v, "Dat ass hei net erlaabt", [r"^\s*een\b"], 0)
    assert out is not None and out.lower().startswith("een"), (out, how)
    mid, _ = inject_forbidden(v, "Dat ass hei net erlaabt", [r"\been\b"], 3)
    assert mid is not None and "een" in mid.split()


def test_source_conditional_cards_need_no_target_predicate():
    """A source-derived condition decides applicability, so no target-side predicate is required."""
    from grammetarl.validator import has_source_conditional

    assert not has_source_conditional(_conditional_card_plain())
    assert has_source_conditional(_conditional_card())
    # a source_conditional with no condition cannot decide anything -> not source-derived
    bare = _conditional_card()
    bare.output_constraints[0].source_pattern = None
    assert not has_source_conditional(bare)


def _conditional_card_plain() -> MBGCard:
    c = _conditional_card()
    c.output_constraints = [RuleConstraint(
        kind="must_include_any_regex", description="comparative suffix",
        values=[r"\b\w+er\b"], level="token")]
    return c


def test_main_verify_subcommand_emits_json():
    """`main.py verify` printed `report.__dict__`, but VerificationReport/RuleCheckResult are
    @dataclass(slots=True) and have no __dict__, so the subcommand died with AttributeError.
    Guard the CLI path end-to-end, not just asdict()."""
    import json
    import subprocess
    import sys

    repo = Path(__file__).resolve().parents[1]
    cards = repo / "data" / "rules" / "mbg_cards_condonly.jsonl"
    proc = subprocess.run(
        [sys.executable, "main.py", "verify",
         "--cards", str(cards),
         "--sentence", "Our children are taller than their parents.",
         "--translation", "Eis Kanner si méi grouss wéi hir Elteren.",
         "--rule-ids", "LB_9789dfb15623,LB_946a24dd25d6"],
        cwd=repo, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr[-500:]
    report = json.loads(proc.stdout[proc.stdout.index("{"):])
    assert report["decided"] == 2 and report["overall_passed"] is True
    assert [r["verdict"] for r in report["results"]] == ["satisfied", "satisfied"]
    # the report must stay weight-alignable: rule_ids and verdicts in the same order
    assert [r["rule_id"] for r in report["results"]] == ["LB_9789dfb15623", "LB_946a24dd25d6"]


# --------------------------------------------------------------------------- measured signal protocol

def test_wilson_interval_brackets_the_rate_and_shrinks_with_n():
    """The smoke prints a rate; at 24 groups the point estimate alone was over-read as a
    29% -> 17% regression. The interval has to be part of the artifact."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from p1_reward_smoke import _wilson

    lo7, hi7 = _wilson(7, 24)
    assert lo7 <= 7 / 24 <= hi7
    assert hi7 - lo7 > 0.15          # n=24 is a wide interval; that is the point of reporting it
    lo70, hi70 = _wilson(70, 240)    # same rate, 10x the groups
    assert hi70 - lo70 < (hi7 - lo7) / 2
    assert _wilson(0, 0) == [0.0, 0.0]


def test_rule_quality_reader_accepts_rule_keyed_artifact(tmp_path):
    """scripts/rule_quality_matrix.py writes a dict keyed by rule id; a reader that assumes a
    {'rules': [...]} shape silently selects zero rules, which looks like a passing empty smoke."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from p1_signal_by_rule import usable_ids

    keyed = tmp_path / "keyed.json"
    keyed.write_text(__import__("json").dumps({
        "LB_A": {"usable": True, "agreement": 0.9},
        "LB_B": {"usable": False, "agreement": 0.3},
    }), encoding="utf-8")
    assert usable_ids(keyed) == {"LB_A"}

    legacy = tmp_path / "legacy.json"
    legacy.write_text(__import__("json").dumps(
        {"rules": [{"rule_id": "LB_C", "usable": True}, {"rule_id": "LB_D", "usable": False}]}),
        encoding="utf-8")
    assert usable_ids(legacy) == {"LB_C"}


def _load_p1_signal_module():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / "p1_signal_by_rule.py"
    spec = importlib.util.spec_from_file_location("p1_signal_by_rule", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_similarity_proxy_is_bounded_and_discriminative():
    """Gradient alignment compares rollouts by lexical overlap with the professional reference.

    If that proxy were degenerate (always 0, or blind to an accent/apostrophe difference), the whole
    concordance measurement - and gate G31 - would be meaningless.
    """
    mod = _load_p1_signal_module()
    near = mod._token_f1("d Fabréck ass grouss", "d Fabréck ass ganz grouss")
    far = mod._token_f1("ganz anescht Wuert hei", "de Fabréck grouss")
    assert 0.5 < near <= 1.0
    assert far == 0.0
    assert mod._token_f1("", "de Fabréck") == 0.0


def test_stratified_signal_artifacts_record_the_exclusion_tradeoff():
    """The measured reason spread is not the objective: dropping the three anti-professional rules
    lowers coverage on identical hypotheses, and both runs must carry the alignment block."""
    root = Path(__file__).resolve().parents[1]
    all8 = root / "artifacts/p1/signal_condv3_stratified.json"
    pos5 = root / "artifacts/p1/signal_condv3_strat_pos5.json"
    if not (all8.exists() and pos5.exists()):
        pytest.skip("stratified signal artifacts not built in this checkout")
    import json

    a, b = json.loads(all8.read_text()), json.loads(pos5.read_text())
    assert len(a["rules_scored"]) > len(b["rules_scored"])
    assert a["groups"] == b["groups"]
    # subset of rules cannot manufacture spread on the same hypotheses
    assert b["groups_nonzero_spread"] <= a["groups_nonzero_spread"]
    for d in (a, b):
        al = d["alignment"]
        dec = al["concordant"] + al["discordant"]
        lo, hi = al["concordant_rate_ci95"]
        assert lo <= al["concordant_rate"] <= hi
        if dec:
            assert abs(al["concordant_rate"] - al["concordant"] / dec) < 1e-3


# --------------------------------------------------------------------- spread/alignment validity

def test_spread_is_reported_as_undefined_not_zero_below_two_rollouts():
    """A group of one rollout has spread 0 BY CONSTRUCTION. A 140-prompt corpus built for error mining
    (--rollouts 1) was briefly quoted as "advantage coverage 0.0% on unbiased prompts", which was
    arithmetic dressed as a measurement. The smoke must null the metric instead of printing 0.0."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from p1_reward_smoke import _spread_undefined

    assert "by construction" in _spread_undefined(1)
    assert _spread_undefined(2) == "" and _spread_undefined(3) == ""


def test_signal_script_refuses_a_one_rollout_log(tmp_path):
    """Same trap on the offline re-scoring side: p1_signal_by_rule must hard-stop rather than print
    spread 0%, alignment 0 decisive pairs and a confident-looking table."""
    import json
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    log = tmp_path / "one.jsonl"
    rows = [{"prompt_id": f"p{i}", "rollout": 0, "src": "Our son is taller.",
             "hyp": "Eise Jong ass méi grouss.", "tgt_ref": "Eise Jong ass méi grouss wéi säi Papp.",
             "verdicts": ["satisfied"], "rule_ids": ["LB_946a24dd25d6"], "r_grammar": 1.0}
            for i in range(3)]
    log.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "scripts/p1_signal_by_rule.py", "--rollout-log", str(log),
         "--rule-ids", "LB_946a24dd25d6", "--out", str(tmp_path / "sig.json")],
        cwd=root, capture_output=True, text=True)
    assert proc.returncode != 0
    assert "one hypothesis per prompt" in (proc.stdout + proc.stderr)
    assert not (tmp_path / "sig.json").exists()


def test_binomial_null_flags_an_unmeasurable_alignment():
    """2 concordant vs 8 discordant at n=10 is p=0.11 against the 0.5 null: NOT proof that the reward
    points the wrong way. Every concordance number must travel with this p-value."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from screen_concordance import _binom_two_sided_p

    assert abs(_binom_two_sided_p(5, 10) - 1.0) < 1e-9          # exactly the null
    assert 0.05 < _binom_two_sided_p(2, 10) < 0.20              # the measured stratified case
    assert _binom_two_sided_p(0, 10) < 0.01                     # extreme is significant
    assert _binom_two_sided_p(0, 0) == 1.0                      # no evidence at all


def test_concordance_scorer_direction_for_required_and_exclusion_constraints():
    """The direction rule (satisfied = preferred) is inverted for `negate: true` candidates, and both
    bugs in this session were exactly here. Pin both constraint kinds with a synthetic group."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from screen_concordance import score_candidate

    def group(hyps, sims):
        return {"src": "if our son is taller", "ref": "wann eise Jong méi grouss ass",
                "hyp_norms": hyps, "sims": sims}

    groups = [group(["wann eise Jong méi grouss ass", "eise Jong ass méi grouss"], [1.0, 0.6])]
    pool = [("if our son is taller", "wann eise Jong méi grouss ass")]

    required = {"name": "cond_if", "source_pattern": r"\bif\b", "negate": False,
                "values": [r"\bwann\b"]}
    r = score_candidate(required, pool, groups, fit_boundary=1)
    assert r["concordant"] == 1 and r["discordant"] == 0 and r["heldout_agreement"] is None
    assert r["fit_agreement"] == 1.0

    # exclusion: the forbidden word appears in the rollout CLOSER to the reference, so preferring its
    # absence ranks the pair AGAINST the reference -> discordant.
    forbidden = {"name": "pg_been_all", "source_pattern": r"\bif\b", "negate": True,
                 "values": [r"\balle\b"]}
    groups2 = [group(["wann eise Jong ass alle", "wann eise Jong grouss ass"], [1.0, 0.6])]
    e = score_candidate(forbidden, pool, groups2, fit_boundary=1)
    assert e["discordant"] == 1 and e["concordant"] == 0
    assert e["concordance_p_vs_0.5"] > 0.05      # one pair is never evidence
    # same rule, forbidden word in the FARTHER rollout -> concordant
    groups3 = [group(["wann eise Jong ass", "wann eise Jong ass alle"], [1.0, 0.4])]
    e2 = score_candidate(forbidden, pool, groups3, fit_boundary=1)
    assert e2["concordant"] == 1 and e2["discordant"] == 0


def test_every_script_in_scripts_parses():
    """A careless edit once merged two statements in scripts/smoke_test.py into one line, which is a
    SyntaxError only visible when the file is actually imported. Gate the whole directory instead."""
    import ast

    scripts = (Path(__file__).resolve().parents[1] / "scripts")
    for path in sorted(scripts.glob("*.py")):
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:  # pragma: no cover - the point is to fail loudly
            raise AssertionError(f"{path}: {exc}") from exc
