"""Unit tests for the P0 extraction/verification fixes.

Run: /raid/yewei/.venvs/gm-lite/bin/python -m pytest tests -q
"""

from __future__ import annotations

from grammetarl import MBGCard, TranslationVerifier
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
        "output_constraints": [{"kind": "must_not_include_regex", "description": "no zur+der",
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
    assert applies.results[0].verdict == "satisfied"
    assert does_not.results[0].verdict == "not_applicable"
    assert does_not.decided == 0
    base = grammar_reward_verdicts(applies.verdicts())
    with_noise = grammar_reward_verdicts(applies.verdicts() + does_not.verdicts())
    assert base == with_noise
    assert grammar_reward_verdicts(does_not.verdicts()) == 0.0


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
