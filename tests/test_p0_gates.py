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
    assert card.operation_type == "phonotactics"  # not collapsed to fallback/reorder


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


def test_chrf_is_bounded_and_discriminative():
    ref = "Hie ass an d'Schoul gaangen."
    assert abs(_chr_f(ref, ref) - 1.0) < 1e-9
    assert _chr_f("He went to school.", ref) < 0.5
    assert _chr_f("Hie ass an d'Schoul gaangen", ref) > _chr_f("He went to school.", ref)
