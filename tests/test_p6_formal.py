"""Regression tests for the formal layer (P6).

The anchor cases are here because a wrong anchor reading silently changes a constraint's language;
that exact bug was caught by the Lean kernel, not by the regex tests.
The alphabets below are deliberately small, and every literal used by a pattern is inside the
alphabet, otherwise `is_empty()` is an alphabet artifact rather than a property of the pattern.
"""
from grammetarl.formal.regex_dfa import (DFA, UnsupportedRegex, compile_pattern,
                                        normalize_anchor)

AL = tuple(sorted("abcde n u zr")) + ("\x00", "\x01")   # letters used by the patterns below


def test_anchor_normalization_is_not_silent_stripping():
    # prefixing ".*" (rather than stripping "$") keeps the suffix reading; the pattern already
    # began with ".*", hence the doubled star, which is language-equivalent.
    assert normalize_anchor(".*en$") == (".*.*en", False)
    assert normalize_anchor("^(dat|hatt)$") == ("(dat|hatt)", False)
    assert normalize_anchor("^ass") == ("ass.*", False)     # prefix
    assert normalize_anchor("datts|wéinis") == ("datts|wéinis", True)


def test_suffix_obligation_language():
    core, infix = normalize_anchor(".*en$")
    d = DFA(compile_pattern(core, AL, search=infix), AL)
    assert d.accepts("aen") and d.accepts("c nen")
    assert not d.accepts("aenb")      # must END in 'en': the point of the trailing anchor
    assert not d.accepts("aeb")


def test_prefix_obligation_language():
    core, infix = normalize_anchor("^zur")
    d = DFA(compile_pattern(core, AL, search=infix), AL)
    assert d.accepts("zur") and d.accepts("zurb")
    assert not d.accepts("azur")


def test_lookahead_is_reported_not_silently_miscompiled():
    try:
        compile_pattern("^(?!zur).*der$", AL, search=False)
        raise AssertionError("lookahead must be rejected as out-of-fragment")
    except UnsupportedRegex as exc:
        assert "(?!" in exc.reason


def test_conflict_uses_complement_and_intersection():
    a = DFA(compile_pattern("zur", AL, search=True), AL)
    b = DFA(compile_pattern("der", AL, search=True), AL)
    inter = a.intersect(b.complement())
    assert not inter.is_empty(), "'zur' without 'der' must exist"
    p = DFA(compile_pattern("zur", AL, search=True), AL)
    notp = DFA(compile_pattern("zur", AL, search=True), AL).complement()
    assert p.intersect(notp).is_empty(), "P and not-P cannot coexist"


def test_pattern_outside_the_alphabet_is_uninhabited_in_the_model():
    """Documented limit: "no match exists" is a bounded-model verdict over SIGMA, so a pattern
    using symbols outside SIGMA is uninhabited for model reasons, not card reasons."""
    tiny = ("a", "b")
    d = DFA(compile_pattern("c", tiny, search=False), tiny)
    assert d.is_empty() and d.witness() is None
    d2 = DFA(compile_pattern("ab", tiny, search=False), tiny)
    assert d2.accepts("ab") and not d2.accepts("aba") and not d2.accepts("a")


def test_fullmatch_class_language():
    d = DFA(compile_pattern("[a-b]*", AL, search=False), AL)
    assert d.accepts("ab") and not d.accepts("abz")   # 'z' outside [a-b]


def test_epsilon_matching_include_is_universal_under_search_semantics():
    """The verifier uses re.search, so a must_include that can match the empty string accepts
    every translation: it is a constant reward term. The analysis reports it as `universal`,
    and this is why `[a-b]*` is NOT a usable obligation."""
    d = DFA(compile_pattern("[a-b]*", AL, search=True), AL)
    assert d.universal()
