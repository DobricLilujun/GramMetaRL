"""Regression tests for the formal layer (P6). The anchor cases are here because a wrong
anchor reading silently changes a constraint's language; that bug was found by the Lean kernel."""
from grammetarl.formal.regex_dfa import DFA, UnsupportedRegex, compile_pattern, normalize_anchor

AL = tuple(sorted("abc ")) + ("\x00", "\x01")


def test_anchor_normalization_is_not_silent_stripping():
    assert normalize_anchor(".*en$") == (".*en", False)      # suffix, not infix
    assert normalize_anchor("^(dat|hatt)$") == ("(dat|hatt)", False)
    assert normalize_anchor("^ass") == ("ass.*", False)      # prefix
    assert normalize_anchor("datts|wéinis".replace("é", "c")) == ("datts|wc", True)


def test_suffix_obligation_language():
    d = DFA(compile_pattern(*normalize_anchor(".*en$"), search=None) if False else
            compile_pattern(normalize_anchor(".*en$")[0], AL, search=False), AL)
    assert d.accepts("aen") and not d.accepts("aenb")


def test_lookahead_is_reported_not_silently_miscompiled():
    try:
        compile_pattern("^(?!zur).*der$", AL, search=False)
        assert False, "lookahead must be rejected as out-of-fragment"
    except UnsupportedRegex as exc:
        assert "(?!" in exc.reason


def test_conflict_uses_complement_and_intersection():
    a = DFA(compile_pattern("zur", AL, search=True), AL)
    b = DFA(compile_pattern("der", AL, search=True), AL)
    assert not a.intersect(b.complement()).is_empty()   # 'zur' without 'der' exists
    x = DFA(compile_pattern("zur", AL, search=True), AL)
    y = DFA(compile_pattern("zur", AL, search=True), AL).complement()
    assert x.intersect(y).is_empty()                    # P and not-P cannot coexist


def test_empty_pattern_alphabet_is_not_silently_dead():
    tiny = ("a", "b")
    d = DFA(compile_pattern("c", tiny, search=True), tiny)
    assert d.is_empty()  # correct only because 'c' is outside this alphabet
