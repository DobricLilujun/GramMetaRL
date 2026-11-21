from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass

from .llm_http import DEFAULT_MAX_TOKENS, chat_text
from .prompt_manager import render_prompt
from .schema import MBGCard, RuleConstraint


@dataclass(slots=True)
class RuleCheckResult:
    rule_id: str
    passed: bool
    messages: list[str]
    # True when a must_exclude pattern matched nothing at all, so the "pass" proves nothing.
    # Anchored excludes such as ^(?!zur).*der$ searched over a whole sentence always match zero
    # tokens, which inflated apparent rule firing (measured on 400 real en->lb pairs).
    vacuous: bool = False


@dataclass(slots=True)
class VerificationReport:
    sentence: str
    translation: str
    overall_passed: bool
    results: list[RuleCheckResult]


class TranslationVerifier:
    def __init__(
        self,
        llm_endpoint: str | None = None,
        llm_model: str | None = None,
        api_key: str | None = None,
        normalize: bool = True,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self.llm_endpoint = llm_endpoint.rstrip("/") if llm_endpoint else None
        self.llm_model = llm_model
        self.api_key = api_key
        self.normalize = normalize
        self.max_tokens = max_tokens

    @staticmethod
    def normalize_text(text: str) -> str:
        """NFKC + casefold + whitespace collapse + strip sentence-final punctuation.

        Without this, clause-final constraints (a pattern ending in `$`) fail on a trailing
        period, and case-insensitive rules silently miss. Patterns are matched against this
        normalized form, so write them case-insensitively.
        """
        out = unicodedata.normalize("NFKC", text).casefold()
        out = re.sub(r"\s+", " ", out).strip()
        out = re.sub(r"[.,!?;:\"'’””]+$", "", out)
        return out

    def verify(self, sentence: str, translation: str, rules: list[MBGCard]) -> VerificationReport:
        results = [self._verify_single_rule(sentence, translation, rule) for rule in rules]
        overall = all(r.passed for r in results)
        return VerificationReport(
            sentence=sentence,
            translation=translation,
            overall_passed=overall,
            results=results,
        )

    def _verify_single_rule(
        self,
        sentence: str,
        translation: str,
        rule: MBGCard,
    ) -> RuleCheckResult:
        messages: list[str] = []
        passed = True
        vacuums = 0

        for cons in rule.output_constraints:
            haystack = self.normalize_text(translation) if self.normalize else translation
            ok, msg, vacuous = self._check_constraint(cons, haystack)
            messages.append(msg)
            if vacuous:
                vacuums += 1
            if not ok:
                passed = False

        if self.llm_endpoint and self.llm_model:
            llm_constraints = [c for c in rule.output_constraints if c.kind == "llm_check"]
            for cons in llm_constraints:
                ok, msg = self._llm_check(sentence, translation, rule, cons)
                messages.append(msg)
                if not ok:
                    passed = False

        if not rule.output_constraints:
            messages.append("No machine-checkable constraints on this rule.")

        return RuleCheckResult(
            rule_id=rule.id,
            passed=passed,
            messages=messages,
            vacuous=vacuums > 0,
        )

    # --- level-aware matching -------------------------------------------------------
    # A card constraint operates at some linguistic level. Extraction routinely produces
    # word-level anchored patterns (^word$) that cannot match a sentence, so the level must be
    # resolved before matching, otherwise every correct translation fails the rule.

    @staticmethod
    def _is_anchored(pattern: str) -> bool:
        return pattern.startswith("^") or pattern.endswith("$")

    # A pattern spans more than one word if it mentions spaces or unbounded interior spans.
    _PHRASAL_MARKERS = (" ", r"\s", ".*", ".+", "{2", r"\W")

    @classmethod
    def _is_phrasal(cls, pattern: str) -> bool:
        return any(m in pattern for m in cls._PHRASAL_MARKERS)

    @classmethod
    def _resolve_level(cls, c: RuleConstraint, pattern: str) -> str:
        """Explicit `level` wins; otherwise infer.

        \b(well|datt)\b.*\b(ass|gaangen)$ is sentence-level even though it is anchored,
        while ^(dat|hatt|et|seng)$ is a word-level form inventory.
        """
        if c.level:
            return c.level
        if cls._is_anchored(pattern) and not cls._is_phrasal(pattern):
            return "token"
        return "sentence"

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return [t for t in re.split(r"\s+", text.strip()) if t]

    @classmethod
    def _pattern_hits(cls, pattern: str, text: str, level: str) -> tuple[int, str]:
        """Return (n_matches, how). Anchored patterns are matched token-wise."""
        try:
            if level == "token":
                n = 0
                for tok in cls._tokens(text):
                    bare = tok.strip("\"“”‘’()[]{}«»,.;:!?-")
                    for candidate in {tok, bare}:
                        if candidate and re.search(pattern, candidate):
                            n += 1
                            break
                return n, "token-wise"
            return len(re.findall(pattern, text)), "whole-string"
        except re.error as exc:
            return -1, f"invalid regex: {exc}"

    def _check_constraint(self, c: RuleConstraint, translation: str) -> tuple[bool, str, bool]:
        if c.kind == "must_include_regex":
            if not c.value:
                return False, "must_include_regex missing value", False
            level = self._resolve_level(c, c.value)
            n, how = self._pattern_hits(c.value, translation, level)
            if n < 0:
                return False, f"must_include_regex({c.value}) => {how}", False
            if n == 0 and level == "token":
                # Anchored pattern also failed token-wise; report the level so the card can be fixed.
                return False, f"must_include_regex({c.value}) => False [{how}; anchored]", False
            return n > 0, f"must_include_regex({c.value}) => {n} hit(s) [{how}]", False

        if c.kind == "must_exclude_regex":
            if not c.value:
                return False, "must_exclude_regex missing value", False
            level = self._resolve_level(c, c.value)
            n_tok, _ = self._pattern_hits(c.value, translation, "token")
            n_str, how = self._pattern_hits(c.value, translation, "sentence")
            if n_tok < 0 or n_str < 0:
                return False, f"must_exclude_regex({c.value}) => invalid regex", False
            hits = max(n_tok, n_str)
            if hits == 0:
                # Nothing matched: the exclusion is untested, which is not evidence of correctness.
                return True, f"must_exclude_regex({c.value}) => 0 hits [{how}] VACUOUS", True
            return False, f"must_exclude_regex({c.value}) => {hits} violation(s)", False

        if c.kind == "token_count_range":
            toks = translation.split()
            min_t = c.min_tokens if c.min_tokens is not None else 0
            max_t = c.max_tokens if c.max_tokens is not None else 10**9
            ok = min_t <= len(toks) <= max_t
            return ok, f"token_count_range({min_t}, {max_t}) => {len(toks)}", False

        if c.kind in {"llm_check", "note"}:
            return True, f"Deferred check: {c.description}", True

        return False, f"Unknown constraint kind: {c.kind}", False

    def _llm_check(
        self,
        sentence: str,
        translation: str,
        rule: MBGCard,
        constraint: RuleConstraint,
    ) -> tuple[bool, str]:
        prompt = render_prompt(
            "validator_user.j2",
            sentence=sentence,
            translation=translation,
            rule_id=rule.id,
            rule_scope=rule.scope,
            constraint_description=constraint.description,
        )
        system_prompt = render_prompt("validator_system.j2")
        content, _usage = chat_text(
            endpoint=self.llm_endpoint,
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            api_key=self.api_key,
            timeout=120,
            max_tokens=self.max_tokens,
            temperature=0.0,
        )

        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            return False, f"LLM verifier invalid output: {content[:120]}"
        parsed = json.loads(match.group(0))
        ok = bool(parsed.get("passed", False))
        reason = str(parsed.get("reason", ""))
        return ok, f"llm_check => {ok}. {reason}"
