from __future__ import annotations

import json
import re
from dataclasses import dataclass

import requests

from .prompt_manager import render_prompt
from .schema import MBGCard, RuleConstraint


@dataclass(slots=True)
class RuleCheckResult:
    rule_id: str
    passed: bool
    messages: list[str]


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
    ) -> None:
        self.llm_endpoint = llm_endpoint.rstrip("/") if llm_endpoint else None
        self.llm_model = llm_model
        self.api_key = api_key

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

        for cons in rule.output_constraints:
            ok, msg = self._check_constraint(cons, translation)
            messages.append(msg)
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

        return RuleCheckResult(rule_id=rule.id, passed=passed, messages=messages)

    def _check_constraint(self, c: RuleConstraint, translation: str) -> tuple[bool, str]:
        if c.kind == "must_include_regex":
            if not c.value:
                return False, "must_include_regex missing value"
            ok = re.search(c.value, translation) is not None
            return ok, f"must_include_regex({c.value}) => {ok}"

        if c.kind == "must_exclude_regex":
            if not c.value:
                return False, "must_exclude_regex missing value"
            ok = re.search(c.value, translation) is None
            return ok, f"must_exclude_regex({c.value}) => {ok}"

        if c.kind == "token_count_range":
            toks = translation.split()
            min_t = c.min_tokens if c.min_tokens is not None else 0
            max_t = c.max_tokens if c.max_tokens is not None else 10**9
            ok = min_t <= len(toks) <= max_t
            return ok, f"token_count_range({min_t}, {max_t}) => {len(toks)}"

        if c.kind in {"llm_check", "note"}:
            return True, f"Deferred check: {c.description}"

        return False, f"Unknown constraint kind: {c.kind}"

    def _llm_check(
        self,
        sentence: str,
        translation: str,
        rule: MBGCard,
        constraint: RuleConstraint,
    ) -> tuple[bool, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        prompt = render_prompt(
            "validator_user.j2",
            sentence=sentence,
            translation=translation,
            rule_id=rule.id,
            rule_scope=rule.scope,
            constraint_description=constraint.description,
        )
        system_prompt = render_prompt("validator_system.j2")
        payload = {
            "model": self.llm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
        }
        r = requests.post(
            f"{self.llm_endpoint}/chat/completions",
            headers=headers,
            data=json.dumps(payload),
            timeout=120,
        )
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]

        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            return False, f"LLM verifier invalid output: {content[:120]}"
        parsed = json.loads(match.group(0))
        ok = bool(parsed.get("passed", False))
        reason = str(parsed.get("reason", ""))
        return ok, f"llm_check => {ok}. {reason}"
