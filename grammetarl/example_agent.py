from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from pydantic import BaseModel, Field

from .llm_extract import LANGUAGE_CODE_TO_NAME
from .prompt_manager import render_prompt
from .schema import MBGCard


EXAMPLE_AGENT_SYSTEM_PROMPT_TEMPLATE = "example_agent_system.j2"
EXAMPLE_AGENT_USER_PROMPT_TEMPLATE = "example_agent_user.j2"


class ExampleCheckResult(BaseModel):
    is_valid: bool
    grammar_alignment: bool
    semantic_alignment: bool
    teaching_value: bool


class ExampleAgentResult(BaseModel):
    language_pair: str
    english_source: str
    wrong_expression: str
    correct_expression: str
    check_result: ExampleCheckResult
    confidence: int = Field(ge=1, le=10)
    check_notes: str


class ExampleAgentRecord(BaseModel):
    rule_id: str
    language: str
    source_id: str
    section_title: str | None = None
    page_start: int
    page_end: int
    source_excerpt: str
    extracted_grammar: dict[str, Any]
    example: ExampleAgentResult

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


@dataclass(slots=True)
class ExampleAgent:
    endpoint: str
    model: str
    api_key: str | None = None
    timeout: int = 240
    min_confidence: int = 7

    def __post_init__(self) -> None:
        self.endpoint = self.endpoint.rstrip("/")

    def _resolve_language(self, language: str) -> tuple[str, str]:
        code = (language or "").strip().lower()
        if not code:
            return "unknown", "Unknown"
        return code, LANGUAGE_CODE_TO_NAME.get(code, code)

    def generate_record(
        self,
        card: MBGCard,
        source_text: str,
    ) -> ExampleAgentRecord | None:
        language_code, language_name = self._resolve_language(card.language)
        system_prompt = render_prompt(
            EXAMPLE_AGENT_SYSTEM_PROMPT_TEMPLATE,
            language_code=language_code,
            language_name=language_name,
        )
        user_prompt = render_prompt(
            EXAMPLE_AGENT_USER_PROMPT_TEMPLATE,
            language_code=language_code,
            language_name=language_name,
            extracted_grammar=json.dumps(card.as_dict(), ensure_ascii=False, indent=2),
            source_text=source_text,
        )
        payload = {
            "model": self.model,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        r = requests.post(
            f"{self.endpoint}/chat/completions",
            headers=headers,
            data=json.dumps(payload),
            timeout=self.timeout,
        )
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        parsed = self._extract_json_or_null(content)
        if parsed is None:
            return None

        try:
            result = ExampleAgentResult.model_validate(parsed)
        except Exception:
            return None

        if result.confidence < self.min_confidence:
            return None
        if not result.check_result.is_valid:
            return None
        if not result.check_result.grammar_alignment:
            return None
        if not result.check_result.semantic_alignment:
            return None
        if not result.check_result.teaching_value:
            return None

        return ExampleAgentRecord(
            rule_id=card.id,
            language=card.language,
            source_id=card.source.source_id,
            section_title=card.source.section_title,
            page_start=card.source.page_start,
            page_end=card.source.page_end,
            source_excerpt=card.source.excerpt or source_text[:500],
            extracted_grammar=card.as_dict(),
            example=result,
        )

    def _extract_json_or_null(self, raw_text: str) -> dict[str, Any] | None:
        stripped = raw_text.strip()
        if stripped.lower() == "null":
            return None
        if stripped.startswith("{"):
            parsed = json.loads(stripped)
            if not isinstance(parsed, dict):
                return None
            return parsed

        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            if "null" in stripped.lower():
                return None
            return None
        parsed = json.loads(match.group(0))
        if not isinstance(parsed, dict):
            return None
        return parsed
