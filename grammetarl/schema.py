from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


OperationType = Literal[
    "reorder",
    "insert",
    "delete",
    "inflect",
    "agree",
    "mark_case",
    "negate",
    "tense_aspect",
    "mood",
    "clitic",
    "lexical_override",
    "phonotactics",
    "other",
    "fallback",
]


RuleStatus = Literal[
    "categorical",
    "default",
    "optional",
    "variable",
    "lexical_exception",
    "uncertain",
]

ExceptionType = Literal[
    "exception",
    "optional_variant",
    "dialectal_variant",
    "register_variant",
    "lexicalized_form",
]


class RuleExceptionVariant(BaseModel):
    type: ExceptionType = "exception"
    description: str
    condition: str | None = None
    preferred_form: str | None = None


class RuleConstraint(BaseModel):
    kind: Literal[
        "must_include_regex",
        "must_exclude_regex",
        "token_count_range",
        "llm_check",
        "note",
    ]
    description: str
    value: str | None = None
    min_tokens: int | None = None
    max_tokens: int | None = None


class RuleExample(BaseModel):
    source_sentence: str
    target_sentence: str
    gloss: str | None = None
    note: str | None = None


class SourceInfo(BaseModel):
    source_id: str
    page_start: int
    page_end: int
    section_title: str | None = None
    excerpt: str | None = None


class RuleDependency(BaseModel):
    rule_id: str
    relation: Literal["requires", "overrides", "incompatible_with", "preferred_with"]
    note: str | None = None


class MBGCard(BaseModel):
    id: str = Field(description="Unique and stable rule ID")
    language: str = Field(description="Language identifier, e.g. lb, kmr, qu")
    statement: str = Field(
        default="",
        description="Concise falsifiable statement of the rule, in English",
    )
    rule_status: RuleStatus | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    phenomenon_tags: list[str] = Field(default_factory=list)
    trigger_conditions: list[str] = Field(default_factory=list)
    scope: str = Field(description="What structure this rule applies to")
    operation_type: OperationType
    operation_steps: list[str] = Field(default_factory=list)
    output_constraints: list[RuleConstraint] = Field(default_factory=list)
    exceptions_and_variants: list[RuleExceptionVariant] = Field(default_factory=list)
    examples: list[RuleExample] = Field(default_factory=list)
    source: SourceInfo
    priority: int = Field(default=50, ge=0, le=100)
    dependencies: list[RuleDependency] = Field(default_factory=list)
    retrieval_hints: list[str] = Field(default_factory=list)
    verifier_hints: list[str] = Field(default_factory=list)

    def as_index_text(self) -> str:
        parts = [
            self.id,
            self.statement,
            self.scope,
            " ".join(self.phenomenon_tags),
            " ".join(self.trigger_conditions),
            " ".join(self.operation_steps),
            " ".join(self.retrieval_hints),
            " ".join(e.description for e in self.exceptions_and_variants),
        ]
        return "\n".join(p for p in parts if p)

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
