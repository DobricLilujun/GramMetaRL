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
        # Optionality as a disjunction: any one alternative satisfies the rule. This is what a
        # wrong allomorph set should have been - the book's own example is `datt s du`, not `datts`.
        "must_include_any_regex",
        "token_count_range",
        # Semantics containment cannot express. Extraction was forcing these into must_include,
        # which guarantees a failure on correct translations:
        #   equivalence - "nominative and accusative forms are identical"
        #   optionality - "the particle may be omitted in this construction"
        # They abstain (no reward term) until a morphology engine can decide them (P3).
        "equivalence",
        "optionality",
        # The form a checkable translation rule must take: IF the source exhibits X the target must
        # realize Y. Applicability comes from the source side, so it does not depend on a separately
        # validated target-side predicate (measured precision of those: 0.25-0.43).
        "source_conditional",
        "llm_check",
        "note",
    ]
    description: str
    value: str | None = None
    min_tokens: int | None = None
    max_tokens: int | None = None
    source_pattern: str | None = Field(
        default=None,
        description="Regex over the SOURCE sentence for source_conditional constraints; when it does "
        "not match, the rule does not govern the pair and contributes no verdict",
    )
    values: list[str] = Field(
        default_factory=list,
        description="Alternatives for must_include_any_regex / optionality; at least one must match",
    )
    # What the pattern applies to. Without this field a word-level anchored pattern such as
    # ^(dat|hatt|et|seng)$ gets searched against a whole sentence, which can never match, so the
    # rule fails on every correct translation (measured: 81% constraint failure on real pairs).
    level: Literal["token", "phrase", "sentence", "any"] | None = None


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


class TriggerCondition(BaseModel):
    """Compiled applicability predicate: when is this rule even in play?

    Cards always state triggers in prose, and prose cannot be evaluated, so the verifier used to
    check output constraints on sentences the rule never governed (measured: 85.2% of checks on
    48k real en->lb pairs failed, mostly for that reason). A compiled predicate lets the
    verifier abstain instead.
    """

    source_text: str = Field(description="The prose trigger this was compiled from")
    kind: Literal["regex", "llm_check", "note"] = "note"
    side: Literal["source", "target"] = Field(
        default="target",
        description="Which side of the pair the predicate is observable on",
    )
    pattern: str | None = None
    level: Literal["token", "phrase", "sentence", "any"] = "sentence"
    negated: bool = False
    accepted: bool = Field(
        default=False,
        description="Self-test admission: the pattern fires on at least one of the card's own examples",
    )
    reject_reason: str | None = None


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
    applicability: list[TriggerCondition] = Field(
        default_factory=list,
        description="Executable subset of trigger_conditions; empty means applicability is unknown",
    )
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
