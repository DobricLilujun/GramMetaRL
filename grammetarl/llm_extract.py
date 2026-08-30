from __future__ import annotations

import json
import re
import hashlib
from dataclasses import dataclass
from typing import Any

import requests

from .prompt_manager import render_prompt
from .schema import MBGCard


@dataclass(slots=True)
class ExtractionChunk:
    language: str
    source_id: str
    section_id: str
    section_title: str
    page_start: int
    page_end: int
    text: str


MBG_SYSTEM_PROMPT_TEMPLATE = "mbg_system.j2"
MBG_USER_PROMPT_TEMPLATE = "mbg_user.j2"


LANGUAGE_CODE_TO_NAME = {
    "lb": "Luxembourgish",
    "en": "English",
    "de": "German",
    "fr": "French",
}


class MBGExtractor:
    def __init__(
        self,
        endpoint: str,
        model: str,
        api_key: str | None = None,
        timeout: int = 240,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def extract_cards(self, chunk: ExtractionChunk) -> list[MBGCard]:
        user_prompt = self._build_user_prompt(chunk)
        language_code, language_name = self._resolve_language(chunk.language)
        system_prompt = render_prompt(
            MBG_SYSTEM_PROMPT_TEMPLATE,
            language_code=language_code,
            language_name=language_name,
        )
        payload = {
            "model": self.model,
            "temperature": 0.1,
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
        cards_raw = self._extract_json(content)

        cards: list[MBGCard] = []
        for obj in cards_raw:
            obj = self._normalize_card_obj(obj)
            if "language" not in obj:
                obj["language"] = chunk.language
            if "source" not in obj:
                obj["source"] = {
                    "source_id": chunk.source_id,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "section_title": chunk.section_title,
                    "excerpt": chunk.text[:400],
                }
            cards.append(MBGCard.model_validate(obj))
        return cards

    def _normalize_card_obj(self, obj: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(obj)

        if not str(normalized.get("id", "")).strip():
            for key in ["rule_id", "rule_name", "statement", "formal_pattern", "trigger_condition"]:
                value = str(normalized.get(key, "")).strip()
                if value:
                    normalized["id"] = f"AUTO_{hashlib.sha1(value.encode('utf-8')).hexdigest()[:12]}"
                    break
        if not str(normalized.get("id", "")).strip():
            serialized = json.dumps(normalized, ensure_ascii=False, sort_keys=True)
            normalized["id"] = f"AUTO_{hashlib.sha1(serialized.encode('utf-8')).hexdigest()[:12]}"

        # --- scope: accept rich dict OR string; never silently drop ---
        scope_val = normalized.get("scope")
        if isinstance(scope_val, dict):
            syntactic_domain = str(scope_val.get("syntactic_domain", "")).strip()
            constituent_target = str(scope_val.get("constituent_target", "")).strip()
            linear_domain = str(scope_val.get("linear_domain", "")).strip()
            semantic_scope = str(scope_val.get("semantic_pragmatic_scope", "")).strip()
            scope_parts = [p for p in [syntactic_domain, constituent_target, linear_domain, semantic_scope] if p]
            normalized["scope"] = " | ".join(scope_parts) if scope_parts else "unspecified"
        elif scope_val is None:
            normalized["scope"] = "unspecified"
        else:
            normalized["scope"] = str(scope_val).strip() or "unspecified"

        # --- trigger_conditions: accept list-of-objects OR list-of-strings ---
        raw_triggers = normalized.get("trigger_conditions")
        if raw_triggers is None and "trigger_condition" in normalized:
            raw_triggers = normalized.get("trigger_condition")
        trig_strs: list[str] = []
        for item in raw_triggers if isinstance(raw_triggers, list) else [raw_triggers]:
            if item is None:
                continue
            if isinstance(item, dict):
                cond = str(item.get("condition", "")).strip()
                if cond:
                    trig_strs.append(cond)
            elif isinstance(item, str) and item.strip():
                trig_strs.append(item.strip())
        normalized["trigger_conditions"] = trig_strs

        # --- operation_steps: accept list-of-objects OR list-of-strings ---
        raw_steps = normalized.get("operation_steps")
        if raw_steps is None and "operation_step" in normalized:
            raw_steps = normalized.get("operation_step")
        step_strs: list[str] = []
        for item in raw_steps if isinstance(raw_steps, list) else [raw_steps]:
            if item is None:
                continue
            if isinstance(item, dict):
                picked = ""
                for k in ("step", "action", "text", "description"):
                    v = str(item.get(k, "")).strip()
                    if v:
                        picked = v
                        break
                if picked:
                    step_strs.append(picked)
            elif isinstance(item, str) and item.strip():
                step_strs.append(item.strip())
        normalized["operation_steps"] = step_strs

        for key in ["phenomenon_tags", "retrieval_hints", "verifier_hints"]:
            val = normalized.get(key)
            if val is None:
                normalized[key] = []
            elif isinstance(val, str):
                normalized[key] = [val]
            elif isinstance(val, list):
                normalized[key] = [str(v).strip() for v in val if str(v).strip()]
            else:
                normalized[key] = [str(val)]

        # --- output_constraints: map rich prompt kinds -> schema kinds WITHOUT losing them ---
        constraints = normalized.get("output_constraints", [])
        if isinstance(constraints, dict):
            constraints = [constraints]
        elif isinstance(constraints, str):
            constraints = [constraints]
        elif not isinstance(constraints, list):
            constraints = []
        valid_kinds = {"must_include_regex", "must_exclude_regex", "token_count_range", "llm_check", "note"}
        norm_constraints: list[dict[str, Any]] = []
        for c in constraints:
            if isinstance(c, str):
                norm_constraints.append({"kind": "note", "description": c})
                continue
            if not isinstance(c, dict):
                continue
            c2 = dict(c)
            c2.pop("rule_id", None)
            kind = str(c2.get("kind", "")).strip().lower()
            desc = str(c2.get("description", "")).strip()
            value = c2.get("value")
            if value is not None:
                value = str(value)
            # rich kind -> machine-checkable schema kind
            if kind == "required_order":
                c2["kind"] = "must_include_regex"; c2["value"] = value or desc
            elif kind in ("required_form", "agreement", "case"):
                c2["kind"] = "must_include_regex"; c2["value"] = value or desc
            elif kind == "forbidden_form":
                c2["kind"] = "must_exclude_regex"; c2["value"] = value or desc
            elif kind == "count":
                c2["kind"] = "token_count_range"
            elif kind in valid_kinds:
                c2["kind"] = kind
            else:  # grammaticality, position, cooccurrence, or unknown
                c2["kind"] = "llm_check"
            if not str(c2.get("description", "")).strip():
                c2["description"] = desc or f"{c2['kind']} check"
            if "value" in c2 and c2["value"] is not None:
                c2["value"] = str(c2["value"])
            if c2.get("kind") in ("must_include_regex", "must_exclude_regex") and not str(c2.get("value", "")).strip():
                # no usable pattern; keep as a deferred llm_check so the verifier still has something to inspect
                c2["kind"] = "llm_check"
            norm_constraints.append(c2)
        normalized["output_constraints"] = norm_constraints

        # --- examples: map rich prompt example shape -> schema shape (keep all fields) ---
        examples = normalized.get("examples", [])
        if isinstance(examples, dict):
            examples = [examples]
        elif not isinstance(examples, list):
            examples = []
        norm_examples: list[dict[str, Any]] = []
        for ex in examples:
            if not isinstance(ex, dict):
                continue
            ex2 = dict(ex)
            ex2["source_sentence"] = str(
                ex2.get("source_sentence") or ex2.get("sentence") or ex2.get("src") or ""
            ).strip()
            ex2["target_sentence"] = str(
                ex2.get("target_sentence") or ex2.get("translation") or ex2.get("gloss") or ""
            ).strip()
            if ex2.get("gloss") is None and ex2.get("source_sentence"):
                ex2["gloss"] = None
            ex2.setdefault("note", None)
            norm_examples.append(ex2)
        normalized["examples"] = norm_examples

        # --- dependencies: accept target_rule_id + relation; drop rule_status noise ---
        deps = normalized.get("dependencies", [])
        if isinstance(deps, dict):
            deps = [deps]
        elif not isinstance(deps, list):
            deps = []
        valid_relations = {"requires", "overrides", "incompatible_with", "preferred_with"}
        norm_deps: list[dict[str, Any]] = []
        for dep in deps:
            if not isinstance(dep, dict):
                continue
            dep2 = dict(dep)
            rule_id = str(dep2.get("rule_id") or dep2.get("target_rule_id") or "").strip()
            relation = str(dep2.get("relation", "")).strip()
            if not rule_id or relation not in valid_relations:
                continue
            dep2["rule_id"] = rule_id
            dep2["relation"] = relation
            if "note" in dep2 and dep2["note"] is None:
                dep2["note"] = str(dep2.get("description", ""))
            norm_deps.append(dep2)
        normalized["dependencies"] = norm_deps

        if "priority" in normalized:
            try:
                normalized["priority"] = int(normalized["priority"])
            except Exception:
                normalized["priority"] = 50

        valid_operation_types = {
            "reorder", "insert", "delete", "inflect", "agree", "mark_case",
            "negate", "tense_aspect", "mood", "clitic", "lexical_override", "fallback",
        }
        op_val = str(normalized.get("operation_type", "")).strip().lower()
        normalized["operation_type"] = op_val if op_val in valid_operation_types else "fallback"

        # --- inferred hints (only fill when empty; never overwrite real content) ---
        if not normalized.get("phenomenon_tags"):
            normalized["phenomenon_tags"] = self._infer_phenomenon_tags(normalized)
        if not normalized.get("retrieval_hints"):
            normalized["retrieval_hints"] = self._infer_retrieval_hints(normalized)
        if not normalized.get("verifier_hints"):
            normalized["verifier_hints"] = self._infer_verifier_hints(normalized)

        return normalized

    def _infer_phenomenon_tags(self, obj: dict[str, Any]) -> list[str]:
        text_blob = " ".join(
            [
                str(obj.get("scope", "")),
                " ".join(obj.get("trigger_conditions", []) or []),
                " ".join(obj.get("operation_steps", []) or []),
                str(obj.get("operation_type", "")),
            ]
        ).lower()
        tags: list[str] = []

        if "preposition" in text_blob:
            tags.append("preposition_governance")
        if "dative" in text_blob or "accusative" in text_blob or "case" in text_blob:
            tags.append("case_marking")
        if "plural" in text_blob or "singular" in text_blob or "inflect" in text_blob:
            tags.append("inflection")
        if "order" in text_blob or "before" in text_blob or "after" in text_blob:
            tags.append("word_order")
        if "clitic" in text_blob:
            tags.append("cliticization")

        op = str(obj.get("operation_type", "")).strip().lower()
        if op == "agree":
            tags.append("agreement")
        elif op == "negate":
            tags.append("negation")
        elif op == "tense_aspect":
            tags.append("tense_aspect")
        elif op == "mood":
            tags.append("mood")

        if not tags:
            tags.append("grammar_rule")
        # Keep deterministic order and small size.
        dedup: list[str] = []
        for t in tags:
            if t not in dedup:
                dedup.append(t)
        return dedup[:4]

    def _infer_retrieval_hints(self, obj: dict[str, Any]) -> list[str]:
        hints: list[str] = []
        for cond in obj.get("trigger_conditions", []) or []:
            c = str(cond).strip()
            if c:
                hints.append(c[:120])
            if len(hints) >= 2:
                break
        op = str(obj.get("operation_type", "")).strip().lower()
        if op and len(hints) < 2:
            hints.append(f"operation_type={op}")
        return hints

    def _infer_verifier_hints(self, obj: dict[str, Any]) -> list[str]:
        hints: list[str] = []
        constraints = obj.get("output_constraints", []) or []
        if isinstance(constraints, list):
            for c in constraints:
                if isinstance(c, dict):
                    desc = str(c.get("description", "")).strip()
                    if desc:
                        hints.append(f"Check: {desc[:140]}")
                if len(hints) >= 2:
                    break
        if not hints:
            trig = (obj.get("trigger_conditions", []) or [""])[0]
            trig_s = str(trig).strip()
            if trig_s:
                hints.append(f"Verify rule applies only when: {trig_s[:140]}")
        return hints[:2]

    def _build_user_prompt(self, chunk: ExtractionChunk) -> str:
        language_code, language_name = self._resolve_language(chunk.language)
        return render_prompt(
            MBG_USER_PROMPT_TEMPLATE,
            language=language_code,
            language_name=language_name,
            source_id=chunk.source_id,
            section_id=chunk.section_id,
            section_title=chunk.section_title,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            text=chunk.text,
        )

    def _resolve_language(self, language: str) -> tuple[str, str]:
        code = (language or "").strip().lower()
        if not code:
            return "unknown", "Unknown"
        return code, LANGUAGE_CODE_TO_NAME.get(code, code)

    def _extract_json(self, raw_text: str) -> list[dict[str, Any]]:
        stripped = raw_text.strip()
        if stripped.startswith("["):
            parsed = json.loads(stripped)
            if not isinstance(parsed, list):
                raise ValueError("Model output is not a JSON array")
            return parsed

        # Recovery: some models wrap JSON in markdown fences.
        match = re.search(r"\[.*\]", stripped, flags=re.DOTALL)
        if not match:
            raise ValueError("No JSON array found in model response")
        parsed = json.loads(match.group(0))
        if not isinstance(parsed, list):
            raise ValueError("Recovered payload is not a list")
        return parsed
