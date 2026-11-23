from __future__ import annotations

import json
import re
import hashlib
from dataclasses import dataclass
from typing import Any

from .llm_http import DEFAULT_MAX_TOKENS, chat_text
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


_NONE_TOKENS = {"", "none", "null", "n/a", "na", "nan", "-", "unknown"}

VALID_OPERATION_TYPES = {
    "reorder", "insert", "delete", "inflect", "agree", "mark_case", "negate",
    "tense_aspect", "mood", "clitic", "lexical_override", "phonotactics", "other",
    "fallback",
}

VALID_RULE_STATUSES = {"categorical", "default", "optional", "variable", "lexical_exception", "uncertain"}

VALID_EXCEPTION_TYPES = {"exception", "optional_variant", "dialectal_variant", "register_variant", "lexicalized_form"}


def _clean_str(value: object) -> str:
    """Stringify a value, collapsing None and sentinel strings to the empty string.

    Fixes the previous `str(None) == 'None'` leakage that put literal 'None' into `scope`.
    """
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return ""
    text = str(value).strip()
    if text.lower() in _NONE_TOKENS:
        return ""
    return text


def looks_like_prose(pattern: str) -> bool:
    """True when a 'pattern' string is really a natural-language sentence, not a matcher.

    The extraction prompt puts free text in the constraint `value` slot; feeding that to
    re.search makes every check fail, so `grammar_reward` becomes a constant. Reject those.
    """
    if not pattern:
        return True
    import re as _re

    try:
        _re.compile(pattern)
    except _re.error:
        return True
    # Observed in real extraction: 'same_suffix_for_definite_and_indefinite' - an identifier, not
    # a matcher. It has no whitespace so the bare-word test below misses it, and as a regex it can
    # only match that literal text, so it silently always fails (measured: this class accounts for
    # a large share of the 63.8% constraint-violation rate on professional human translations).
    if _re.fullmatch(r"[A-Za-z][A-Za-z0-9]*(?:[_\-][A-Za-z0-9]+)+", pattern.strip()):
        return True

    # No regex machinery at all AND several whitespace-separated tokens => an English clause was
    # pasted into the pattern slot. The machinery test is what keeps legitimate literal patterns
    # alive: must_not_include 'zur der' (a cliticization rule) and 'datt s du' (the book's own
    # clitic example) are plain words and must NOT be rejected.
    machinery = any(tok in pattern for tok in ("\\b", "\\s", "\\w", "\\d", "^", "$", "[", "]", "(", ")", "|", "?", "*", "+", "{", "."))
    if machinery:
        return False
    return len(pattern.split()) >= 4


class MBGExtractor:
    def __init__(
        self,
        endpoint: str,
        model: str,
        api_key: str | None = None,
        timeout: int = 240,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        reasoning_effort: str | None = None,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort

    def extract_cards(self, chunk: ExtractionChunk) -> list[MBGCard]:
        user_prompt = self._build_user_prompt(chunk)
        language_code, language_name = self._resolve_language(chunk.language)
        system_prompt = render_prompt(
            MBG_SYSTEM_PROMPT_TEMPLATE,
            language_code=language_code,
            language_name=language_name,
        )
        content, meta = chat_text(
            endpoint=self.endpoint,
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            api_key=self.api_key,
            timeout=self.timeout,
            max_tokens=self.max_tokens,
            temperature=0.1,
            reasoning_effort=self.reasoning_effort,
        )
        if meta.get("finish_reason") == "length":
            print(
                f"[MBG][WARN] completion truncated at max_tokens={self.max_tokens} "
                f"(section={chunk.section_id}); salvaging complete cards",
                flush=True,
            )
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

        # --- scope: accept rich dict OR string; never silently drop; never leak 'None' ---
        scope_val = normalized.get("scope")
        if isinstance(scope_val, dict):
            scope_parts = [
                _clean_str(scope_val.get(k))
                for k in ("syntactic_domain", "constituent_target", "linear_domain", "semantic_pragmatic_scope")
            ]
            scope_parts = [p for p in scope_parts if p]
            normalized["scope"] = " | ".join(scope_parts) if scope_parts else "unspecified"
        else:
            normalized["scope"] = _clean_str(scope_val) or "unspecified"

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
        valid_kinds = {
            "must_include_regex",
            "must_exclude_regex",
            "must_include_any_regex",
            "token_count_range",
            "equivalence",
            "optionality",
            "llm_check",
            "note",
        }
        # Legacy / prompt-vocabulary names that must not silently degrade into llm_check.
        kind_aliases = {"must_not_include_regex": "must_exclude_regex", "forbidden_shape": "must_exclude_regex"}
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
            elif kind in kind_aliases:
                c2["kind"] = kind_aliases[kind]
            elif kind in valid_kinds:
                c2["kind"] = kind
            else:  # grammaticality, position, cooccurrence, or unknown
                c2["kind"] = "llm_check"
            if not str(c2.get("description", "")).strip():
                c2["description"] = desc or f"{c2['kind']} check"
            if "value" in c2 and c2["value"] is not None:
                c2["value"] = str(c2["value"])
            has_alternatives = bool(c2.get("values")) and c2.get("kind") == "must_include_any_regex"
            if c2.get("kind") in ("must_include_regex", "must_exclude_regex") and not has_alternatives and not str(
                c2.get("value", "")
            ).strip():
                # no usable pattern; keep as a deferred llm_check so the verifier still has something to inspect
                c2["kind"] = "llm_check"
            if c2.get("kind") in ("must_include_regex", "must_exclude_regex") and looks_like_prose(
                str(c2.get("value", ""))
            ):
                # A prose value in a regex slot matches nothing (or matches by accident).
                # Downgrade instead of emitting a fake machine check: the reward must not be noise.
                desc = str(c2.get("description", "")).strip()
                c2["kind"] = "llm_check"
                c2["description"] = f"{desc or str(c2.get('value', '')).strip()}"
                c2["value"] = None
            norm_constraints.append(c2)
        normalized["output_constraints"] = norm_constraints

        # --- examples: map rich prompt example shape -> schema shape (keep all fields) ---
        # Orientation rule for a target-language grammar card: the book's example sentence IS
        # the target-language string, and `translation` (if given) is the English/source side.
        # Feeding {sentence -> source_sentence} makes verifiers check the constraint against the
        # wrong (English) side, so the pairing must be sentence=target, translation=source.
        card_language = _clean_str(normalized.get("language")).lower()
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
            book_sentence = _clean_str(ex2.get("sentence")) or _clean_str(ex2.get("target_sentence"))
            en_side = _clean_str(ex2.get("translation")) or _clean_str(ex2.get("source_sentence"))
            gloss = _clean_str(ex2.get("gloss"))
            if book_sentence and not en_side:
                # No source-language side available: keep the target string, leave src empty
                # rather than duplicating it (an identical src/tgt pair is degenerate).
                en_side = ""
            if en_side and en_side == book_sentence:
                # The model duplicated the book sentence into both slots: not a translation pair.
                en_side = ""
            ex2["source_sentence"] = en_side
            ex2["target_sentence"] = book_sentence
            ex2["gloss"] = gloss or None
            ex2.setdefault("note", None)
            if not ex2["target_sentence"]:
                continue
            norm_examples.append(ex2)
        normalized["examples"] = norm_examples
        del card_language  # orientation is language-independent given `translation`; kept for clarity

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

        # --- statement / confidence / rule_status / exceptions_and_variants ---
        # These were requested by the prompt but silently dropped by the schema, so the single
        # most informative field (the rule statement) never reached retrieval or the agent.
        normalized["statement"] = _clean_str(normalized.get("statement"))
        if "confidence" in normalized:
            try:
                conf = float(normalized["confidence"])
                normalized["confidence"] = max(0.0, min(1.0, conf))
            except (TypeError, ValueError):
                normalized["confidence"] = None
        status = _clean_str(normalized.get("rule_status")).lower()
        normalized["rule_status"] = status if status in VALID_RULE_STATUSES else None

        variants = normalized.get("exceptions_and_variants", [])
        if isinstance(variants, dict):
            variants = [variants]
        elif not isinstance(variants, list):
            variants = []
        norm_variants: list[dict[str, Any]] = []
        for var in variants:
            if isinstance(var, str):
                norm_variants.append({"type": "exception", "description": var})
                continue
            if not isinstance(var, dict):
                continue
            desc = _clean_str(var.get("description")) or _clean_str(var.get("statement"))
            if not desc:
                continue
            vtype = _clean_str(var.get("type")).lower()
            norm_variants.append(
                {
                    "type": vtype if vtype in VALID_EXCEPTION_TYPES else "exception",
                    "description": desc,
                    "condition": _clean_str(var.get("condition")) or None,
                    "preferred_form": _clean_str(var.get("preferred_form")) or None,
                }
            )
        normalized["exceptions_and_variants"] = norm_variants

        op_val = _clean_str(normalized.get("operation_type")).lower()
        normalized["operation_type"] = op_val if op_val in VALID_OPERATION_TYPES else "fallback"

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
        elif op == "phonotactics":
            tags.append("phonotactics")

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
        """Parse the card array, salvaging every COMPLETE object from a truncated response.

        A thinking model can spend the token budget on reasoning and then emit a long array that
        gets cut mid-object. Raising on the whole section used to discard the good cards with it.
        """
        stripped = raw_text.strip()
        text = stripped if stripped.startswith("[") else ""
        if not text:
            # Recovery: some models wrap JSON in markdown fences or prose.
            match = re.search(r"\[.*", stripped, flags=re.DOTALL)
            text = match.group(0) if match else ""
        if not text:
            obj_match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
            if not obj_match:
                raise ValueError("No JSON array or object found in model response")
            text = "[" + obj_match.group(0) + "]"

        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return [parsed]
            if isinstance(parsed, list):
                return [p for p in parsed if isinstance(p, dict)]
        except json.JSONDecodeError:
            pass

        decoder = json.JSONDecoder()
        i = text.find("[") + 1
        salvaged: list[dict[str, Any]] = []
        while i < len(text):
            while i < len(text) and text[i] in " \n\r\t,":
                i += 1
            if i >= len(text) or text[i] != "{":
                break
            try:
                obj, end = decoder.raw_decode(text, i)
            except json.JSONDecodeError:
                break
            if isinstance(obj, dict):
                salvaged.append(obj)
            i = end
        if not salvaged:
            raise ValueError(
                "No complete card object could be parsed (response truncated before the first "
                "closing brace?)"
            )
        print(f"[MBG][WARN] salvaged {len(salvaged)} complete card(s) from a malformed/truncated response",
              flush=True)
        return salvaged
        if not isinstance(parsed, list):
            raise ValueError("Recovered payload is not a list")
        return parsed
