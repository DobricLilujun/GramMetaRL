from __future__ import annotations

import json
from pathlib import Path

from grammetarl import load_cards
from grammetarl.llm_http import chat_text
from grammetarl.workspace import get_workspace_paths

APPLY_SYSTEM = """You are a Luxembourgish translation worker with a rule bundle in hand.

You get an English source sentence and the grammar rules retrieved for it. Produce one
Luxembourgish translation that obeys the bundle. Then list which rules you actually applied.

Return strict JSON:
{"translation": "<Luxembourgish only>", "rules_applied": ["<rule id>", ...]}

Rules of engagement:
- Never echo the English source.
- Apply the constraints literally (word order, inflection, articles); if two rules conflict,
  prefer the higher-priority one and say so in rules_applied order.
- If a rule is not applicable to this sentence, do not force it."""


def _card_bundle(cards) -> list[dict]:
    bundle = []
    for c in cards:
        bundle.append(
            {
                "id": c.id,
                "statement": c.statement,
                "scope": c.scope,
                "operation_type": c.operation_type,
                "priority": c.priority,
                "steps": c.operation_steps[:3],
                "constraints": [
                    {"kind": x.kind, "level": x.level, "value": x.value}
                    for x in c.output_constraints
                    if x.kind in ("must_include_regex", "must_exclude_regex", "token_count_range")
                ],
                "examples": [
                    {"src": e.source_sentence, "tgt": e.target_sentence} for e in c.examples[:2]
                ],
            }
        )
    return bundle


def grammar_rule_apply(
    rule_ids: list[str],
    src_sentence: str,
    lexical_hints: dict[str, str] | None = None,
    cards_path: str | None = None,
    namespace: str | None = None,
    llm_endpoint: str | None = None,
    llm_model: str | None = None,
    api_key: str | None = None,
    timeout: int = 240,
    max_tokens: int = 1024,
) -> dict[str, object]:
    """Return the retrieved rule bundle, and a draft only if a policy model is supplied.

    The previous version returned `draft = src_sentence`, i.e. a fabricated English draft dressed
    up as a translation. That is what made the eval replay produce chrF 0.13 against a reference
    and what let G10/G13 stay red: the tool is not the policy, so without an explicit model it must
    return `draft=None` and let the agent produce the translation.
    """
    if not src_sentence.strip():
        raise ValueError("src_sentence cannot be empty")
    if not rule_ids:
        return {"draft": None, "applied": [], "rule_bundle": [], "notes": ["no rules requested"]}

    resolved_cards_path = (
        Path(cards_path) if cards_path else get_workspace_paths(namespace=namespace).rules_cards_file
    )
    cards = {c.id: c for c in load_cards(resolved_cards_path)}
    selected = [cards[rid] for rid in rule_ids if rid in cards]
    notes = [f"missing rule: {rid}" for rid in rule_ids if rid not in cards]
    if lexical_hints:
        notes.append(f"lexical_hints={len(lexical_hints)}")

    bundle = _card_bundle(selected)
    out: dict[str, object] = {
        "draft": None,
        "applied": [c.id for c in selected],
        "rule_bundle": bundle,
        "notes": notes,
    }
    if not bundle:
        return out

    if not (llm_endpoint and llm_model):
        out["notes"].append("no policy model configured: draft left empty on purpose")
        return out

    user = (
        f"Source (en): {src_sentence}\n"
        f"Rule bundle:\n{json.dumps(bundle, ensure_ascii=False)}\n"
    )
    if lexical_hints:
        user += f"Lexical hints: {json.dumps(lexical_hints, ensure_ascii=False)}\n"
    try:
        text, meta = chat_text(
            endpoint=llm_endpoint,
            model=llm_model,
            messages=[{"role": "system", "content": APPLY_SYSTEM}, {"role": "user", "content": user}],
            api_key=api_key,
            timeout=timeout,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        out["notes"].append(f"policy call failed: {exc}")
        return out

    translation = _parse_translation(text)
    if translation and translation.strip().lower() != src_sentence.strip().lower():
        out["draft"] = translation.strip()
        out["notes"].append(f"finish_reason={meta.get('finish_reason')}")
    else:
        out["notes"].append("policy returned empty or an echo of the source; draft left empty")
    return out


def _parse_translation(text: str) -> str:
    text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            t = obj.get("translation")
            if isinstance(t, str) and t.strip():
                return t
        except json.JSONDecodeError:
            pass
    # Fall back to the longest non-JSON line, which is what chatty models often emit.
    lines = [ln.strip(" \"'`*") for ln in text.splitlines() if ln.strip()]
    return max(lines, key=len) if lines else ""
