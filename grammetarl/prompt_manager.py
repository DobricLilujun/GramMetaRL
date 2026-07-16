from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from jinja2 import Environment, FileSystemLoader


PACKAGE_TEMPLATE_ROOT = Path(__file__).resolve().parent / "prompts" / "jinja"
LEGACY_TEMPLATE_ROOT = Path(__file__).resolve().parents[1] / "prompts" / "jinja"


@lru_cache(maxsize=1)
def _get_env() -> Environment:
    template_root = PACKAGE_TEMPLATE_ROOT if PACKAGE_TEMPLATE_ROOT.exists() else LEGACY_TEMPLATE_ROOT
    if not template_root.exists():
        raise FileNotFoundError(
            f"Prompt template directory not found: {PACKAGE_TEMPLATE_ROOT} or {LEGACY_TEMPLATE_ROOT}"
        )
    return Environment(
        loader=FileSystemLoader(str(template_root)),
        autoescape=False,
        trim_blocks=False,
        lstrip_blocks=False,
        keep_trailing_newline=True,
    )


def render_prompt(template_name: str, **context: object) -> str:
    template = _get_env().get_template(template_name)
    return template.render(**context).strip()
