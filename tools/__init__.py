from importlib import import_module
import sys

from grammetarl.tools import dictionary_lookup, grammar_rule_apply, grammar_rule_search, grammar_verify


_SUBMODULE_ALIASES = {
    "tools.dictionary_lookup": "grammetarl.tools.dictionary_lookup",
    "tools.grammar_apply": "grammetarl.tools.grammar_apply",
    "tools.grammar_search": "grammetarl.tools.grammar_search",
    "tools.grammar_verify": "grammetarl.tools.grammar_verify",
}

for legacy_name, canonical_name in _SUBMODULE_ALIASES.items():
    sys.modules.setdefault(legacy_name, import_module(canonical_name))

__all__ = [
    "grammar_rule_search",
    "dictionary_lookup",
    "grammar_rule_apply",
    "grammar_verify",
]
