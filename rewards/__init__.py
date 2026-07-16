from importlib import import_module
import sys

from grammetarl.rewards import aggregate_reward


_SUBMODULE_ALIASES = {
	"rewards.aggregate": "grammetarl.rewards.aggregate",
	"rewards.grammar_reward": "grammetarl.rewards.grammar_reward",
	"rewards.mt_reward": "grammetarl.rewards.mt_reward",
	"rewards.tool_reward": "grammetarl.rewards.tool_reward",
}

for legacy_name, canonical_name in _SUBMODULE_ALIASES.items():
	sys.modules.setdefault(legacy_name, import_module(canonical_name))

__all__ = ["aggregate_reward"]
