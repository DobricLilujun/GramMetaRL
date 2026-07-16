from importlib import import_module
import sys

from grammetarl.agent_env import AgentAction, AgentState, GrammarFirstEnv


_SUBMODULE_ALIASES = {
	"agent_env.env": "grammetarl.agent_env.env",
	"agent_env.state": "grammetarl.agent_env.state",
	"agent_env.action_schema": "grammetarl.agent_env.action_schema",
}

for legacy_name, canonical_name in _SUBMODULE_ALIASES.items():
	sys.modules.setdefault(legacy_name, import_module(canonical_name))

__all__ = ["GrammarFirstEnv", "AgentState", "AgentAction"]
