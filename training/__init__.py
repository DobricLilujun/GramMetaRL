from importlib import import_module
import sys

from grammetarl.training import build_min_dataset


_SUBMODULE_ALIASES = {
    "training.build_dataset": "grammetarl.training.build_dataset",
    "training.run_eval": "grammetarl.training.run_eval",
    "training.run_grpo": "grammetarl.training.run_grpo",
    "training.run_sft": "grammetarl.training.run_sft",
}

for legacy_name, canonical_name in _SUBMODULE_ALIASES.items():
    sys.modules.setdefault(legacy_name, import_module(canonical_name))

__all__ = ["build_min_dataset"]
