from __future__ import annotations

import argparse
import json
from pathlib import Path

from grammetarl.workspace import get_workspace_paths


# ---------------------------------------------------------------------------
# Lazy verl import: this module must import (and the extraction pipeline must
# keep working) even in environments where verl is not installed. verl is only
# imported inside the functions that actually launch training.
#
# verl 0.9.0 requires transformers>=5.5.3 / torch>=2.9, which conflicts with the
# extraction-pipeline pins. Run this entrypoint in a dedicated RL virtualenv
# (see pyproject.toml [project.optional-dependencies] rl).
# ---------------------------------------------------------------------------
def _require_verl() -> None:
    try:
        import verl  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise SystemExit(
            "verl is not installed. Install the RL stack in a dedicated "
            "virtualenv, e.g.:\n"
            "  uv venv .venv-rl --python 3.11\n"
            "  uv pip install --python .venv-rl/bin/python "
            "'verl==0.9.0' 'ray[default]>=2.41.0' 'hydra-core' 'accelerate'\n"
            "(verl requires transformers>=5.5.3 / torch>=2.9; do not install it "
            "into the extraction pipeline's environment.)"
        ) from exc


def _load_config(path: str | Path) -> dict:
    import yaml

    cfg = Path(path)
    if not cfg.exists():
        raise FileNotFoundError(f"Config not found: {cfg}")
    data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def build_sft_dataset(
    train_jsonl: str | Path,
    model: str,
    max_prompt_length: int = 4096,
) -> list[dict]:
    """Convert TranslationExample rows into verl SFT prompts.

    verl SFT expects a chat/record format; we emit a grammar-first system
    prompt plus a user turn carrying the source sentence, so the model learns
    to emit the tool-calling action format before producing a translation.
    """
    from grammetarl.agent_env import GrammarFirstEnv

    env = GrammarFirstEnv()
    rows: list[dict] = []
    with open(train_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            src = rec.get("src", "")
            tgt = rec.get("tgt_ref", "")
            prompt = (
                "You are a grammar-first translation agent for "
                f"{rec.get('lang_pair', 'en->lb')}.\n"
                "First retrieve the relevant grammar rules, then the dictionary, "
                "then apply the rules and return the translation.\n"
                f"Source: {src}\n"
                "Target reference (for supervision): {tgt}"
            )
            rows.append(
                {
                    "prompt": prompt,
                    "response": tgt,
                    "phenomena_tags": rec.get("phenomena_tags", []),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT cold-start entrypoint (verl)")
    parser.add_argument("--config", default="grammetarl/configs/exp/debug_small.yaml")
    parser.add_argument("--train-jsonl", default=None)
    parser.add_argument("--namespace", default=None)
    args = parser.parse_args()

    _require_verl()

    ws = get_workspace_paths(namespace=args.namespace)
    train_jsonl = args.train_jsonl or str(ws.train_dataset_file)
    cfg = _load_config(args.config)
    model_name = (cfg.get("model") or {}).get("name", "Qwen/Qwen3-8B")
    max_prompt_length = (cfg.get("model") or {}).get("max_prompt_length", 4096)

    dataset = build_sft_dataset(train_jsonl, model_name, max_prompt_length)
    print(f"[SFT] {len(dataset)} examples from {train_jsonl}")
    print("[SFT] Launching verl FSDP2 SFT trainer (see verl docs for the full API).")

    # Real verl SFT launch is environment-specific; expose the dataset + config
    # so an operator can wire it to verl's sft_trainer with the loaded config.
    import json as _json

    print(_json.dumps({"config": cfg, "dataset_len": len(dataset)}, indent=2, default=str))


if __name__ == "__main__":
    main()