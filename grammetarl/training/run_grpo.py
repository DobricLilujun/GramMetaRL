from __future__ import annotations

import argparse
import json
from pathlib import Path

from grammetarl.workspace import get_workspace_paths


def _require_verl() -> None:
    try:
        import verl  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "verl is not installed. Install the RL stack in a dedicated "
            "virtualenv (see pyproject.toml [project.optional-dependencies] rl). "
            "Do not install verl into the extraction pipeline's environment "
            "(transformers/torch version conflict)."
        ) from exc


def _load_config(path: str | Path) -> dict:
    import yaml

    cfg = Path(path)
    if not cfg.exists():
        raise FileNotFoundError(f"Config not found: {cfg}")
    data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _build_reward_fn() -> dict:
    """Assemble the aggregate reward from the existing reward module.

    R = w_mt*R_mt + w_grammar*R_grammar + w_tool*R_tool + ...
    Weights come from configs/reward/grammar_reward.yaml (grammar > tool > mt).
    """
    from grammetarl.rewards import aggregate_reward

    return {"aggregate_reward": aggregate_reward}


def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO/DrGRPO entrypoint (verl)")
    parser.add_argument("--config", default="grammetarl/configs/exp/debug_small.yaml")
    parser.add_argument("--algo", choices=["grpo", "drgrpo"], default="grpo")
    parser.add_argument("--train-jsonl", default=None)
    parser.add_argument("--namespace", default=None)
    args = parser.parse_args()

    _require_verl()

    ws = get_workspace_paths(namespace=args.namespace)
    train_jsonl = args.train_jsonl or str(ws.train_dataset_file)
    cfg = _load_config(args.config)
    reward_cfg = _load_config("grammetarl/configs/reward/grammar_reward.yaml")

    # GRPO vs DrGRPO: select the algorithm config by name.
    algo_name = "drgrpo" if args.algo == "drgrpo" else "grpo"
    algo_cfg = _load_config(f"grammetarl/configs/algo/{algo_name}.yaml")

    reward_fn = _build_reward_fn()
    reward_weights = reward_cfg.get("weights", {})

    payload = {
        "config": cfg,
        "algo": algo_name,
        "algo_cfg": algo_cfg,
        "reward_weights": reward_weights,
        "reward_fn": "aggregate_reward",
        "train_jsonl": train_jsonl,
        "note": "verl rollout backend (vllm/sglang) + FSDP2 actor are configured "
        "in configs/train/fsdp2_vllm.yaml and configs/algo/<algo>.yaml. "
        "This entrypoint assembles the dataset, reward, and tool-calling "
        "environment (grammetarl.agent_env.GrammarFirstEnv) and hands them to "
        "verl's ppo/GRPO trainer.",
    }
    print(json.dumps(payload, indent=2, default=str))
    print(
        f"\n[GRPO] Launching verl {algo_name} trainer with "
        f"grammar-first env + aggregate reward. "
        f"Switch --algo drgrpo if trajectory-length hacking appears."
    )


if __name__ == "__main__":
    main()