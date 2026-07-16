from __future__ import annotations

import argparse
import json
from pathlib import Path

from grammetarl.workspace import get_workspace_paths


def build_min_dataset(out_path: str | None = None, namespace: str | None = None) -> None:
    p = Path(out_path) if out_path else get_workspace_paths(namespace=namespace).train_dataset_file
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "id": "train_0001",
            "src": "I did not see him.",
            "tgt_ref": "Ech hunn hien net gesinn.",
            "lang_pair": "en->lb",
            "difficulty": "easy",
            "phenomena_tags": ["negation", "word_order"],
            "gold_rules": ["LB_NEG_0001"],
            "gold_lexicon": ["see", "him"],
        }
    ]
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build minimal training dataset")
    parser.add_argument("--output", default=None, help="Optional output JSONL path")
    parser.add_argument("--namespace", default=None, help="Namespace for default output path")
    args = parser.parse_args()
    build_min_dataset(out_path=args.output, namespace=args.namespace)


if __name__ == "__main__":
    main()
