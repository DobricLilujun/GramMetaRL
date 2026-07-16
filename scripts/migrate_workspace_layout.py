from __future__ import annotations

import argparse
import json

from grammetarl.migration import migrate_legacy_layout


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Migrate legacy GramMetaRL paths into standardized workspace layout.")
    parser.add_argument("--workspace-root", default=None, help="Project root path")
    parser.add_argument("--namespace", default=None, help="Workspace namespace (default from env or 'default')")
    parser.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--dry-run", action="store_true", help="Show migration plan without writing files")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = migrate_legacy_layout(
        workspace_root=args.workspace_root,
        namespace=args.namespace,
        mode=args.mode,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
