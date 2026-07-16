from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_NAMESPACE = "default"


def resolve_namespace(explicit_namespace: str | None = None) -> str:
    if explicit_namespace:
        return explicit_namespace.strip() or DEFAULT_NAMESPACE
    env_ns = os.getenv("GRAMMETARL_NAMESPACE")
    if env_ns:
        return env_ns.strip() or DEFAULT_NAMESPACE
    return DEFAULT_NAMESPACE


def _read_path_templates(project_root: Path) -> dict[str, str]:
    cfg = project_root / "configs" / "project" / "paths.yaml"
    if not cfg.exists():
        return {}

    try:
        import yaml  # type: ignore
    except Exception:
        return {}

    try:
        data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}

    if not isinstance(data, dict):
        return {}
    paths = data.get("paths", {})
    if not isinstance(paths, dict):
        return {}
    out: dict[str, str] = {}
    for k, v in paths.items():
        if isinstance(k, str) and isinstance(v, str):
            out[k] = v
    return out


def _resolve_relpath(template: str, root: Path, namespace: str) -> Path:
    return root / template.format(namespace=namespace)


def _pick_template(templates: dict[str, str], namespace: str, key: str) -> str:
    default_map: dict[str, str] = {
        "artifacts_root_default": "artifacts",
        "artifacts_root_namespaced": "artifacts/experiments/{namespace}",
        "input_pdf_dir": "data/raw/pdf",
        "input_ocr_dir": "data/raw/ocr",
        "rules_dir": "data/rules",
        "lexicon_dir": "data/lexicon",
        "processed_dir_default": "data/processed",
        "processed_dir_namespaced": "data/processed/{namespace}",
        "rules_cards_file_default": "data/rules/mbg_cards.jsonl",
        "rules_cards_file_namespaced": "data/rules/mbg_cards_{namespace}.jsonl",
        "lexicon_file": "data/lexicon/dictionary.jsonl",
        "train_dataset_file_default": "data/processed/train.jsonl",
        "train_dataset_file_namespaced": "data/processed/{namespace}/train.jsonl",
        "ocr_run_dir_default": "artifacts/ocr/deepseek_fullbook",
        "ocr_run_dir_namespaced": "artifacts/experiments/{namespace}/ocr/deepseek_fullbook",
        "mbg_work_dir_default": "artifacts/extraction/_mbg_work",
        "mbg_work_dir_namespaced": "artifacts/experiments/{namespace}/extraction/_mbg_work",
    }
    selected = templates.get(key)
    if selected:
        return selected
    fallback = default_map.get(key)
    if fallback:
        return fallback
    raise KeyError(f"Unknown path template key: {key}")


def resolve_project_root(explicit_root: str | Path | None = None) -> Path:
    if explicit_root:
        return Path(explicit_root).expanduser().resolve()
    env_root = os.getenv("GRAMMETARL_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class WorkspacePaths:
    namespace: str
    root: Path
    data: Path
    artifacts: Path
    configs: Path
    prompts: Path
    notebooks: Path

    input_pdf_dir: Path
    input_ocr_dir: Path

    rules_dir: Path
    lexicon_dir: Path
    processed_dir: Path

    extraction_artifacts_dir: Path
    ocr_artifacts_dir: Path
    rl_artifacts_dir: Path
    eval_artifacts_dir: Path
    logs_dir: Path

    rules_cards_file: Path
    lexicon_file: Path
    train_dataset_file: Path

    default_ocr_run_dir: Path
    default_ocr_pages_jsonl: Path
    default_ocr_blocks_jsonl: Path

    default_mbg_output_file: Path
    default_mbg_work_dir: Path

    @staticmethod
    def from_root(root: str | Path | None = None, namespace: str | None = None) -> "WorkspacePaths":
        project_root = resolve_project_root(root)
        ns = resolve_namespace(namespace)
        templates = _read_path_templates(project_root)

        data = project_root / "data"
        configs = project_root / "configs"
        prompts = project_root / "prompts"
        notebooks = project_root / "notebook"

        if ns == DEFAULT_NAMESPACE:
            artifacts = _resolve_relpath(_pick_template(templates, ns, "artifacts_root_default"), project_root, ns)
            processed_dir = _resolve_relpath(_pick_template(templates, ns, "processed_dir_default"), project_root, ns)
            rules_cards_file = _resolve_relpath(
                _pick_template(templates, ns, "rules_cards_file_default"), project_root, ns
            )
            train_dataset_file = _resolve_relpath(
                _pick_template(templates, ns, "train_dataset_file_default"), project_root, ns
            )
            default_ocr_run_dir = _resolve_relpath(_pick_template(templates, ns, "ocr_run_dir_default"), project_root, ns)
            default_mbg_work_dir = _resolve_relpath(_pick_template(templates, ns, "mbg_work_dir_default"), project_root, ns)
        else:
            artifacts = _resolve_relpath(
                _pick_template(templates, ns, "artifacts_root_namespaced"), project_root, ns
            )
            processed_dir = _resolve_relpath(
                _pick_template(templates, ns, "processed_dir_namespaced"), project_root, ns
            )
            rules_cards_file = _resolve_relpath(
                _pick_template(templates, ns, "rules_cards_file_namespaced"), project_root, ns
            )
            train_dataset_file = _resolve_relpath(
                _pick_template(templates, ns, "train_dataset_file_namespaced"), project_root, ns
            )
            default_ocr_run_dir = _resolve_relpath(
                _pick_template(templates, ns, "ocr_run_dir_namespaced"), project_root, ns
            )
            default_mbg_work_dir = _resolve_relpath(
                _pick_template(templates, ns, "mbg_work_dir_namespaced"), project_root, ns
            )

        input_pdf_dir = _resolve_relpath(_pick_template(templates, ns, "input_pdf_dir"), project_root, ns)
        input_ocr_dir = _resolve_relpath(_pick_template(templates, ns, "input_ocr_dir"), project_root, ns)
        rules_dir = _resolve_relpath(_pick_template(templates, ns, "rules_dir"), project_root, ns)
        lexicon_dir = _resolve_relpath(_pick_template(templates, ns, "lexicon_dir"), project_root, ns)

        extraction_artifacts_dir = artifacts / "extraction"
        ocr_artifacts_dir = artifacts / "ocr"
        rl_artifacts_dir = artifacts / "rl"
        eval_artifacts_dir = artifacts / "eval"
        logs_dir = artifacts / "logs"

        lexicon_file = _resolve_relpath(_pick_template(templates, ns, "lexicon_file"), project_root, ns)

        default_ocr_pages_jsonl = default_ocr_run_dir / "pages_ocr.jsonl"
        default_ocr_blocks_jsonl = default_ocr_run_dir / "blocks_ocr.jsonl"

        default_mbg_output_file = rules_cards_file

        return WorkspacePaths(
            namespace=ns,
            root=project_root,
            data=data,
            artifacts=artifacts,
            configs=configs,
            prompts=prompts,
            notebooks=notebooks,
            input_pdf_dir=input_pdf_dir,
            input_ocr_dir=input_ocr_dir,
            rules_dir=rules_dir,
            lexicon_dir=lexicon_dir,
            processed_dir=processed_dir,
            extraction_artifacts_dir=extraction_artifacts_dir,
            ocr_artifacts_dir=ocr_artifacts_dir,
            rl_artifacts_dir=rl_artifacts_dir,
            eval_artifacts_dir=eval_artifacts_dir,
            logs_dir=logs_dir,
            rules_cards_file=rules_cards_file,
            lexicon_file=lexicon_file,
            train_dataset_file=train_dataset_file,
            default_ocr_run_dir=default_ocr_run_dir,
            default_ocr_pages_jsonl=default_ocr_pages_jsonl,
            default_ocr_blocks_jsonl=default_ocr_blocks_jsonl,
            default_mbg_output_file=default_mbg_output_file,
            default_mbg_work_dir=default_mbg_work_dir,
        )

    def ensure_layout(self) -> None:
        required_dirs = [
            self.input_pdf_dir,
            self.input_ocr_dir,
            self.rules_dir,
            self.lexicon_dir,
            self.processed_dir,
            self.extraction_artifacts_dir,
            self.ocr_artifacts_dir,
            self.rl_artifacts_dir,
            self.eval_artifacts_dir,
            self.logs_dir,
        ]
        for path in required_dirs:
            path.mkdir(parents=True, exist_ok=True)


def get_workspace_paths(root: str | Path | None = None, namespace: str | None = None) -> WorkspacePaths:
    return WorkspacePaths.from_root(root, namespace=namespace)
