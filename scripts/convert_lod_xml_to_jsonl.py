#!/usr/bin/env python3
"""Convert a large LOD XML dictionary into JSONL records without losing structure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def element_to_lossless_dict(elem: ET.Element) -> dict:
    """Convert an XML element into a recursive dict that preserves structure."""
    return {
        "tag": elem.tag,
        "attrs": dict(elem.attrib),
        "text": elem.text,
        "children": [element_to_lossless_dict(child) for child in list(elem)],
        "tail": elem.tail,
    }


def convert_xml_to_jsonl(input_path: Path, output_path: Path, record_tag: str = "entry") -> int:
    """Stream XML and write one JSON object per matching record tag."""
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    context = ET.iterparse(str(input_path), events=("start", "end"))
    first_event, root = next(context)
    if first_event != "start":
        raise ValueError("Invalid XML parse state: expected a start event first")

    with output_path.open("w", encoding="utf-8") as out:
        for event, elem in context:
            if event != "end" or elem.tag != record_tag:
                continue

            record = {
                "record_type": record_tag,
                "record_id": elem.attrib.get("id"),
                "record": element_to_lossless_dict(elem),
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

            # Release processed elements to keep memory stable on huge files.
            root.clear()

    return count


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert LOD XML into JSONL with one record per <entry> and full structure retained."
    )
    parser.add_argument("input_xml", type=Path, help="Path to input XML file")
    parser.add_argument("output_jsonl", type=Path, help="Path to output JSONL file")
    parser.add_argument(
        "--record-tag",
        default="entry",
        help="XML tag to emit as one JSONL record (default: entry)",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    total = convert_xml_to_jsonl(
        input_path=args.input_xml,
        output_path=args.output_jsonl,
        record_tag=args.record_tag,
    )
    print(f"Wrote {total} records to {args.output_jsonl}")


if __name__ == "__main__":
    main()
