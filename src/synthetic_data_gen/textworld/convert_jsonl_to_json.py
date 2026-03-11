#!/usr/bin/env python3
"""
Convert JSONL file to individual JSON files with added source field.

This script processes TextWorld trajectory JSONL files and converts them to
individual JSON files suitable for the AMA-Bench dataset format.

Operations:
1. Reads trajectory.jsonl file (one episode per line)
2. Adds "source": "textworld" field to each episode
3. Removes deprecated "action_space" field if present
4. Saves each episode as a separate JSON file

Usage:
    python convert_jsonl_to_json.py
    python convert_jsonl_to_json.py input_file=path/to/trajectory.jsonl output_dir=output/
"""

import json
from pathlib import Path
import sys


def parse_kv_args(argv):
    """Parse key=value command line arguments."""
    args = {}
    for token in argv[1:]:
        if "=" not in token:
            continue
        k, v = token.split("=", 1)
        k = k.strip().lower()
        v = v.strip()
        if k:
            args[k] = v
    return args


def convert_jsonl_to_json(input_file_path: str = None, output_dir_path: str = "dataset/textworld"):
    """
    Convert JSONL to individual JSON files in AMA-Bench dataset format.

    Args:
        input_file_path: Path to input JSONL file. If None, uses tw_out/trajectory.jsonl
        output_dir_path: Output directory for JSON files (default: dataset/textworld)

    Returns:
        Number of successfully converted entries
    """
    if input_file_path is None:
        input_file = Path("tw_out/trajectory.jsonl")
    else:
        input_file = Path(input_file_path)

    output_dir = Path(output_dir_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_file.exists():
        print(f"Error: Input file {input_file} not found")
        return 0

    count = 0
    errors = []

    with input_file.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)

                # Add source field for dataset compatibility
                data["source"] = "textworld"

                # Remove deprecated fields
                data.pop("action_space", None)

                # Generate output filename: task_textworld_1.json, task_textworld_2.json, ...
                output_file = output_dir / f"task_textworld_{line_num}.json"

                with output_file.open("w", encoding="utf-8") as out_f:
                    json.dump(data, out_f, ensure_ascii=False, indent=2)

                count += 1
                print(f"Converted [{count}]: {output_file.name}")

            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: JSON parse error - {e}")
            except Exception as e:
                errors.append(f"Line {line_num}: {e}")

    print(f"\nSuccessfully converted {count} entries")
    print(f"Output directory: {output_dir.absolute()}")

    if errors:
        print(f"\nWarning: {len(errors)} errors occurred:")
        for error in errors[:5]:  # Show first 5 errors
            print(f"  - {error}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")

    return count


if __name__ == "__main__":
    kv = parse_kv_args(sys.argv)
    input_file = kv.get("input_file", None)
    output_dir = kv.get("output_dir", "dataset/textworld")

    num_converted = convert_jsonl_to_json(input_file_path=input_file, output_dir_path=output_dir)
    sys.exit(0 if num_converted > 0 else 1)
