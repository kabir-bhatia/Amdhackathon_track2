#!/usr/bin/env python3
"""Validate an output results.json against the Track 2 schema.

Usage: python validate_results.py test/output/results.json test/input/tasks.json
Exits non-zero on any schema violation.
"""
import json
import sys

STYLES_REQUIRED_FROM_TASKS = True


def main() -> int:
    results_path = sys.argv[1] if len(sys.argv) > 1 else "test/output/results.json"
    tasks_path = sys.argv[2] if len(sys.argv) > 2 else "test/input/tasks.json"

    with open(results_path, encoding="utf-8") as f:
        results = json.load(f)
    with open(tasks_path, encoding="utf-8") as f:
        tasks = json.load(f)

    want = {t["task_id"]: t.get("styles", []) for t in tasks}
    errors = []

    if not isinstance(results, list):
        print("FAIL: results.json is not a JSON array")
        return 1

    seen = set()
    for i, r in enumerate(results):
        if not isinstance(r, dict):
            errors.append(f"entry {i}: not an object")
            continue
        tid = r.get("task_id")
        seen.add(tid)
        caps = r.get("captions")
        if tid is None:
            errors.append(f"entry {i}: missing task_id")
        if not isinstance(caps, dict):
            errors.append(f"{tid}: 'captions' missing or not an object")
            continue
        for style in want.get(tid, []):
            val = caps.get(style)
            if not isinstance(val, str) or not val.strip():
                errors.append(f"{tid}: style '{style}' missing or empty")

    for tid in want:
        if tid not in seen:
            errors.append(f"task '{tid}' has no result entry")

    if errors:
        print("SCHEMA INVALID:")
        for e in errors:
            print("  -", e)
        return 1

    print(f"OK: {len(results)} tasks, all requested styles present and non-empty.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
