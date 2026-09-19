"""
Validate a Q&A fine-tuning dataset in JSONL format.

Checks performed:
1. Every line is valid JSON.
2. Every record has exactly the expected schema: "instruction" and "output" keys,
   both non-empty strings.
3. No duplicate instructions (exact match, case-insensitive, whitespace-normalized).
4. No near-duplicate instructions (high text overlap) — flagged as a warning, not an error.
5. No suspiciously short answers (likely truncated or low-effort).
6. No suspiciously long answers (likely rambling / off-format).
7. Basic dataset statistics: count, avg/min/max lengths, word counts.

Usage:
    python src/validate_dataset.py data/qa_dataset.jsonl
"""

import json
import sys
import re
from collections import Counter
from difflib import SequenceMatcher

REQUIRED_KEYS = {"instruction", "output"}
MIN_OUTPUT_WORDS = 8          # flag answers shorter than this as too short
MAX_OUTPUT_WORDS = 200        # flag answers longer than this as too long
NEAR_DUP_THRESHOLD = 0.85     # similarity ratio above which two instructions are "near-duplicate"


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def validate(path: str) -> int:
    errors = []
    warnings = []
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for lineno, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue  # skip blank lines silently

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"Line {lineno}: invalid JSON — {e}")
                continue

            if not isinstance(obj, dict):
                errors.append(f"Line {lineno}: expected a JSON object, got {type(obj).__name__}")
                continue

            missing = REQUIRED_KEYS - obj.keys()
            extra = obj.keys() - REQUIRED_KEYS
            if missing:
                errors.append(f"Line {lineno}: missing required key(s) {sorted(missing)}")
                continue
            if extra:
                warnings.append(f"Line {lineno}: unexpected extra key(s) {sorted(extra)} (not an error, just noting)")

            instruction = obj["instruction"]
            output = obj["output"]

            if not isinstance(instruction, str) or not instruction.strip():
                errors.append(f"Line {lineno}: 'instruction' is empty or not a string")
                continue
            if not isinstance(output, str) or not output.strip():
                errors.append(f"Line {lineno}: 'output' is empty or not a string")
                continue

            records.append({"lineno": lineno, "instruction": instruction, "output": output})

    # Duplicate check (exact, normalized)
    seen = {}
    for r in records:
        key = normalize(r["instruction"])
        if key in seen:
            errors.append(
                f"Line {r['lineno']}: duplicate instruction of line {seen[key]} "
                f"— \"{r['instruction'][:70]}...\""
            )
        else:
            seen[key] = r["lineno"]

    # Near-duplicate check (only compare within a reasonable window to keep it fast)
    instructions_norm = [(r["lineno"], normalize(r["instruction"])) for r in records]
    for i in range(len(instructions_norm)):
        for j in range(i + 1, len(instructions_norm)):
            ln_i, a = instructions_norm[i]
            ln_j, b = instructions_norm[j]
            if a == b:
                continue  # already caught as exact duplicate
            ratio = SequenceMatcher(None, a, b).ratio()
            if ratio >= NEAR_DUP_THRESHOLD:
                warnings.append(
                    f"Lines {ln_i} & {ln_j}: near-duplicate instructions (similarity {ratio:.2f})"
                )

    # Length checks
    word_counts = []
    for r in records:
        n_words = len(r["output"].split())
        word_counts.append(n_words)
        if n_words < MIN_OUTPUT_WORDS:
            warnings.append(f"Line {r['lineno']}: output is very short ({n_words} words) — check it's a full answer")
        if n_words > MAX_OUTPUT_WORDS:
            warnings.append(f"Line {r['lineno']}: output is very long ({n_words} words) — check it's not rambling")

    # ---- Report ----
    print(f"Validated: {path}")
    print(f"Total records parsed: {len(records)}")
    if word_counts:
        print(
            f"Output length (words): avg={sum(word_counts)/len(word_counts):.1f}, "
            f"min={min(word_counts)}, max={max(word_counts)}"
        )

    print(f"\nErrors: {len(errors)}")
    for e in errors:
        print(f"  ERROR: {e}")

    print(f"\nWarnings: {len(warnings)}")
    for w in warnings:
        print(f"  WARNING: {w}")

    print("\n" + ("PASS — dataset is clean, ready for training." if not errors else "FAIL — fix errors before training."))
    return 1 if errors else 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python validate_dataset.py <path_to_jsonl>")
        sys.exit(2)
    sys.exit(validate(sys.argv[1]))