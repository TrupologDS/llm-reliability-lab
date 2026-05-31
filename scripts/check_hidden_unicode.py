"""Fail if source/config/docs contain hidden Unicode control characters."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

TEXT_EXTENSIONS = {".py", ".yml", ".yaml", ".md", ".toml", ".jsonl"}
SKIP_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".venv-gpu",
    "__pycache__",
    "checkpoints",
    "mlruns",
    "models",
    "outputs",
    "wandb",
}
HIDDEN_UNICODE = {
    "\u061c": "ARABIC LETTER MARK",
    "\u200b": "ZERO WIDTH SPACE",
    "\u200c": "ZERO WIDTH NON-JOINER",
    "\u200d": "ZERO WIDTH JOINER",
    "\u200e": "LEFT-TO-RIGHT MARK",
    "\u200f": "RIGHT-TO-LEFT MARK",
    "\u202a": "LEFT-TO-RIGHT EMBEDDING",
    "\u202b": "RIGHT-TO-LEFT EMBEDDING",
    "\u202c": "POP DIRECTIONAL FORMATTING",
    "\u202d": "LEFT-TO-RIGHT OVERRIDE",
    "\u202e": "RIGHT-TO-LEFT OVERRIDE",
    "\u2066": "LEFT-TO-RIGHT ISOLATE",
    "\u2067": "RIGHT-TO-LEFT ISOLATE",
    "\u2068": "FIRST STRONG ISOLATE",
    "\u2069": "POP DIRECTIONAL ISOLATE",
    "\ufeff": "BYTE ORDER MARK",
}


def iter_text_files(root: Path) -> Iterable[Path]:
    """Yield files with extensions covered by this guard."""

    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        if any(part in SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        yield path


def find_hidden_unicode(path: Path) -> list[dict[str, object]]:
    """Return hidden Unicode findings for a file."""

    text = path.read_text(encoding="utf-8")
    findings: list[dict[str, object]] = []
    line = 1
    column = 0
    for character in text:
        if character == "\n":
            line += 1
            column = 0
            continue
        column += 1
        if character in HIDDEN_UNICODE:
            findings.append(
                {
                    "path": str(path),
                    "line": line,
                    "column": column,
                    "codepoint": f"U+{ord(character):04X}",
                    "name": HIDDEN_UNICODE[character],
                }
            )
    return findings


def scan(root: Path) -> list[dict[str, object]]:
    """Scan all relevant files under root."""

    findings: list[dict[str, object]] = []
    for path in iter_text_files(root):
        findings.extend(find_hidden_unicode(path))
    return findings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    findings = scan(Path(args.root).resolve())
    if findings:
        for finding in findings:
            print(f"{finding['path']}:{finding['line']}:{finding['column']} {finding['codepoint']} {finding['name']}")
        raise SystemExit(1)
    print("No hidden Unicode control characters found.")


if __name__ == "__main__":
    main()
