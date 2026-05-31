from __future__ import annotations

from pathlib import Path

from scripts.check_hidden_unicode import find_hidden_unicode


def test_hidden_unicode_detection(tmp_path: Path) -> None:
    path = tmp_path / "bad.py"
    path.write_text("print('safe')\u202e\n", encoding="utf-8")
    findings = find_hidden_unicode(path)
    assert findings[0]["codepoint"] == "U+202E"
