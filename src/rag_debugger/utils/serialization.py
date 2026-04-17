from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(path: str | Path, payload: Any, *, indent: int = 2) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=indent), encoding="utf-8")
    return target


def load_json(path: str | Path) -> Any:
    source = Path(path)
    return json.loads(source.read_text(encoding="utf-8"))
