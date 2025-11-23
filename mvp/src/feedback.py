from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from .config_loader import project_paths


@dataclass
class CorrectionRecord:
    timestamp: str
    description: str
    predicted_category_id: str
    corrected_category_id: str
    metadata: Dict[str, Any]


def _corrections_path() -> Path:
    paths = project_paths()
    return paths["data"] / "corrections.json"


def _load_corrections() -> List[Dict[str, Any]]:
    path = _corrections_path()
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_corrections(records: List[Dict[str, Any]]) -> None:
    path = _corrections_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


def record_correction(
    description: str,
    predicted_category_id: str,
    corrected_category_id: str,
    metadata: Dict[str, Any] | None = None,
) -> None:
    if metadata is None:
        metadata = {}
    now = datetime.now(timezone.utc).isoformat()
    record = CorrectionRecord(
        timestamp=now,
        description=description,
        predicted_category_id=predicted_category_id,
        corrected_category_id=corrected_category_id,
        metadata=metadata,
    )
    records = _load_corrections()
    records.append(asdict(record))
    _save_corrections(records)
