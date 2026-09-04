"""Tuning record store: one JSON file per (hardware, model, workload) key, written atomically."""

from __future__ import annotations

import builtins
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from servepilot.constants import TUNING_CACHE_SUBDIR, TUNING_RECORD_SCHEMA_VERSION
from servepilot.exceptions import CacheError
from servepilot.fsutil import atomic_write_json, read_json
from servepilot.logging import get_logger
from servepilot.schemas.runtime import TuningRecord, make_record_key

log = get_logger(__name__)


class CacheStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.dir = root / TUNING_CACHE_SUBDIR

    def path_for(self, key: str) -> Path:
        safe = "".join(ch for ch in key if ch.isalnum() or ch in "-_")
        return self.dir / f"{safe}.json"

    def save(self, record: TuningRecord) -> Path:
        record.updated_at = datetime.now(tz=UTC)
        path = self.path_for(record.key)
        atomic_write_json(path, record.model_dump(mode="json"))
        return path

    def load(self, key: str) -> TuningRecord:
        path = self.path_for(key)
        if not path.exists():
            raise CacheError(
                f"no tuning record with key {key!r}", hints=["Run `servepilot cache list`."]
            )
        return self._load_path(path)

    def _load_path(self, path: Path) -> TuningRecord:
        try:
            raw = read_json(path)
        except (OSError, ValueError) as exc:
            raise CacheError(f"tuning record {path} is unreadable: {exc}") from exc
        if not isinstance(raw, dict):
            raise CacheError(f"tuning record {path} is not a JSON object")
        version = raw.get("schema_version")
        if version != TUNING_RECORD_SCHEMA_VERSION:
            raise CacheError(
                f"tuning record {path.name} uses schema version {version}; this ServePilot understands "
                f"version {TUNING_RECORD_SCHEMA_VERSION}",
                hints=[
                    "Run `servepilot cache clear` to discard records written by another version."
                ],
            )
        try:
            return TuningRecord.model_validate(raw)
        except ValidationError as exc:
            raise CacheError(f"tuning record {path.name} is invalid: {exc}") from exc

    def find(self, hardware_fp: str, model_fp: str, workload_fp: str) -> TuningRecord | None:
        path = self.path_for(make_record_key(hardware_fp, model_fp, workload_fp))
        if not path.exists():
            return None
        return self._load_path(path)

    def list(self) -> builtins.list[TuningRecord]:
        if not self.dir.exists():
            return []
        records: builtins.list[TuningRecord] = []
        for path in sorted(self.dir.glob("*.json")):
            try:
                records.append(self._load_path(path))
            except CacheError as exc:
                log.warning("skipping %s: %s", path.name, exc)
        return records

    def list_paths(self) -> builtins.list[Path]:
        return sorted(self.dir.glob("*.json")) if self.dir.exists() else []

    def delete(self, key: str) -> bool:
        path = self.path_for(key)
        if path.exists():
            path.unlink()
            return True
        return False

    def clear(self) -> int:
        count = 0
        for path in self.list_paths():
            path.unlink()
            count += 1
        return count
