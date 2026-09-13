from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from app.core.constants import ErrorCode
from app.core.errors import AppError


@dataclass
class CasePaths:
    root: Path
    input_dir: Path
    output_dir: Path
    metadata_dir: Path
    metadata_file: Path


class CaseStorage:
    """Isolated per-case filesystem storage. Original filenames are never used as paths."""

    def __init__(self, upload_root: Path) -> None:
        self.upload_root = upload_root.resolve()
        self.cases_root = self.upload_root / "cases"
        self.cases_root.mkdir(parents=True, exist_ok=True)

    def paths_for(self, case_id: str) -> CasePaths:
        root = self._case_root(case_id)
        input_dir = root / "input"
        output_dir = root / "output"
        metadata_dir = root / "metadata"
        return CasePaths(
            root=root,
            input_dir=input_dir,
            output_dir=output_dir,
            metadata_dir=metadata_dir,
            metadata_file=metadata_dir / "case.json",
        )

    def prepare_case(self, case_id: str) -> CasePaths:
        paths = self.paths_for(case_id)
        paths.input_dir.mkdir(parents=True, exist_ok=True)
        paths.output_dir.mkdir(parents=True, exist_ok=True)
        paths.metadata_dir.mkdir(parents=True, exist_ok=True)
        self._assert_inside(paths.root)
        return paths

    def destination_for(self, case_id: str, stored_name: str) -> Path:
        if stored_name != Path(stored_name).name or ".." in stored_name:
            raise AppError(
                ErrorCode.STORAGE_ERROR,
                "Refusing to write an unsafe storage filename.",
                status_code=500,
                case_id=case_id,
            )
        dest = self.paths_for(case_id).input_dir / stored_name
        self._assert_inside(dest)
        return dest

    def write_metadata(self, case_id: str, payload: dict[str, object]) -> None:
        paths = self.prepare_case(case_id)
        serialized = json.dumps(payload, indent=2, default=str)
        paths.metadata_file.write_text(serialized, encoding="utf-8")

    def relative_to_root(self, path: Path) -> str:
        resolved = path.resolve()
        self._assert_inside(resolved)
        return str(resolved.relative_to(self.upload_root))

    def _case_root(self, case_id: str) -> Path:
        if not case_id or any(part in case_id for part in ("/", "\\", "..")):
            raise AppError(
                ErrorCode.STORAGE_ERROR,
                "Invalid case identifier.",
                status_code=500,
            )
        root = (self.cases_root / case_id).resolve()
        self._assert_inside(root)
        return root

    def _assert_inside(self, path: Path) -> None:
        resolved = path.resolve()
        if resolved != self.upload_root and self.upload_root not in resolved.parents:
            raise AppError(
                ErrorCode.STORAGE_ERROR,
                "Storage path escaped the upload root.",
                status_code=500,
            )


def metadata_dict(payload: object) -> dict[str, object]:
    if hasattr(payload, "__dataclass_fields__"):
        return asdict(payload)  # type: ignore[arg-type]
    if isinstance(payload, dict):
        return payload
    raise TypeError("Unsupported metadata payload")
