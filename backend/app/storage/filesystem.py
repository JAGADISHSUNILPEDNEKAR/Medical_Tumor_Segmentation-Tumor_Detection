from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from app.core.constants import ALLOWED_NIFTI_EXTENSIONS, ErrorCode, VIEWER_ARTIFACTS
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

    def resolve_artifact(self, case_id: str, artifact: str) -> Path:
        """Resolve an allowlisted viewer artifact inside the case directory.

        The artifact name is never joined into a path until it matches
        VIEWER_ARTIFACTS exactly. Path traversal, absolute paths, and
        unknown names are rejected.
        """
        if not _is_safe_artifact_name(artifact) or artifact not in VIEWER_ARTIFACTS:
            raise AppError(
                ErrorCode.INVALID_ARTIFACT,
                "Unknown artifact. Allowed values: t1, t1ce, t2, flair, segmentation.",
                status_code=400,
                case_id=case_id,
            )
        folder, stem = VIEWER_ARTIFACTS[artifact]
        paths = self.paths_for(case_id)
        base = paths.input_dir if folder == "input" else paths.output_dir
        self._assert_inside(base)
        for extension in ALLOWED_NIFTI_EXTENSIONS:
            candidate = (base / f"{stem}{extension}").resolve()
            self._assert_inside(candidate)
            if candidate.is_file():
                return candidate
        raise AppError(
            ErrorCode.ARTIFACT_NOT_FOUND,
            f"Artifact '{artifact}' is not available for this case.",
            status_code=404,
            case_id=case_id,
        )

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


def _is_safe_artifact_name(artifact: str) -> bool:
    if not artifact or not artifact.isascii():
        return False
    if artifact != artifact.strip():
        return False
    if "/" in artifact or "\\" in artifact or ".." in artifact:
        return False
    if Path(artifact).is_absolute() or Path(artifact).name != artifact:
        return False
    return artifact.isalnum() or artifact.replace("_", "").isalnum()


def metadata_dict(payload: object) -> dict[str, object]:
    if hasattr(payload, "__dataclass_fields__"):
        return asdict(payload)  # type: ignore[arg-type]
    if isinstance(payload, dict):
        return payload
    raise TypeError("Unsupported metadata payload")
