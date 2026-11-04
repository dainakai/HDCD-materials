from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path


SCENE_RE = re.compile(r"^C\d{3}H\d{3}S(?P<scene>\d{4})$")


@dataclass(frozen=True)
class CalibrationSelection:
    scene_num: int
    scene_name: str
    directory: Path
    candidates: tuple[int, ...]
    rejected_normal_scenes: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["directory"] = str(self.directory)
        return data


def calibration_scene_num(path: Path) -> int | None:
    match = SCENE_RE.match(path.name)
    return int(match.group("scene")) if match else None


def _has_required_files(path: Path) -> bool:
    return (path / "coeffs.txt").is_file() and (path / "focal_adjustment.txt").is_file()


def _image_count_at_least(directory: Path, threshold: int) -> int:
    if not directory.is_dir():
        return 0
    count = 0
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in {".bmp", ".png", ".tif", ".tiff"}:
            count += 1
            if count >= threshold:
                break
    return count


def _is_260212_normal_scene(data_root: Path | None, scene_num: int) -> bool:
    """Implement the dataset-specific rule recorded in AGENTS.md.

    A regular 260212 acquisition has 10,917 frames and must never be selected
    as the preceding calibration scene.
    """
    if data_root is None:
        return False
    scene = f"H001S{scene_num:04d}"
    counts = [
        _image_count_at_least(data_root / "00_img" / f"C001{scene}", 10917),
        _image_count_at_least(data_root / "00_img" / f"C002{scene}", 10917),
    ]
    return any(count >= 10917 for count in counts)


def resolve_calibration(
    calibration_root: Path,
    target_scene_num: int,
    *,
    data_date: str = "",
    data_root: Path | None = None,
) -> CalibrationSelection:
    """Select the latest valid calibration before the target scene.

    No numerical fallback is allowed.  Missing or non-matching calibration
    data is therefore detected before any image processing starts.
    """
    calibration_root = calibration_root.expanduser().resolve()
    data_root = data_root.expanduser().resolve() if data_root is not None else None
    direct_num = calibration_scene_num(calibration_root)
    if direct_num is not None and _has_required_files(calibration_root):
        candidates = [(direct_num, calibration_root)]
    else:
        candidates = []
        if calibration_root.is_dir():
            for child in calibration_root.iterdir():
                num = calibration_scene_num(child)
                if child.is_dir() and num is not None and _has_required_files(child):
                    candidates.append((num, child.resolve()))

    candidates.sort(key=lambda item: item[0])
    preceding = [(num, path) for num, path in candidates if num < int(target_scene_num)]
    rejected: list[int] = []
    if data_date == "260212":
        kept: list[tuple[int, Path]] = []
        for num, path in preceding:
            if _is_260212_normal_scene(data_root, num):
                rejected.append(num)
            else:
                kept.append((num, path))
        preceding = kept

    if not preceding:
        valid_nums = [num for num, _path in candidates]
        raise RuntimeError(
            "No matching calibration scene exists before target scene "
            f"S{int(target_scene_num):04d}. calibration_root={calibration_root}, "
            f"valid_scenes={valid_nums}, rejected_regular_260212_scenes={rejected}. "
            "Processing was stopped before background removal."
        )

    num, path = preceding[-1]
    return CalibrationSelection(
        scene_num=num,
        scene_name=path.name,
        directory=path,
        candidates=tuple(num for num, _path in candidates),
        rejected_normal_scenes=tuple(rejected),
    )
