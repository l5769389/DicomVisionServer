from pathlib import Path

import pytest

from app.models.viewer import InstanceRecord
from app.services.dicom_compatibility import get_instances_volume_compatibility


def _instance(index: int, **overrides: object) -> InstanceRecord:
    values: dict[str, object] = {
        "path": Path(f"slice-{index}.dcm"),
        "sop_instance_uid": f"1.2.3.{index}",
        "instance_number": index,
        "rows": 128,
        "columns": 256,
        "pixel_spacing": (0.8, 0.6),
        "image_orientation_patient": (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        "image_position_patient": (0.0, 0.0, float(index - 1)),
        "number_of_frames": 1,
    }
    values.update(overrides)
    return InstanceRecord(**values)


def test_volume_compatibility_accepts_consistent_in_plane_geometry() -> None:
    result = get_instances_volume_compatibility([_instance(1), _instance(2), _instance(3)])

    assert result.supported is True


@pytest.mark.parametrize(
    ("overrides", "blocked_code"),
    [
        ({"rows": 64}, "mixed-image-size"),
        ({"pixel_spacing": None}, "missing-pixel-spacing"),
        ({"pixel_spacing": (0.9, 0.6)}, "mixed-pixel-spacing"),
        (
            {"image_orientation_patient": (0.0, 1.0, 0.0, -1.0, 0.0, 0.0)},
            "mixed-slice-orientations",
        ),
        (
            {"image_orientation_patient": (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0)},
            "mixed-slice-orientations",
        ),
    ],
)
def test_volume_compatibility_blocks_inconsistent_in_plane_geometry(
    overrides: dict[str, object],
    blocked_code: str,
) -> None:
    result = get_instances_volume_compatibility([_instance(1), _instance(2, **overrides), _instance(3)])

    assert result.supported is False
    assert result.blocked_code == blocked_code


@pytest.mark.parametrize(
    "overrides",
    [
        {"rows": None},
        {"columns": 0},
        {"pixel_spacing": (0.0, 0.6)},
        {"image_orientation_patient": (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)},
        {"image_position_patient": (0.0, float("nan"), 1.0)},
    ],
)
def test_volume_compatibility_blocks_invalid_geometry(overrides: dict[str, object]) -> None:
    result = get_instances_volume_compatibility([_instance(1), _instance(2, **overrides)])

    assert result.supported is False
