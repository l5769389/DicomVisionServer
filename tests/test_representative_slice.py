from pathlib import Path
from types import SimpleNamespace

import numpy as np

from app.models.viewer import InstanceRecord, SeriesRecord, ViewGroupRecord
from app.services.viewer_service import ViewerService


def _build_instance(index: int) -> InstanceRecord:
    return InstanceRecord(
        path=Path(f"slice-{index}.dcm"),
        sop_instance_uid=f"1.2.3.{index}",
        instance_number=index + 1,
        rows=16,
        columns=16,
    )


def _build_series(instance_count: int) -> SeriesRecord:
    return SeriesRecord(
        series_id="series-content",
        folder_path=".",
        series_instance_uid="1.2.3.series",
        study_instance_uid=None,
        patient_id=None,
        patient_name=None,
        study_date=None,
        study_description=None,
        accession_number=None,
        modality="CT",
        series_description=None,
        instances=[_build_instance(index) for index in range(instance_count)],
    )


def test_representative_stack_index_prefers_slice_with_real_content(monkeypatch) -> None:
    background = np.full((16, 16), -1000.0, dtype=np.float32)
    content = background.copy()
    content[4:12, 4:12] = 90.0
    content[6:10, 6:10] = 420.0
    pixels_by_uid = {
        "1.2.3.0": background,
        "1.2.3.1": background,
        "1.2.3.2": content,
        "1.2.3.3": background,
        "1.2.3.4": background,
    }

    fake_cache = SimpleNamespace(
        get=lambda instance_uid, path: SimpleNamespace(source_pixels=pixels_by_uid[instance_uid])
    )
    monkeypatch.setattr("app.services.viewer_service.dicom_cache", fake_cache)

    service = ViewerService()
    assert service._resolve_representative_stack_index(_build_series(5)) == 2


def test_representative_stack_index_falls_back_to_middle_for_blank_series(monkeypatch) -> None:
    background = np.full((16, 16), -1000.0, dtype=np.float32)
    fake_cache = SimpleNamespace(get=lambda instance_uid, path: SimpleNamespace(source_pixels=background))
    monkeypatch.setattr("app.services.viewer_service.dicom_cache", fake_cache)

    service = ViewerService()
    assert service._resolve_representative_stack_index(_build_series(5)) == 2


def test_mpr_group_reset_uses_volume_center_not_representative_slice() -> None:
    service = ViewerService()
    group = ViewGroupRecord(group_id="mpr-group", group_type="MPR", series_id="series-content")

    service._reset_mpr_group_geometry(group, (9, 11, 13))

    assert group.axial_index == 4
    assert group.coronal_index == 5
    assert group.sagittal_index == 6
    assert group.mpr_reference_center == (4.0, 5.0, 6.0)
