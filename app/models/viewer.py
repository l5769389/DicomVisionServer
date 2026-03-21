from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class InstanceRecord:
    path: Path
    sop_instance_uid: str | None
    instance_number: int
    rows: int | None
    columns: int | None


@dataclass
class SeriesRecord:
    series_id: str
    folder_path: str
    series_instance_uid: str | None
    study_instance_uid: str | None
    patient_id: str | None
    modality: str | None
    series_description: str | None
    instances: list[InstanceRecord] = field(default_factory=list)


@dataclass
class ViewRecord:
    view_id: str
    series_id: str
    view_type: str
    width: int | None = None
    height: int | None = None
    current_index: int = 0
    zoom: float = 1.0
    offset_x: float = 0.0
    offset_y: float = 0.0
    hor_flip: bool = False
    ver_flip: bool = False
