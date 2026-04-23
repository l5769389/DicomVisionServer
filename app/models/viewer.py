from dataclasses import dataclass, field
from pathlib import Path


from app.models.measurement import MeasurementRecord


Quaternion = tuple[float, float, float, float]
Vec3 = tuple[float, float, float]
Mat3 = tuple[Vec3, Vec3, Vec3]


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
class ViewTransformState:
    zoom: float = 1.0
    offset_x: float = 0.0
    offset_y: float = 0.0
    rotation_degrees: int = 0
    rotation_quaternion: Quaternion = (0.0, 0.0, 0.0, 1.0)
    hor_flip: bool = False
    ver_flip: bool = False
    volume_preset: str = "aaa"
    volume_render_config: dict[str, object] | None = None


@dataclass
class WindowState:
    window_width: float | None = None
    window_center: float | None = None


@dataclass
class DragState:
    drag_origin_zoom: float | None = None
    drag_origin_offset_x: float | None = None
    drag_origin_offset_y: float | None = None
    drag_origin_rotation_quaternion: Quaternion | None = None
    drag_origin_arcball_x: float | None = None
    drag_origin_arcball_y: float | None = None
    drag_origin_window_width: float | None = None
    drag_origin_window_center: float | None = None
    drag_origin_volume_render_config: dict[str, object] | None = None


@dataclass
class MprMipViewportState:
    thickness: int = 12


@dataclass
class MprMipState:
    enabled: bool = False
    algorithm: str = "maximum"
    viewports: dict[str, MprMipViewportState] = field(
        default_factory=lambda: {
            "mpr-ax": MprMipViewportState(),
            "mpr-cor": MprMipViewportState(),
            "mpr-sag": MprMipViewportState(),
        }
    )


@dataclass
class MprObliquePlaneState:
    row: tuple[float, float, float]
    col: tuple[float, float, float]
    normal: tuple[float, float, float]
    is_oblique: bool = False


@dataclass
class MprFrameState:
    center: Vec3
    axis_slice: Vec3
    axis_row: Vec3
    axis_col: Vec3


@dataclass
class MprCursorRecord:
    center_world: Vec3
    reference_center_world: Vec3
    orientation_world: Mat3
    linked_to_volume_rotation: bool = False


@dataclass
class MprRotationDragRecord:
    viewport: str
    line: str
    start_angle_rad: float
    start_cursor: MprCursorRecord


def create_default_mpr_frame_state() -> MprFrameState:
    return MprFrameState(
        center=(0.0, 0.0, 0.0),
        axis_slice=(1.0, 0.0, 0.0),
        axis_row=(0.0, 1.0, 0.0),
        axis_col=(0.0, 0.0, 1.0),
    )


@dataclass
class ViewGroupRecord:
    group_id: str
    group_type: str
    series_id: str
    active_viewport: str = "mpr-ax"
    axial_index: int = 0
    coronal_index: int = 0
    sagittal_index: int = 0
    window: WindowState = field(default_factory=WindowState)
    drag_origin_window_width: float | None = None
    drag_origin_window_center: float | None = None
    drag_origin_volume_render_config: dict[str, object] | None = None
    crosshair_drag_active: bool = False
    crosshair_drag_origin_center: tuple[float, float, float] | None = None
    crosshair_drag_origin_image: tuple[float, float] | None = None
    oblique_drag_active: bool = False
    oblique_source_viewport: str | None = None
    oblique_source_line: str | None = None
    mpr_mip: MprMipState = field(default_factory=MprMipState)
    mpr_cursor: MprCursorRecord | None = None
    rotation_drag: MprRotationDragRecord | None = None
    mpr_reference_center: tuple[float, float, float] | None = None


@dataclass
class ViewRecord:
    view_id: str
    series_id: str
    view_type: str
    pseudocolor_preset: str = "bw"
    width: int | None = None
    height: int | None = None
    current_index: int = 0
    transform: ViewTransformState = field(default_factory=ViewTransformState)
    window: WindowState = field(default_factory=WindowState)
    drag: DragState = field(default_factory=DragState)
    view_group: ViewGroupRecord | None = None
    measurements: list[MeasurementRecord] = field(default_factory=list)
    is_initialized: bool = False

    @property
    def zoom(self) -> float:
        return self.transform.zoom

    @zoom.setter
    def zoom(self, value: float) -> None:
        self.transform.zoom = value

    @property
    def offset_x(self) -> float:
        return self.transform.offset_x

    @offset_x.setter
    def offset_x(self, value: float) -> None:
        self.transform.offset_x = value

    @property
    def offset_y(self) -> float:
        return self.transform.offset_y

    @offset_y.setter
    def offset_y(self, value: float) -> None:
        self.transform.offset_y = value

    @property
    def rotation_quaternion(self) -> Quaternion:
        return self.transform.rotation_quaternion

    @rotation_quaternion.setter
    def rotation_quaternion(self, value: Quaternion) -> None:
        self.transform.rotation_quaternion = value

    @property
    def rotation_degrees(self) -> int:
        return self.transform.rotation_degrees

    @rotation_degrees.setter
    def rotation_degrees(self, value: int) -> None:
        self.transform.rotation_degrees = value

    @property
    def volume_preset(self) -> str:
        return self.transform.volume_preset

    @volume_preset.setter
    def volume_preset(self, value: str) -> None:
        self.transform.volume_preset = value

    @property
    def volume_render_config(self) -> dict[str, object] | None:
        return self.transform.volume_render_config

    @volume_render_config.setter
    def volume_render_config(self, value: dict[str, object] | None) -> None:
        self.transform.volume_render_config = value

    @property
    def hor_flip(self) -> bool:
        return self.transform.hor_flip

    @hor_flip.setter
    def hor_flip(self, value: bool) -> None:
        self.transform.hor_flip = value

    @property
    def ver_flip(self) -> bool:
        return self.transform.ver_flip

    @ver_flip.setter
    def ver_flip(self, value: bool) -> None:
        self.transform.ver_flip = value

    @property
    def window_width(self) -> float | None:
        if self.view_group is not None:
            return self.view_group.window.window_width
        return self.window.window_width

    @window_width.setter
    def window_width(self, value: float | None) -> None:
        if self.view_group is not None:
            self.view_group.window.window_width = value
            return
        self.window.window_width = value

    @property
    def window_center(self) -> float | None:
        if self.view_group is not None:
            return self.view_group.window.window_center
        return self.window.window_center

    @window_center.setter
    def window_center(self, value: float | None) -> None:
        if self.view_group is not None:
            self.view_group.window.window_center = value
            return
        self.window.window_center = value

    @property
    def drag_origin_zoom(self) -> float | None:
        return self.drag.drag_origin_zoom

    @drag_origin_zoom.setter
    def drag_origin_zoom(self, value: float | None) -> None:
        self.drag.drag_origin_zoom = value

    @property
    def drag_origin_offset_x(self) -> float | None:
        return self.drag.drag_origin_offset_x

    @drag_origin_offset_x.setter
    def drag_origin_offset_x(self, value: float | None) -> None:
        self.drag.drag_origin_offset_x = value

    @property
    def drag_origin_offset_y(self) -> float | None:
        return self.drag.drag_origin_offset_y

    @drag_origin_offset_y.setter
    def drag_origin_offset_y(self, value: float | None) -> None:
        self.drag.drag_origin_offset_y = value

    @property
    def drag_origin_rotation_quaternion(self) -> Quaternion | None:
        return self.drag.drag_origin_rotation_quaternion

    @drag_origin_rotation_quaternion.setter
    def drag_origin_rotation_quaternion(self, value: Quaternion | None) -> None:
        self.drag.drag_origin_rotation_quaternion = value

    @property
    def drag_origin_arcball_x(self) -> float | None:
        return self.drag.drag_origin_arcball_x

    @drag_origin_arcball_x.setter
    def drag_origin_arcball_x(self, value: float | None) -> None:
        self.drag.drag_origin_arcball_x = value

    @property
    def drag_origin_arcball_y(self) -> float | None:
        return self.drag.drag_origin_arcball_y

    @drag_origin_arcball_y.setter
    def drag_origin_arcball_y(self, value: float | None) -> None:
        self.drag.drag_origin_arcball_y = value

    @property
    def drag_origin_window_width(self) -> float | None:
        if self.view_group is not None:
            return self.view_group.drag_origin_window_width
        return self.drag.drag_origin_window_width

    @drag_origin_window_width.setter
    def drag_origin_window_width(self, value: float | None) -> None:
        if self.view_group is not None:
            self.view_group.drag_origin_window_width = value
            return
        self.drag.drag_origin_window_width = value

    @property
    def drag_origin_window_center(self) -> float | None:
        if self.view_group is not None:
            return self.view_group.drag_origin_window_center
        return self.drag.drag_origin_window_center

    @drag_origin_window_center.setter
    def drag_origin_window_center(self, value: float | None) -> None:
        if self.view_group is not None:
            self.view_group.drag_origin_window_center = value
            return
        self.drag.drag_origin_window_center = value

    @property
    def drag_origin_volume_render_config(self) -> dict[str, object] | None:
        return self.drag.drag_origin_volume_render_config

    @drag_origin_volume_render_config.setter
    def drag_origin_volume_render_config(self, value: dict[str, object] | None) -> None:
        self.drag.drag_origin_volume_render_config = value

    @property
    def mpr_active_viewport(self) -> str:
        return self.view_group.active_viewport if self.view_group is not None else "mpr-ax"

    @mpr_active_viewport.setter
    def mpr_active_viewport(self, value: str) -> None:
        if self.view_group is not None:
            self.view_group.active_viewport = value

    @property
    def mpr_axial_index(self) -> int:
        return self.view_group.axial_index if self.view_group is not None else 0

    @mpr_axial_index.setter
    def mpr_axial_index(self, value: int) -> None:
        if self.view_group is not None:
            self.view_group.axial_index = value
            self.view_group.mpr_cursor = None

    @property
    def mpr_coronal_index(self) -> int:
        return self.view_group.coronal_index if self.view_group is not None else 0

    @mpr_coronal_index.setter
    def mpr_coronal_index(self, value: int) -> None:
        if self.view_group is not None:
            self.view_group.coronal_index = value
            self.view_group.mpr_cursor = None

    @property
    def mpr_sagittal_index(self) -> int:
        return self.view_group.sagittal_index if self.view_group is not None else 0

    @mpr_sagittal_index.setter
    def mpr_sagittal_index(self, value: int) -> None:
        if self.view_group is not None:
            self.view_group.sagittal_index = value
            self.view_group.mpr_cursor = None

    @property
    def mpr_crosshair_drag_active(self) -> bool:
        return self.view_group.crosshair_drag_active if self.view_group is not None else False

    @mpr_crosshair_drag_active.setter
    def mpr_crosshair_drag_active(self, value: bool) -> None:
        if self.view_group is not None:
            self.view_group.crosshair_drag_active = value

    @property
    def mpr_mip(self) -> MprMipState:
        if self.view_group is not None:
            return self.view_group.mpr_mip
        return MprMipState()

