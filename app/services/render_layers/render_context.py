from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np
from PIL import Image

from app.models.measurement import MeasurementRecord
from app.models.viewer import InstanceRecord, ViewRecord
from app.services.dicom_cache import CachedDicom
from app.services.viewport_transformer import AffineTransform

LayerSpace = Literal["image", "screen"]
ColorRGBA = tuple[int, int, int, int]


@dataclass(frozen=True)
class MprCrosshairOverlay:
    width: int
    height: int
    image_left: float
    image_top: float
    image_width: float
    image_height: float
    horizontal_position: float | None
    horizontal_color: ColorRGBA | None
    vertical_position: float | None
    vertical_color: ColorRGBA | None
    horizontal_angle_rad: float = 0.0
    vertical_angle_rad: float = 1.5707963267948966
    center_x: float | None = None
    center_y: float | None = None
    is_active: bool = False


@dataclass(frozen=True)
class CornerInfoOverlay:
    top_left: tuple[str, ...] = field(default_factory=tuple)
    top_right: tuple[str, ...] = field(default_factory=tuple)
    bottom_left: tuple[str, ...] = field(default_factory=tuple)
    bottom_right: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class OrientationOverlay:
    top: str | None = None
    right: str | None = None
    bottom: str | None = None
    left: str | None = None


@dataclass(frozen=True)
class RenderContext:
    view: ViewRecord
    source_pixels: np.ndarray
    pixel_min: float
    pixel_max: float
    image_transform: AffineTransform
    instance: InstanceRecord | None = None
    cached: CachedDicom | None = None
    mpr_viewport: str | None = None
    measurements: tuple[MeasurementRecord, ...] = ()
    mpr_crosshair: MprCrosshairOverlay | None = None
    corner_info: CornerInfoOverlay | None = None
    orientation: OrientationOverlay | None = None


class RenderLayer(Protocol):
    name: str
    space: LayerSpace
    resample: Image.Resampling

    def render(self, context: RenderContext) -> Image.Image | None: ...
