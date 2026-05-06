from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


MeasurementToolType = Literal["line", "rect", "ellipse", "angle", "curve", "freeform"]
MeasurementUnit = Literal["mm", "px"]
MeasurementAreaUnit = Literal["mm2", "px2"]


@dataclass(frozen=True)
class MeasurementPoint:
    x: float
    y: float


@dataclass(frozen=True)
class MeasurementSliceContext:
    kind: Literal["stack", "mpr"]
    slice_index: int
    sop_instance_uid: str | None = None


@dataclass(frozen=True)
class MeasurementMetrics:
    unit: MeasurementUnit
    area_unit: MeasurementAreaUnit
    length: float | None = None
    width: float | None = None
    height: float | None = None
    area: float | None = None
    angle_degrees: float | None = None
    mean: float | None = None
    standard_deviation: float | None = None
    minimum: float | None = None
    maximum: float | None = None


@dataclass(frozen=True)
class MeasurementRecord:
    measurement_id: str
    tool_type: MeasurementToolType
    points: tuple[MeasurementPoint, ...]
    slice_context: MeasurementSliceContext
    metrics: MeasurementMetrics
    label_anchor: MeasurementPoint
    label_lines: tuple[str, ...] = field(default_factory=tuple)
