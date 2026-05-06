from __future__ import annotations

from dataclasses import dataclass

from app.models.measurement import MeasurementToolType


POINT_SEQUENCE_TOOL_TYPES: frozenset[MeasurementToolType] = frozenset({"curve", "freeform"})


@dataclass(frozen=True)
class MeasurementPointRequirement:
    min_points: int
    accepts_more_points: bool


def is_point_sequence_tool(tool_type: MeasurementToolType | str) -> bool:
    return tool_type in POINT_SEQUENCE_TOOL_TYPES


def get_measurement_point_requirement(tool_type: MeasurementToolType | str) -> MeasurementPointRequirement:
    if tool_type == "angle":
        return MeasurementPointRequirement(min_points=3, accepts_more_points=False)
    if tool_type == "curve":
        return MeasurementPointRequirement(min_points=2, accepts_more_points=True)
    if tool_type == "freeform":
        return MeasurementPointRequirement(min_points=3, accepts_more_points=True)
    return MeasurementPointRequirement(min_points=2, accepts_more_points=False)


def has_required_measurement_points(tool_type: MeasurementToolType | str, point_count: int) -> bool:
    requirement = get_measurement_point_requirement(tool_type)
    return (
        point_count >= requirement.min_points
        if requirement.accepts_more_points
        else point_count == requirement.min_points
    )
