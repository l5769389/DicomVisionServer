from app.services.measurement_rules import (
    get_measurement_point_requirement,
    has_required_measurement_points,
    is_point_sequence_tool,
)


def test_measurement_point_requirements() -> None:
    assert get_measurement_point_requirement("line").min_points == 2
    assert get_measurement_point_requirement("alignment-horizontal").min_points == 2
    assert get_measurement_point_requirement("alignment-vertical").min_points == 2
    assert get_measurement_point_requirement("angle").min_points == 3
    assert get_measurement_point_requirement("curve").accepts_more_points is True
    assert get_measurement_point_requirement("freeform").min_points == 3


def test_has_required_measurement_points() -> None:
    assert has_required_measurement_points("line", 2) is True
    assert has_required_measurement_points("alignment-horizontal", 2) is True
    assert has_required_measurement_points("line", 3) is False
    assert has_required_measurement_points("curve", 4) is True
    assert has_required_measurement_points("freeform", 2) is False
    assert is_point_sequence_tool("curve") is True
