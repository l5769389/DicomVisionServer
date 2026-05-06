from app.services.measurement_geometry import build_smooth_path_points


def test_build_smooth_path_points_samples_curve_between_control_points() -> None:
    points = (
        (0.1, 0.8),
        (0.5, 0.1),
        (0.9, 0.8),
    )

    sampled = build_smooth_path_points(points, samples_per_segment=2)

    assert len(sampled) == 5
    assert sampled[0] == points[0]
    assert abs(sampled[1][0] - 0.275) < 1e-9
    assert abs(sampled[1][1] - 0.40625) < 1e-9
    assert sampled[-1] == points[-1]


def test_build_smooth_path_points_closes_freeform_paths() -> None:
    points = (
        (0.2, 0.2),
        (0.6, 0.2),
        (0.5, 0.6),
    )

    sampled = build_smooth_path_points(points, close_path=True, samples_per_segment=1)

    assert sampled[0] == points[0]
    assert sampled[-1] == points[0]
