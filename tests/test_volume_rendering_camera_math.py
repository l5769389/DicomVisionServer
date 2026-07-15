from dataclasses import replace
from math import acos, degrees
from types import SimpleNamespace

import numpy as np
import pytest
from vtkmodules.vtkRenderingCore import vtkCamera

from app.models.viewer import ViewRecord
from app.core import VIEW_OP_TYPE_ROTATE_3D
from app.schemas.view import ViewOperationRequest
from app.services.volume_rendering.camera_math import (
    ANATOMICAL_ORIENTATION_FACES,
    DIRECT_MODEL_TRACKBALL_MOTION_FACTOR,
    VTK_TRACKBALL_MOTION_FACTOR,
    anatomical_orientation_quaternion,
    apply_direct_model_trackball_control_points_to_quaternion,
    apply_direct_model_trackball_delta_to_quaternion,
    apply_vtk_trackball_camera_delta_to_quaternion,
    normalize_quaternion,
    quaternion_to_rotation_matrix,
    resolve_direct_model_trackball_control_point,
    resolve_anatomical_orientation_face,
    rotation_matrix_to_quaternion,
)
from app.services.volume_rendering.camera_fit import (
    fit_distance_for_bounds,
    fit_parallel_scale_for_bounds,
    fit_stable_distance_for_bounds,
    fit_stable_parallel_scale_for_bounds,
)
from app.services.volume_rendering.contracts import VolumeRenderRequest
from app.services.volume_rendering.vtk_volume_renderer import VtkVolumeRenderer
from app.services.viewer_service import ViewerService


def test_camera_quaternion_math_round_trips_rotation_matrix() -> None:
    quaternion = normalize_quaternion((0.2, 0.3, 0.1, 0.9))
    matrix = quaternion_to_rotation_matrix(quaternion)

    assert rotation_matrix_to_quaternion(matrix) == pytest.approx(quaternion)


def test_anatomical_orientation_quaternions_face_the_camera_and_round_trip() -> None:
    face_normals = {
        "A": np.asarray([0.0, -1.0, 0.0]),
        "P": np.asarray([0.0, 1.0, 0.0]),
        "L": np.asarray([1.0, 0.0, 0.0]),
        "R": np.asarray([-1.0, 0.0, 0.0]),
        "S": np.asarray([0.0, 0.0, 1.0]),
        "I": np.asarray([0.0, 0.0, -1.0]),
    }
    face_up = {
        "A": np.asarray([0.0, 0.0, 1.0]),
        "P": np.asarray([0.0, 0.0, 1.0]),
        "L": np.asarray([0.0, 0.0, 1.0]),
        "R": np.asarray([0.0, 0.0, 1.0]),
        "S": np.asarray([0.0, -1.0, 0.0]),
        "I": np.asarray([0.0, -1.0, 0.0]),
    }

    for face in ANATOMICAL_ORIENTATION_FACES:
        quaternion = anatomical_orientation_quaternion(face)
        assert quaternion is not None
        rotation = quaternion_to_rotation_matrix(quaternion)
        assert rotation @ face_normals[face] == pytest.approx((0.0, -1.0, 0.0))
        assert rotation @ face_up[face] == pytest.approx((0.0, 0.0, 1.0))
        assert resolve_anatomical_orientation_face(quaternion) == face


def test_anatomical_orientation_operation_preserves_non_rotation_3d_state() -> None:
    service = ViewerService()
    view = ViewRecord(view_id="orientation-view", series_id="series", view_type="3D")
    view.zoom = 1.6
    view.offset_x = 17.0
    view.offset_y = -9.0
    view.render_3d_mode = "surface"
    view.volume_remove_bed = True
    view.volume_clip_mode = "inside"
    view.drag_origin_arcball_x = 0.2
    view.drag_origin_rotation_quaternion = (0.0, 0.0, 0.0, 1.0)

    changed = service._handle_anatomical_orientation(
        view,
        ViewOperationRequest(
            viewId=view.view_id,
            opType=VIEW_OP_TYPE_ROTATE_3D,
            subOpType="orientation:L",
        ),
    )

    assert changed is True
    assert resolve_anatomical_orientation_face(view.rotation_quaternion) == "L"
    assert view.zoom == pytest.approx(1.6)
    assert view.offset_x == pytest.approx(17.0)
    assert view.offset_y == pytest.approx(-9.0)
    assert view.render_3d_mode == "surface"
    assert view.volume_remove_bed is True
    assert view.volume_clip_mode == "inside"
    assert view.drag_origin_arcball_x is None
    assert view.drag_origin_rotation_quaternion is None

    assert service._handle_anatomical_orientation(
        view,
        ViewOperationRequest(
            viewId=view.view_id,
            opType=VIEW_OP_TYPE_ROTATE_3D,
            subOpType="orientation:invalid",
        ),
    ) is False


def test_vtk_trackball_motion_factor_matches_vtk_default() -> None:
    assert VTK_TRACKBALL_MOTION_FACTOR == pytest.approx(10.0)

    updated = apply_vtk_trackball_camera_delta_to_quaternion(
        (0.0, 0.0, 0.0, 1.0),
        delta_x_pixels=100.0,
        delta_y_pixels=0.0,
        canvas_width=1000.0,
        canvas_height=800.0,
    )

    assert updated != pytest.approx((0.0, 0.0, 0.0, 1.0))
    assert np.linalg.norm(np.asarray(updated)) == pytest.approx(1.0)
    assert degrees(2.0 * acos(abs(updated[3]))) == pytest.approx(20.0)


def test_3d_camera_fit_uses_model_bounds_and_viewport_aspect() -> None:
    bounds = (0.0, 260.0, 0.0, 80.0, 0.0, 120.0)

    landscape_distance = fit_distance_for_bounds(
        bounds,
        view_angle_degrees=30.0,
        aspect_ratio=16.0 / 9.0,
    )
    square_distance = fit_distance_for_bounds(
        bounds,
        view_angle_degrees=30.0,
        aspect_ratio=1.0,
    )
    portrait_distance = fit_distance_for_bounds(
        bounds,
        view_angle_degrees=30.0,
        aspect_ratio=9.0 / 16.0,
    )

    assert landscape_distance < square_distance < portrait_distance
    assert fit_parallel_scale_for_bounds(bounds, aspect_ratio=9.0 / 16.0) > fit_parallel_scale_for_bounds(bounds, aspect_ratio=16.0 / 9.0)


def test_3d_stable_camera_fit_uses_viewport_without_rotation_breathing() -> None:
    bounds = (0.0, 260.0, 0.0, 80.0, 0.0, 120.0)

    landscape_distance = fit_stable_distance_for_bounds(
        bounds,
        view_angle_degrees=30.0,
        aspect_ratio=16.0 / 9.0,
    )
    portrait_distance = fit_stable_distance_for_bounds(
        bounds,
        view_angle_degrees=30.0,
        aspect_ratio=9.0 / 16.0,
    )
    rolled_projection_distance = fit_distance_for_bounds(
        bounds,
        view_angle_degrees=30.0,
        aspect_ratio=9.0 / 16.0,
        rotation_matrix=quaternion_to_rotation_matrix(normalize_quaternion((0.0, 0.0, 0.38, 0.92))),
    )

    assert portrait_distance > landscape_distance
    assert portrait_distance >= rolled_projection_distance
    assert fit_stable_parallel_scale_for_bounds(bounds, aspect_ratio=9.0 / 16.0) > fit_stable_parallel_scale_for_bounds(bounds, aspect_ratio=16.0 / 9.0)


def _vtk_camera_trackball_oracle(
    *,
    delta_x_pixels: float,
    delta_y_pixels: float,
    canvas_width: float,
    canvas_height: float,
) -> tuple[float, float, float, float]:
    camera = vtkCamera()
    camera.SetFocalPoint(0.0, 0.0, 0.0)
    camera.SetPosition(0.0, -10.0, 0.0)
    camera.SetViewUp(0.0, 0.0, 1.0)
    camera.Azimuth(delta_x_pixels * -20.0 / canvas_width * VTK_TRACKBALL_MOTION_FACTOR)
    camera.Elevation(delta_y_pixels * -20.0 / canvas_height * VTK_TRACKBALL_MOTION_FACTOR)
    camera.OrthogonalizeViewUp()

    current_forward = np.asarray(camera.GetFocalPoint(), dtype=np.float64) - np.asarray(camera.GetPosition(), dtype=np.float64)
    current_forward /= np.linalg.norm(current_forward)
    current_up = np.asarray(camera.GetViewUp(), dtype=np.float64)
    current_up /= np.linalg.norm(current_up)
    current_right = np.cross(current_forward, current_up)
    current_right /= np.linalg.norm(current_right)
    current_up = np.cross(current_right, current_forward)
    current_up /= np.linalg.norm(current_up)

    camera_rotation_matrix = np.column_stack((current_right, current_forward, current_up))
    return rotation_matrix_to_quaternion(camera_rotation_matrix.T)


def test_vtk_trackball_delta_matches_vtk_camera_oracle() -> None:
    expected = _vtk_camera_trackball_oracle(
        delta_x_pixels=100.0,
        delta_y_pixels=50.0,
        canvas_width=1000.0,
        canvas_height=800.0,
    )
    actual = apply_vtk_trackball_camera_delta_to_quaternion(
        (0.0, 0.0, 0.0, 1.0),
        delta_x_pixels=100.0,
        delta_y_pixels=50.0,
        canvas_width=1000.0,
        canvas_height=800.0,
    )

    assert actual == pytest.approx(expected)


def _project_front_point_to_screen(quaternion: tuple[float, float, float, float]) -> tuple[float, float]:
    rotated = quaternion_to_rotation_matrix(quaternion) @ np.asarray([0.0, -1.0, 0.0], dtype=np.float64)
    return float(rotated[0]), float(rotated[2])


def _delta_rotation_matrix(
    before: tuple[float, float, float, float],
    after: tuple[float, float, float, float],
) -> np.ndarray:
    return quaternion_to_rotation_matrix(after) @ quaternion_to_rotation_matrix(before).T


def test_direct_model_trackball_motion_factor_matches_vtk_sensitivity_scale() -> None:
    assert DIRECT_MODEL_TRACKBALL_MOTION_FACTOR == pytest.approx(10.0)

    updated = apply_direct_model_trackball_delta_to_quaternion(
        (0.0, 0.0, 0.0, 1.0),
        delta_x_pixels=100.0,
        delta_y_pixels=0.0,
        canvas_width=1000.0,
        canvas_height=800.0,
    )

    assert updated != pytest.approx((0.0, 0.0, 0.0, 1.0))
    assert np.linalg.norm(np.asarray(updated)) == pytest.approx(1.0)
    assert degrees(2.0 * acos(abs(updated[3]))) == pytest.approx(20.0)


def test_direct_model_trackball_moves_visible_model_with_drag_direction() -> None:
    base_screen_x, base_screen_y = _project_front_point_to_screen((0.0, 0.0, 0.0, 1.0))
    right_drag = apply_direct_model_trackball_delta_to_quaternion(
        (0.0, 0.0, 0.0, 1.0),
        delta_x_pixels=100.0,
        delta_y_pixels=0.0,
        canvas_width=1000.0,
        canvas_height=800.0,
    )
    up_drag = apply_direct_model_trackball_delta_to_quaternion(
        (0.0, 0.0, 0.0, 1.0),
        delta_x_pixels=0.0,
        delta_y_pixels=-80.0,
        canvas_width=1000.0,
        canvas_height=800.0,
    )
    down_drag = apply_direct_model_trackball_delta_to_quaternion(
        (0.0, 0.0, 0.0, 1.0),
        delta_x_pixels=0.0,
        delta_y_pixels=80.0,
        canvas_width=1000.0,
        canvas_height=800.0,
    )

    right_screen_x, _ = _project_front_point_to_screen(right_drag)
    _, up_screen_y = _project_front_point_to_screen(up_drag)
    _, down_screen_y = _project_front_point_to_screen(down_drag)

    assert right_screen_x > base_screen_x
    assert up_screen_y > base_screen_y
    assert down_screen_y < base_screen_y


def test_direct_model_trackball_direction_does_not_flip_after_model_roll() -> None:
    rolled_model = (0.0, -1.0, 0.0, 0.0)
    screen_front = quaternion_to_rotation_matrix(rolled_model) @ np.asarray([0.0, -1.0, 0.0], dtype=np.float64)
    right_drag = apply_direct_model_trackball_delta_to_quaternion(
        rolled_model,
        delta_x_pixels=100.0,
        delta_y_pixels=0.0,
        canvas_width=1000.0,
        canvas_height=800.0,
    )
    up_drag = apply_direct_model_trackball_delta_to_quaternion(
        rolled_model,
        delta_x_pixels=0.0,
        delta_y_pixels=-80.0,
        canvas_width=1000.0,
        canvas_height=800.0,
    )

    right_delta = _delta_rotation_matrix(rolled_model, right_drag)
    up_delta = _delta_rotation_matrix(rolled_model, up_drag)

    assert (right_delta @ screen_front)[0] > screen_front[0]
    assert (up_delta @ screen_front)[2] > screen_front[2]


def test_direct_model_trackball_control_point_follows_pointer() -> None:
    origin_quaternion = normalize_quaternion((0.13, -0.22, 0.08, 0.96))
    origin_control = resolve_direct_model_trackball_control_point(
        canvas_x=600.0,
        canvas_y=400.0,
        canvas_width=1000.0,
        canvas_height=800.0,
    )
    current_control = resolve_direct_model_trackball_control_point(
        canvas_x=660.0,
        canvas_y=340.0,
        canvas_width=1000.0,
        canvas_height=800.0,
    )

    updated = apply_direct_model_trackball_control_points_to_quaternion(
        origin_quaternion,
        origin_control_point=origin_control,
        current_control_point=current_control,
    )
    origin_rotation = quaternion_to_rotation_matrix(origin_quaternion)
    picked_model_point = origin_rotation.T @ np.asarray(origin_control, dtype=np.float64)
    moved_control_point = quaternion_to_rotation_matrix(updated) @ picked_model_point

    assert moved_control_point == pytest.approx(current_control)


def test_direct_model_trackball_control_point_moves_up_when_pointer_moves_up() -> None:
    origin_control = resolve_direct_model_trackball_control_point(
        canvas_x=500.0,
        canvas_y=400.0,
        canvas_width=1000.0,
        canvas_height=800.0,
    )
    current_control = resolve_direct_model_trackball_control_point(
        canvas_x=500.0,
        canvas_y=320.0,
        canvas_width=1000.0,
        canvas_height=800.0,
    )

    updated = apply_direct_model_trackball_control_points_to_quaternion(
        (0.0, 0.0, 0.0, 1.0),
        origin_control_point=origin_control,
        current_control_point=current_control,
    )
    moved_front_point = quaternion_to_rotation_matrix(updated) @ np.asarray(origin_control, dtype=np.float64)

    assert moved_front_point[2] > origin_control[2]


def test_direct_model_trackball_control_point_direction_does_not_flip_after_roll() -> None:
    rolled_model = (0.0, -1.0, 0.0, 0.0)
    origin_control = resolve_direct_model_trackball_control_point(
        canvas_x=500.0,
        canvas_y=400.0,
        canvas_width=1000.0,
        canvas_height=800.0,
    )
    current_control = resolve_direct_model_trackball_control_point(
        canvas_x=500.0,
        canvas_y=320.0,
        canvas_width=1000.0,
        canvas_height=800.0,
    )

    updated = apply_direct_model_trackball_control_points_to_quaternion(
        rolled_model,
        origin_control_point=origin_control,
        current_control_point=current_control,
    )
    picked_model_point = quaternion_to_rotation_matrix(rolled_model).T @ np.asarray(origin_control, dtype=np.float64)
    moved_control_point = quaternion_to_rotation_matrix(updated) @ picked_model_point

    assert moved_control_point[2] > origin_control[2]


def _build_volume_request(view_id: str) -> VolumeRenderRequest:
    return VolumeRenderRequest(
        view_id=view_id,
        volume=np.zeros((4, 5, 6), dtype=np.float32),
        spacing_xyz=(0.7, 0.8, 1.2),
        canvas_width=200,
        canvas_height=100,
        window_width=400.0,
        window_center=40.0,
        zoom=1.0,
        offset_x=0.0,
        offset_y=0.0,
        rotation_quaternion=(0.0, 0.0, 0.0, 1.0),
    )


def test_volume_renderer_session_lru_finalizes_oldest_session(monkeypatch) -> None:
    renderer = VtkVolumeRenderer()
    finalized = []

    def fake_create_session(volume, spacing_xyz, volume_token):
        del volume, spacing_xyz
        return SimpleNamespace(
            volume_token=volume_token,
            render_window=SimpleNamespace(Finalize=lambda: finalized.append(len(finalized))),
        )

    monkeypatch.setattr(renderer, "_create_session", fake_create_session)

    for index in range(9):
        request = _build_volume_request(f"volume-view-{index}")
        renderer._get_or_create_session(request, request.volume)

    assert len(renderer._sessions) == 8
    assert len(finalized) == 1
    assert ("volume-view-0", "shared") not in renderer._sessions


def test_volume_renderer_drop_session_finalizes_matching_view_session() -> None:
    renderer = VtkVolumeRenderer()
    finalized = []
    renderer._sessions[("view-a", "shared")] = SimpleNamespace(
        render_window=SimpleNamespace(Finalize=lambda: finalized.append("view-a-shared"))
    )
    renderer._sessions[("view-b", "shared")] = SimpleNamespace(
        render_window=SimpleNamespace(Finalize=lambda: finalized.append("view-b"))
    )

    renderer._drop_session_in_executor("view-a")

    assert finalized == ["view-a-shared"]
    assert list(renderer._sessions.keys()) == [("view-b", "shared")]


def test_volume_fast_preview_preserves_source_volume_and_spacing() -> None:
    volume = np.zeros((96, 512, 384), dtype=np.float32)

    sampled, spacing = VtkVolumeRenderer._prepare_render_volume(volume, (0.5, 0.6, 1.2), fast_preview=True)

    assert sampled is volume
    assert sampled.shape == volume.shape
    assert spacing == pytest.approx((0.5, 0.6, 1.2))


def test_volume_preview_and_final_share_session_key(monkeypatch) -> None:
    renderer = VtkVolumeRenderer()
    created: list[object] = []

    def fake_create_session(volume, spacing_xyz, volume_token):
        del volume, spacing_xyz
        created.append(volume_token[0])
        return SimpleNamespace(
            volume_token=volume_token,
            render_window=SimpleNamespace(Finalize=lambda: None),
        )

    monkeypatch.setattr(renderer, "_create_session", fake_create_session)

    request = _build_volume_request("view-a")
    renderer._get_or_create_session(request, request.volume)
    renderer._get_or_create_session(replace(request, fast_preview=True), request.volume)
    renderer._get_or_create_session(request, request.volume)

    assert created == [id(request.volume)]
    assert list(renderer._sessions.keys()) == [("view-a", "shared")]


def test_volume_preview_session_uses_stable_volume_token(monkeypatch) -> None:
    renderer = VtkVolumeRenderer()
    created: list[object] = []

    def fake_create_session(volume, spacing_xyz, volume_token):
        del volume, spacing_xyz
        created.append(volume_token[0])
        return SimpleNamespace(
            volume_token=volume_token,
            render_window=SimpleNamespace(Finalize=lambda: None),
        )

    monkeypatch.setattr(renderer, "_create_session", fake_create_session)

    request = replace(_build_volume_request("view-a"), fast_preview=True, volume_token="series-token")
    renderer._get_or_create_session(request, np.zeros_like(request.volume))
    renderer._get_or_create_session(request, np.ones_like(request.volume))

    assert created == ["series-token"]


def test_volume_renderer_reuses_transfer_and_sampling_configuration(monkeypatch) -> None:
    renderer = VtkVolumeRenderer()
    request = _build_volume_request("view-a")
    calls: list[str] = []
    session = SimpleNamespace(
        canvas_size=(request.canvas_width, request.canvas_height),
        transfer_function_token=None,
        sampling_token=None,
        render_quality_token=None,
    )

    monkeypatch.setattr(renderer, "_update_transfer_functions", lambda *args, **kwargs: calls.append("transfer"))
    monkeypatch.setattr(renderer, "_update_sampling", lambda *args: calls.append("sampling"))
    monkeypatch.setattr(renderer, "_update_render_quality", lambda *args: calls.append("quality"))
    monkeypatch.setattr(renderer, "_update_camera", lambda *args: calls.append("camera"))

    renderer._configure_session(session, request)
    renderer._configure_session(session, request)
    renderer._configure_session(session, replace(request, fast_preview=True))

    assert calls == ["transfer", "sampling", "quality", "camera", "camera", "sampling", "quality", "camera"]


def test_volume_preview_request_caps_canvas_without_changing_final_or_gesture_size() -> None:
    service = ViewerService()
    view = ViewRecord(view_id="view-a", series_id="series-a", view_type="3D")
    view.width = 1600
    view.height = 1000
    view.zoom = 1.8
    view.offset_x = 80.0
    view.offset_y = -40.0
    volume = np.zeros((4, 5, 6), dtype=np.float32)

    preview = service._build_volume_render_request(
        view,
        volume=volume,
        spacing_xyz=(1.0, 1.0, 1.0),
        fast_preview=True,
    )
    gesture = service._build_volume_render_request(
        view,
        volume=volume,
        spacing_xyz=(1.0, 1.0, 1.0),
        fast_preview=True,
        scale_fast_preview_canvas=False,
    )
    final = service._build_volume_render_request(
        view,
        volume=volume,
        spacing_xyz=(1.0, 1.0, 1.0),
        fast_preview=False,
    )

    assert preview.canvas_width == 720
    assert preview.canvas_height == 450
    assert preview.zoom == pytest.approx(1.8)
    assert preview.offset_x == pytest.approx(36.0)
    assert preview.offset_y == pytest.approx(-18.0)
    assert gesture.canvas_width == 1600
    assert gesture.canvas_height == 1000
    assert gesture.offset_x == pytest.approx(80.0)
    assert gesture.offset_y == pytest.approx(-40.0)
    assert final.canvas_width == 1600
    assert final.canvas_height == 1000


def test_volume_mode_3d_rotation_updates_quaternion_without_vtk(monkeypatch) -> None:
    service = ViewerService()
    view = ViewRecord(
        view_id="volume-view",
        series_id="series-a",
        view_type="3D",
        width=200,
        height=100,
    )
    view.render_3d_mode = "volume"

    def fake_apply_trackball_camera_delta(request, *, delta_x_pixels: float, delta_y_pixels: float):
        del request, delta_x_pixels, delta_y_pixels
        raise AssertionError("volume rotate move should not enter VTK")

    monkeypatch.setattr(
        "app.services.viewer_service.vtk_volume_renderer.apply_trackball_camera_delta",
        fake_apply_trackball_camera_delta,
    )

    service._handle_drag_rotate_3d(
        view,
        ViewOperationRequest(
            viewId=view.view_id,
            opType=VIEW_OP_TYPE_ROTATE_3D,
            actionType="start",
            x=0.5,
            y=0.5,
        ),
    )
    service._handle_drag_rotate_3d(
        view,
        ViewOperationRequest(
            viewId=view.view_id,
            opType=VIEW_OP_TYPE_ROTATE_3D,
            actionType="move",
            x=0.58,
            y=0.42,
        ),
    )

    assert view.rotation_quaternion != pytest.approx((0.0, 0.0, 0.0, 1.0))
    assert view.is_initialized is True


def test_rotate_3d_prefers_canvas_coordinates(monkeypatch) -> None:
    service = ViewerService()
    view = ViewRecord(view_id="volume-view", series_id="series-a", view_type="3D", width=200, height=100)
    captured: dict[str, tuple[float, float, float]] = {}

    def fake_apply_direct_model_trackball_control_points_to_quaternion(
        quaternion,
        *,
        origin_control_point: tuple[float, float, float],
        current_control_point: tuple[float, float, float],
    ):
        del quaternion
        captured.update(
            origin_control_point=origin_control_point,
            current_control_point=current_control_point,
        )
        return (0.0, 0.0, 0.0, 1.0)

    monkeypatch.setattr(
        "app.services.viewer_service.apply_direct_model_trackball_control_points_to_quaternion",
        fake_apply_direct_model_trackball_control_points_to_quaternion,
    )

    service._handle_drag_rotate_3d(
        view,
        ViewOperationRequest(
            viewId=view.view_id,
            opType=VIEW_OP_TYPE_ROTATE_3D,
            actionType="start",
            x=0.1,
            y=0.1,
            canvasX=500.0,
            canvasY=250.0,
            canvasWidth=1000.0,
            canvasHeight=500.0,
        ),
    )
    service._handle_drag_rotate_3d(
        view,
        ViewOperationRequest(
            viewId=view.view_id,
            opType=VIEW_OP_TYPE_ROTATE_3D,
            actionType="move",
            x=0.1,
            y=0.1,
            canvasX=530.0,
            canvasY=270.0,
            canvasWidth=1000.0,
            canvasHeight=500.0,
        ),
    )

    assert captured["origin_control_point"] == pytest.approx(
        resolve_direct_model_trackball_control_point(
            canvas_x=500.0,
            canvas_y=250.0,
            canvas_width=1000.0,
            canvas_height=500.0,
        )
    )
    assert captured["current_control_point"] == pytest.approx(
        resolve_direct_model_trackball_control_point(
            canvas_x=530.0,
            canvas_y=270.0,
            canvas_width=1000.0,
            canvas_height=500.0,
        )
    )


def test_rotate_3d_falls_back_to_normalized_view_coordinates(monkeypatch) -> None:
    service = ViewerService()
    view = ViewRecord(view_id="volume-view", series_id="series-a", view_type="3D", width=200, height=100)
    captured: dict[str, tuple[float, float, float]] = {}

    def fake_apply_direct_model_trackball_control_points_to_quaternion(
        quaternion,
        *,
        origin_control_point: tuple[float, float, float],
        current_control_point: tuple[float, float, float],
    ):
        del quaternion
        captured.update(
            origin_control_point=origin_control_point,
            current_control_point=current_control_point,
        )
        return (0.0, 0.0, 0.0, 1.0)

    monkeypatch.setattr(
        "app.services.viewer_service.apply_direct_model_trackball_control_points_to_quaternion",
        fake_apply_direct_model_trackball_control_points_to_quaternion,
    )

    service._handle_drag_rotate_3d(
        view,
        ViewOperationRequest(
            viewId=view.view_id,
            opType=VIEW_OP_TYPE_ROTATE_3D,
            actionType="start",
            x=0.2,
            y=0.3,
        ),
    )
    service._handle_drag_rotate_3d(
        view,
        ViewOperationRequest(
            viewId=view.view_id,
            opType=VIEW_OP_TYPE_ROTATE_3D,
            actionType="move",
            x=0.25,
            y=0.2,
        ),
    )

    assert captured["origin_control_point"] == pytest.approx(
        resolve_direct_model_trackball_control_point(
            canvas_x=40.0,
            canvas_y=30.0,
            canvas_width=200.0,
            canvas_height=100.0,
        )
    )
    assert captured["current_control_point"] == pytest.approx(
        resolve_direct_model_trackball_control_point(
            canvas_x=50.0,
            canvas_y=20.0,
            canvas_width=200.0,
            canvas_height=100.0,
        )
    )


def test_rotate_3d_uses_drag_origin_quaternion_so_coalesced_moves_match() -> None:
    service = ViewerService()
    origin_quaternion = normalize_quaternion((0.13, -0.22, 0.08, 0.96))

    coalesced = ViewRecord(view_id="coalesced", series_id="series-a", view_type="3D", width=1000, height=800)
    coalesced.rotation_quaternion = origin_quaternion
    service._handle_drag_rotate_3d(
        coalesced,
        ViewOperationRequest(
            viewId=coalesced.view_id,
            opType=VIEW_OP_TYPE_ROTATE_3D,
            actionType="start",
            canvasX=100.0,
            canvasY=120.0,
            canvasWidth=1000.0,
            canvasHeight=800.0,
        ),
    )
    service._handle_drag_rotate_3d(
        coalesced,
        ViewOperationRequest(
            viewId=coalesced.view_id,
            opType=VIEW_OP_TYPE_ROTATE_3D,
            actionType="move",
            canvasX=190.0,
            canvasY=150.0,
            canvasWidth=1000.0,
            canvasHeight=800.0,
        ),
    )

    stepped = ViewRecord(view_id="stepped", series_id="series-a", view_type="3D", width=1000, height=800)
    stepped.rotation_quaternion = origin_quaternion
    service._handle_drag_rotate_3d(
        stepped,
        ViewOperationRequest(
            viewId=stepped.view_id,
            opType=VIEW_OP_TYPE_ROTATE_3D,
            actionType="start",
            canvasX=100.0,
            canvasY=120.0,
            canvasWidth=1000.0,
            canvasHeight=800.0,
        ),
    )
    service._handle_drag_rotate_3d(
        stepped,
        ViewOperationRequest(
            viewId=stepped.view_id,
            opType=VIEW_OP_TYPE_ROTATE_3D,
            actionType="move",
            canvasX=140.0,
            canvasY=135.0,
            canvasWidth=1000.0,
            canvasHeight=800.0,
        ),
    )
    service._handle_drag_rotate_3d(
        stepped,
        ViewOperationRequest(
            viewId=stepped.view_id,
            opType=VIEW_OP_TYPE_ROTATE_3D,
            actionType="move",
            canvasX=190.0,
            canvasY=150.0,
            canvasWidth=1000.0,
            canvasHeight=800.0,
        ),
    )

    assert stepped.rotation_quaternion == pytest.approx(coalesced.rotation_quaternion)


def test_rotate_3d_end_applies_release_coordinate_then_clears_origin() -> None:
    service = ViewerService()
    view = ViewRecord(view_id="volume-view", series_id="series-a", view_type="3D", width=1000, height=800)

    service._handle_drag_rotate_3d(
        view,
        ViewOperationRequest(
            viewId=view.view_id,
            opType=VIEW_OP_TYPE_ROTATE_3D,
            actionType="start",
            canvasX=100.0,
            canvasY=120.0,
            canvasWidth=1000.0,
            canvasHeight=800.0,
        ),
    )
    service._handle_drag_rotate_3d(
        view,
        ViewOperationRequest(
            viewId=view.view_id,
            opType=VIEW_OP_TYPE_ROTATE_3D,
            actionType="end",
            canvasX=180.0,
            canvasY=90.0,
            canvasWidth=1000.0,
            canvasHeight=800.0,
        ),
    )

    origin_control = resolve_direct_model_trackball_control_point(
        canvas_x=100.0,
        canvas_y=120.0,
        canvas_width=1000.0,
        canvas_height=800.0,
    )
    current_control = resolve_direct_model_trackball_control_point(
        canvas_x=180.0,
        canvas_y=90.0,
        canvas_width=1000.0,
        canvas_height=800.0,
    )
    expected = apply_direct_model_trackball_control_points_to_quaternion(
        (0.0, 0.0, 0.0, 1.0),
        origin_control_point=origin_control,
        current_control_point=current_control,
    )
    assert view.rotation_quaternion == pytest.approx(expected)
    assert view.drag_origin_arcball_x is None
    assert view.drag_origin_arcball_y is None
    assert view.drag_origin_arcball_z is None
    assert view.drag_origin_rotation_quaternion is None
