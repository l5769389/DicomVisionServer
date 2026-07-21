from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
from dataclasses import dataclass, replace
from math import radians, tan
from threading import RLock
from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtkmodules.vtkCommonCore import vtkObject
from vtkmodules.vtkCommonDataModel import vtkImageData, vtkPolyData
from vtkmodules.vtkFiltersCore import (
    vtkDecimatePro,
    vtkFlyingEdges3D,
    vtkPolyDataNormals,
    vtkWindowedSincPolyDataFilter,
)
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderer,
    vtkRenderWindow,
    vtkWindowToImageFilter,
)

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.surface_render_config import normalize_surface_render_config
from app.services.volume_rendering.contracts import SurfaceRenderRequest, VtkRenderTimings
from app.services.volume_rendering.volume_dtype import prepare_vtk_volume
from app.services.volume_rendering.camera_math import (
    VTK_TRACKBALL_AZIMUTH_DEGREES_PER_VIEW_WIDTH,
    VTK_TRACKBALL_ELEVATION_DEGREES_PER_VIEW_HEIGHT,
    VTK_TRACKBALL_MOTION_FACTOR,
    normalize_quaternion,
    normalize_vector,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
)
from app.services.volume_rendering.camera_fit import (
    BASE_CAMERA_FORWARD,
    BASE_CAMERA_UP,
    bounds_center,
    fit_stable_distance_for_bounds,
    fit_stable_parallel_scale_for_bounds,
    normalize_bounds,
)
from app.services.volume_rendering.vtk_threading import should_bypass_vtk_worker_thread


vtkObject.GlobalWarningDisplayOff()
logger = get_logger(__name__)
BACKGROUND_RGB = (0.0, 0.0, 0.0)
TRACKBALL_MOTION_FACTOR = VTK_TRACKBALL_MOTION_FACTOR
TRACKBALL_AZIMUTH_DEGREES_PER_VIEW_WIDTH = VTK_TRACKBALL_AZIMUTH_DEGREES_PER_VIEW_WIDTH
TRACKBALL_ELEVATION_DEGREES_PER_VIEW_HEIGHT = VTK_TRACKBALL_ELEVATION_DEGREES_PER_VIEW_HEIGHT
FAST_PREVIEW_RENDER_SCALE = 0.5
FAST_PREVIEW_RENDER_MAX_DIMENSION = 720
FINAL_RENDER_SCALE = 1.25
FINAL_RENDER_MAX_DIMENSION = 1600
SURFACE_SESSION_LIMIT = 8
SURFACE_MESH_CACHE_LIMIT = 8


@dataclass
class SurfaceRenderSession:
    image_data: vtkImageData
    mapper: vtkPolyDataMapper
    actor: vtkActor
    renderer: vtkRenderer
    render_window: vtkRenderWindow
    window_to_image: vtkWindowToImageFilter
    volume_token: tuple[object, ...]
    config_token: tuple[object, ...]
    geometry_token: tuple[object, ...]
    canvas_size: tuple[int, int]
    base_position: tuple[float, float, float]
    base_focal_point: tuple[float, float, float]
    base_view_up: tuple[float, float, float]
    base_view_angle: float
    base_parallel_scale: float


class VtkSurfaceRenderer:
    def __init__(self, *, use_process: bool = False) -> None:
        self._sessions: OrderedDict[tuple[str, str], SurfaceRenderSession] = OrderedDict()
        self._mesh_cache: OrderedDict[tuple[object, ...], vtkPolyData] = OrderedDict()
        self._warm_preview_keys: set[tuple[object, ...]] = set()
        self._last_timings: dict[str, VtkRenderTimings] = {}
        self._use_process = bool(use_process)
        self._lock = RLock()
        self._executor = None if should_bypass_vtk_worker_thread() else ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="vtk-surface-render",
        )

    def render(self, request: SurfaceRenderRequest) -> Image.Image:
        if self._use_process:
            from app.services.volume_rendering.gpu_render_process import get_gpu_render_process_client

            image, timings = get_gpu_render_process_client().render_surface(request)
            self._last_timings[request.view_id] = timings
            return image
        if self._executor is None:
            return self._render_in_executor(request)
        return self._executor.submit(self._render_in_executor, request).result()

    def get_last_timings(self, view_id: str) -> VtkRenderTimings:
        return self._last_timings.get(view_id, VtkRenderTimings())

    def warm_preview_session(self, request: SurfaceRenderRequest) -> None:
        if self._use_process:
            return
        if self._executor is None:
            return
        if request.fast_preview or request.canvas_width <= 0 or request.canvas_height <= 0 or not request.view_id:
            return

        warm_key = self._build_warm_preview_key(request)
        with self._lock:
            if warm_key in self._warm_preview_keys:
                return
            self._warm_preview_keys.add(warm_key)

        future = self._executor.submit(self._warm_preview_session_in_executor, replace(request, fast_preview=True))
        future.add_done_callback(lambda done_future, key=warm_key: self._consume_warm_preview_result(key, done_future))

    def apply_trackball_camera_delta(
        self,
        request: SurfaceRenderRequest,
        *,
        delta_x_pixels: float,
        delta_y_pixels: float,
    ) -> tuple[float, float, float, float]:
        if self._use_process:
            from app.services.volume_rendering.gpu_render_process import get_gpu_render_process_client

            return get_gpu_render_process_client().apply_surface_trackball(
                request,
                delta_x_pixels,
                delta_y_pixels,
            )
        if self._executor is None:
            return self._apply_trackball_camera_delta_in_executor(
                request,
                delta_x_pixels,
                delta_y_pixels,
            )
        return self._executor.submit(
            self._apply_trackball_camera_delta_in_executor,
            request,
            delta_x_pixels,
            delta_y_pixels,
        ).result()

    def drop_session(self, view_id: str) -> None:
        if not view_id:
            return
        self._last_timings.pop(view_id, None)
        if self._use_process:
            from app.services.volume_rendering.gpu_render_process import get_gpu_render_process_client

            get_gpu_render_process_client().drop_session(view_id)
            return
        if self._executor is None:
            self._drop_session_in_executor(view_id)
            return
        self._executor.submit(self._drop_session_in_executor, view_id).result()

    def _render_in_executor(self, request: SurfaceRenderRequest) -> Image.Image:
        if request.canvas_width <= 0 or request.canvas_height <= 0:
            raise ValueError("canvas size must be positive")
        if not request.view_id:
            raise ValueError("view_id is required")

        volume = np.asarray(request.volume)
        if volume.ndim != 3 or volume.size == 0:
            raise ValueError("volume must be a non-empty 3D array")

        with self._lock:
            session_started_at = perf_counter()
            session = self._get_or_create_session(request, volume)
            session_ms = (perf_counter() - session_started_at) * 1000.0
            configure_started_at = perf_counter()
            self._configure_session(session, request)
            configure_ms = (perf_counter() - configure_started_at) * 1000.0
            render_started_at = perf_counter()
            session.render_window.Render()
            render_ms = (perf_counter() - render_started_at) * 1000.0
            capture_started_at = perf_counter()
            image = self._capture_image(session)
            capture_ms = (perf_counter() - capture_started_at) * 1000.0
            self._last_timings[request.view_id] = VtkRenderTimings(
                vtk_render_ms=render_ms,
                gpu_readback_ms=capture_ms,
                session_ms=session_ms,
                configure_ms=configure_ms,
                source_dtype=str(volume.dtype),
                vtk_dtype=str(session.image_data.GetScalarTypeAsString()),
            )
            return image

    def _warm_preview_session_in_executor(self, request: SurfaceRenderRequest) -> None:
        volume = np.asarray(request.volume)
        if volume.ndim != 3 or volume.size == 0:
            return

        with self._lock:
            session = self._get_or_create_session(request, volume)
            self._configure_session(session, request)

    def _consume_warm_preview_result(self, warm_key: tuple[object, ...], future) -> None:
        with self._lock:
            self._warm_preview_keys.discard(warm_key)
        try:
            future.result()
        except Exception:
            logger.debug("failed to warm surface preview session", exc_info=True)

    def _apply_trackball_camera_delta_in_executor(
        self,
        request: SurfaceRenderRequest,
        delta_x_pixels: float,
        delta_y_pixels: float,
    ) -> tuple[float, float, float, float]:
        if request.canvas_width <= 0 or request.canvas_height <= 0:
            raise ValueError("canvas size must be positive")
        volume = np.asarray(request.volume)
        with self._lock:
            session = self._get_or_create_session(request, volume)
            self._configure_session(session, request)
            camera = session.renderer.GetActiveCamera()
            delta_azimuth = TRACKBALL_AZIMUTH_DEGREES_PER_VIEW_WIDTH / max(float(request.canvas_width), 1.0)
            delta_elevation = TRACKBALL_ELEVATION_DEGREES_PER_VIEW_HEIGHT / max(float(request.canvas_height), 1.0)
            camera.Azimuth(float(delta_x_pixels) * delta_azimuth * TRACKBALL_MOTION_FACTOR)
            camera.Elevation(float(delta_y_pixels) * delta_elevation * TRACKBALL_MOTION_FACTOR)
            camera.OrthogonalizeViewUp()
            session.renderer.ResetCameraClippingRange()
            return self._camera_to_quaternion(session, camera)

    def _drop_session_in_executor(self, view_id: str) -> None:
        with self._lock:
            expired_keys = [key for key in self._sessions if key[0] == view_id]
            for key in expired_keys:
                session = self._sessions.pop(key, None)
                if session is None:
                    continue
                session.render_window.Finalize()
            self._warm_preview_keys = {key for key in self._warm_preview_keys if key[0] != view_id}

    def _get_or_create_session(self, request: SurfaceRenderRequest, volume: np.ndarray) -> SurfaceRenderSession:
        volume_token = self._build_volume_token(volume, request.spacing_xyz)
        if request.volume_token:
            volume_token = (request.volume_token, volume_token[1], volume_token[2], volume_token[3])
        config = normalize_surface_render_config(request.surface_config)
        config_token = self._build_config_token(config)
        geometry_token = self._build_geometry_token(config)
        session_key = self._build_session_key(request.view_id, request.fast_preview)
        session = self._sessions.get(session_key)
        if session is None or session.volume_token != volume_token:
            if session is not None:
                session.render_window.Finalize()
            self._report_progress(request, 84, "正在提取 Surface 等值面...")
            session = self._create_session(
                prepare_vtk_volume(volume),
                request.spacing_xyz,
                volume_token,
                config,
                config_token,
            )
            self._sessions[session_key] = session
            self._evict_sessions_if_needed()
            self._report_progress(request, 92, "Surface 表面优化完成")
            return session

        if session.config_token != config_token:
            if session.geometry_token != geometry_token:
                mesh_key = self._build_mesh_cache_key(volume_token, geometry_token)
                cache_hit = mesh_key in self._mesh_cache
                self._report_progress(
                    request,
                    88 if cache_hit else 84,
                    "正在复用 Surface 网格..." if cache_hit else "正在提取 Surface 等值面...",
                )
                mesh = self._get_or_create_mesh(session.image_data, volume_token, config)
                session.mapper.SetInputData(mesh)
                session.mapper.Update()
                session.geometry_token = geometry_token
                self._report_progress(request, 92, "Surface 表面优化完成")
            else:
                self._report_progress(request, 90, "正在更新 Surface 材质...")
            self._apply_material(session.actor, config)
            session.config_token = config_token

        self._sessions.move_to_end(session_key)
        return session

    def _evict_sessions_if_needed(self) -> None:
        while len(self._sessions) > SURFACE_SESSION_LIMIT:
            _, session = self._sessions.popitem(last=False)
            session.render_window.Finalize()

    def _create_session(
        self,
        volume: np.ndarray,
        spacing_xyz: tuple[float, float, float],
        volume_token: tuple[object, ...],
        config: dict[str, object],
        config_token: tuple[object, ...],
    ) -> SurfaceRenderSession:
        surface_volume, surface_spacing_xyz = self._prepare_surface_volume(volume, spacing_xyz)
        image_data = self._build_image_data(surface_volume, surface_spacing_xyz)
        geometry_token = self._build_geometry_token(config)
        mesh = self._get_or_create_mesh(image_data, volume_token, config)
        mapper = self._build_surface_mapper(mesh)
        actor = vtkActor()
        actor.SetMapper(mapper)
        self._apply_material(actor, config)

        renderer = vtkRenderer()
        renderer.SetBackground(*BACKGROUND_RGB)
        renderer.AddActor(actor)
        if hasattr(renderer, "UseFXAAOn"):
            renderer.UseFXAAOn()

        render_window = vtkRenderWindow()
        render_window.SetOffScreenRendering(1)
        render_window.SetMultiSamples(0)
        render_window.AddRenderer(renderer)
        render_window.SetSize(1, 1)

        renderer.ResetCamera()
        camera = renderer.GetActiveCamera()
        self._set_base_camera_orientation(camera)
        renderer.ResetCameraClippingRange()

        window_to_image = vtkWindowToImageFilter()
        window_to_image.SetInput(render_window)
        window_to_image.SetInputBufferTypeToRGB()
        window_to_image.ReadFrontBufferOff()

        return SurfaceRenderSession(
            image_data=image_data,
            mapper=mapper,
            actor=actor,
            renderer=renderer,
            render_window=render_window,
            window_to_image=window_to_image,
            volume_token=volume_token,
            config_token=config_token,
            geometry_token=geometry_token,
            canvas_size=(0, 0),
            base_position=tuple(float(value) for value in camera.GetPosition()),
            base_focal_point=tuple(float(value) for value in camera.GetFocalPoint()),
            base_view_up=tuple(float(value) for value in camera.GetViewUp()),
            base_view_angle=float(camera.GetViewAngle()),
            base_parallel_scale=float(camera.GetParallelScale()),
        )

    @staticmethod
    def _build_surface_mesh(
        image_data: vtkImageData,
        config: dict[str, object],
    ) -> vtkPolyData:
        contour = vtkFlyingEdges3D()
        contour.SetInputData(image_data)
        contour.SetValue(0, float(config.get("isoValue", 300.0)))
        if hasattr(contour, "ComputeScalarsOff"):
            contour.ComputeScalarsOff()
        if hasattr(contour, "ComputeGradientsOff"):
            contour.ComputeGradientsOff()
        if hasattr(contour, "ComputeNormalsOff"):
            contour.ComputeNormalsOff()

        source_port = contour.GetOutputPort()
        smoothing = max(0.0, min(1.0, float(config.get("smoothing", 0.0))))
        if smoothing > 0.0:
            smoother = vtkWindowedSincPolyDataFilter()
            smoother.SetInputConnection(source_port)
            smoother.SetNumberOfIterations(max(1, int(6 + smoothing * 24)))
            smoother.BoundarySmoothingOff()
            smoother.FeatureEdgeSmoothingOff()
            smoother.SetFeatureAngle(80.0)
            smoother.SetPassBand(max(0.01, 0.18 - smoothing * 0.14))
            smoother.NonManifoldSmoothingOn()
            smoother.NormalizeCoordinatesOn()
            source_port = smoother.GetOutputPort()

        target_reduction = max(0.0, min(0.9, float(config.get("decimation", 0.0))))
        if target_reduction > 0.0:
            decimate = vtkDecimatePro()
            decimate.SetInputConnection(source_port)
            decimate.PreserveTopologyOn()
            decimate.SetTargetReduction(target_reduction)
            source_port = decimate.GetOutputPort()

        normals = vtkPolyDataNormals()
        normals.SetInputConnection(source_port)
        normals.ConsistencyOn()
        normals.AutoOrientNormalsOn()
        normals.SplittingOff()
        normals.SetFeatureAngle(60.0)

        normals.Update()
        mesh = vtkPolyData()
        mesh.DeepCopy(normals.GetOutput())
        return mesh

    @staticmethod
    def _build_surface_mapper(mesh: vtkPolyData) -> vtkPolyDataMapper:
        mapper = vtkPolyDataMapper()
        mapper.SetInputData(mesh)
        mapper.ScalarVisibilityOff()
        mapper.Update()
        return mapper

    def _get_or_create_mesh(
        self,
        image_data: vtkImageData,
        volume_token: tuple[object, ...],
        config: dict[str, object],
    ) -> vtkPolyData:
        geometry_token = self._build_geometry_token(config)
        cache_key = self._build_mesh_cache_key(volume_token, geometry_token)
        cached = self._mesh_cache.get(cache_key)
        if cached is not None:
            self._mesh_cache.move_to_end(cache_key)
            return cached

        mesh = self._build_surface_mesh(image_data, config)
        self._mesh_cache[cache_key] = mesh
        self._mesh_cache.move_to_end(cache_key)
        while len(self._mesh_cache) > SURFACE_MESH_CACHE_LIMIT:
            self._mesh_cache.popitem(last=False)
        return mesh

    @staticmethod
    def _apply_material(actor: vtkActor, config: dict[str, object]) -> None:
        prop = actor.GetProperty()
        prop.SetColor(*VtkSurfaceRenderer._hex_to_rgb(str(config.get("color") or "#f0eadc"), (0.94, 0.92, 0.86)))
        if hasattr(prop, "SetInterpolationToPhong"):
            prop.SetInterpolationToPhong()
        prop.SetAmbient(max(0.0, min(1.0, float(config.get("ambient", 0.18)))))
        prop.SetDiffuse(max(0.0, min(1.0, float(config.get("diffuse", 0.78)))))
        prop.SetSpecular(max(0.0, min(1.0, float(config.get("specular", 0.28)))))
        roughness = max(0.0, min(1.0, float(config.get("roughness", 0.42))))
        prop.SetSpecularPower(max(1.0, 1.0 + (1.0 - roughness) * 79.0))

    def _configure_session(self, session: SurfaceRenderSession, request: SurfaceRenderRequest) -> None:
        render_size = self._resolve_render_size(request)
        if session.canvas_size != render_size:
            session.render_window.SetSize(*render_size)
            session.canvas_size = render_size
        self._update_camera(session, request)

    @staticmethod
    def _build_session_key(view_id: str, fast_preview: bool) -> tuple[str, str]:
        del fast_preview
        return (view_id, "shared")

    @classmethod
    def _build_warm_preview_key(cls, request: SurfaceRenderRequest) -> tuple[object, ...]:
        volume = np.asarray(request.volume)
        config = normalize_surface_render_config(request.surface_config)
        preview_request = replace(request, fast_preview=True)
        volume_token = cls._build_volume_token(volume, request.spacing_xyz)
        if request.volume_token:
            volume_token = (request.volume_token, volume_token[1], volume_token[2], volume_token[3])
        return (
            request.view_id,
            volume_token,
            cls._build_config_token(config),
            cls._resolve_render_size(preview_request),
        )

    @staticmethod
    def _build_volume_token(
        volume: np.ndarray,
        spacing_xyz: tuple[float, float, float],
    ) -> tuple[object, tuple[int, ...], tuple[float, float, float], str]:
        return (
            id(volume),
            tuple(int(size) for size in volume.shape),
            tuple(float(value) for value in spacing_xyz),
            str(volume.dtype),
        )

    @staticmethod
    def _build_config_token(config: dict[str, object]) -> tuple[object, ...]:
        return (
            str(config.get("preset", "bone")),
            round(float(config.get("isoValue", 300.0)), 3),
            round(float(config.get("smoothing", 0.0)), 3),
            round(float(config.get("decimation", 0.0)), 3),
            str(config.get("color", "#f0eadc")),
            round(float(config.get("ambient", 0.18)), 3),
            round(float(config.get("diffuse", 0.78)), 3),
            round(float(config.get("specular", 0.28)), 3),
            round(float(config.get("roughness", 0.42)), 3),
        )

    @staticmethod
    def _build_geometry_token(config: dict[str, object]) -> tuple[object, ...]:
        return (
            round(float(config.get("isoValue", 300.0)), 3),
            round(float(config.get("smoothing", 0.0)), 3),
            round(float(config.get("decimation", 0.0)), 3),
        )

    @staticmethod
    def _build_mesh_cache_key(
        volume_token: tuple[object, tuple[int, ...], tuple[float, float, float]],
        geometry_token: tuple[object, ...],
    ) -> tuple[object, ...]:
        return (*volume_token, *geometry_token)

    @staticmethod
    def _report_progress(request: SurfaceRenderRequest, progress_percent: int, message: str) -> None:
        callback = request.progress_callback
        if callback is None or request.fast_preview:
            return
        try:
            callback({
                "phase": "preprocess",
                "progressPercent": progress_percent,
                "message": message,
            })
        except Exception:
            logger.debug("surface progress callback failed view_id=%s", request.view_id, exc_info=True)

    @staticmethod
    def _prepare_surface_volume(
        volume: np.ndarray,
        spacing_xyz: tuple[float, float, float],
    ) -> tuple[np.ndarray, tuple[float, float, float]]:
        return volume, spacing_xyz

    @staticmethod
    def _resolve_render_size(request: SurfaceRenderRequest) -> tuple[int, int]:
        width = max(1, int(request.canvas_width))
        height = max(1, int(request.canvas_height))
        if not request.fast_preview:
            scale = min(
                FINAL_RENDER_SCALE,
                FINAL_RENDER_MAX_DIMENSION / float(max(width, height)),
            )
            scale = max(1.0, scale)
            return (
                max(1, int(round(width * scale))),
                max(1, int(round(height * scale))),
            )

        scale = FAST_PREVIEW_RENDER_SCALE
        largest = max(width, height)
        if largest > FAST_PREVIEW_RENDER_MAX_DIMENSION:
            scale = min(scale, FAST_PREVIEW_RENDER_MAX_DIMENSION / float(largest))
        return (
            max(96, int(round(width * scale))),
            max(96, int(round(height * scale))),
        )

    @staticmethod
    def _build_image_data(volume: np.ndarray, spacing_xyz: tuple[float, float, float]) -> vtkImageData:
        depth, height, width = volume.shape
        image_data = vtkImageData()
        image_data.SetDimensions(width, height, depth)
        image_data.SetSpacing(*spacing_xyz)
        image_data.SetOrigin(0.0, 0.0, 0.0)

        vtk_array = numpy_to_vtk(
            num_array=np.ascontiguousarray(volume).ravel(order="C"),
            deep=True,
        )
        vtk_array.SetName("Scalars")
        image_data.GetPointData().SetScalars(vtk_array)
        return image_data

    @staticmethod
    def _set_base_camera_orientation(camera) -> None:
        focal_point = np.array(camera.GetFocalPoint(), dtype=np.float64)
        distance = max(float(camera.GetDistance()), 1e-3)
        position = focal_point - BASE_CAMERA_FORWARD * distance
        camera.SetPosition(*position.tolist())
        camera.SetFocalPoint(*focal_point.tolist())
        camera.SetViewUp(*BASE_CAMERA_UP.tolist())
        camera.OrthogonalizeViewUp()

    @staticmethod
    def _resolve_session_bounds(session: SurfaceRenderSession) -> tuple[float, float, float, float, float, float] | None:
        return normalize_bounds(session.actor.GetBounds()) or normalize_bounds(session.image_data.GetBounds())

    def _refresh_base_camera_frame(self, session: SurfaceRenderSession, request: SurfaceRenderRequest) -> None:
        bounds = self._resolve_session_bounds(session)
        if bounds is None:
            return

        aspect_ratio = max(float(request.canvas_width), 1.0) / max(float(request.canvas_height), 1.0)
        focal_point = bounds_center(bounds)
        view_angle = max(float(session.base_view_angle), 1.0)
        distance = fit_stable_distance_for_bounds(
            bounds,
            view_angle_degrees=view_angle,
            aspect_ratio=aspect_ratio,
        )
        position = focal_point - BASE_CAMERA_FORWARD * distance
        session.base_position = tuple(float(value) for value in position)
        session.base_focal_point = tuple(float(value) for value in focal_point)
        session.base_view_up = tuple(float(value) for value in BASE_CAMERA_UP)
        session.base_parallel_scale = fit_stable_parallel_scale_for_bounds(
            bounds,
            aspect_ratio=aspect_ratio,
        )

    def _update_camera(self, session: SurfaceRenderSession, request: SurfaceRenderRequest) -> None:
        self._refresh_base_camera_frame(session, request)
        camera = session.renderer.GetActiveCamera()
        model_rotation_matrix = self._quaternion_to_rotation_matrix(request.rotation_quaternion)
        camera_rotation_matrix = model_rotation_matrix.T
        base_position = np.array(session.base_position, dtype=np.float64)
        base_focal_point = np.array(session.base_focal_point, dtype=np.float64)
        base_view_up = np.array(session.base_view_up, dtype=np.float64)
        relative_position = base_position - base_focal_point
        rotated_position = base_focal_point + camera_rotation_matrix @ relative_position
        rotated_view_up = camera_rotation_matrix @ base_view_up

        clamped_zoom = min(max(0.65, float(request.zoom)), 2.35)
        camera.SetPosition(*rotated_position.tolist())
        camera.SetFocalPoint(*base_focal_point.tolist())
        camera.SetViewUp(*rotated_view_up.tolist())
        camera.SetViewAngle(float(session.base_view_angle))
        camera.SetParallelScale(float(session.base_parallel_scale))
        camera.OrthogonalizeViewUp()
        camera.Zoom(clamped_zoom)
        self._apply_pan(camera, session.renderer, request)
        session.renderer.ResetCameraClippingRange()

    def _camera_to_quaternion(
        self,
        session: SurfaceRenderSession,
        camera,
    ) -> tuple[float, float, float, float]:
        base_forward = self._normalize_vector(np.array(session.base_focal_point, dtype=np.float64) - np.array(session.base_position, dtype=np.float64))
        base_up = self._normalize_vector(np.array(session.base_view_up, dtype=np.float64))
        base_right = self._normalize_vector(np.cross(base_forward, base_up))
        current_forward = self._normalize_vector(np.array(camera.GetFocalPoint(), dtype=np.float64) - np.array(camera.GetPosition(), dtype=np.float64))
        current_up = self._normalize_vector(np.array(camera.GetViewUp(), dtype=np.float64))
        current_right = self._normalize_vector(np.cross(current_forward, current_up))

        current_up = self._normalize_vector(np.cross(current_right, current_forward))

        base_basis = np.column_stack((base_right, base_forward, base_up))
        current_basis = np.column_stack((current_right, current_forward, current_up))
        camera_rotation_matrix = current_basis @ base_basis.T
        return self._rotation_matrix_to_quaternion(camera_rotation_matrix.T)

    @staticmethod
    def _normalize_quaternion(quaternion: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        return normalize_quaternion(quaternion)

    @staticmethod
    def _normalize_vector(vector: np.ndarray) -> np.ndarray:
        return normalize_vector(vector)

    @staticmethod
    def _hex_to_rgb(color: str, fallback: tuple[float, float, float]) -> tuple[float, float, float]:
        text = str(color or "").strip()
        if len(text) == 7 and text.startswith("#"):
            try:
                return tuple(int(text[index:index + 2], 16) / 255.0 for index in (1, 3, 5))
            except ValueError:
                return fallback
        return fallback

    @staticmethod
    def _rotation_matrix_to_quaternion(matrix: np.ndarray) -> tuple[float, float, float, float]:
        return rotation_matrix_to_quaternion(matrix)

    @staticmethod
    def _quaternion_to_rotation_matrix(quaternion: tuple[float, float, float, float]) -> np.ndarray:
        return quaternion_to_rotation_matrix(quaternion)

    @staticmethod
    def _apply_pan(camera, renderer: vtkRenderer, request: SurfaceRenderRequest) -> None:
        if abs(request.offset_x) < 1e-3 and abs(request.offset_y) < 1e-3:
            return

        renderer.ResetCameraClippingRange()
        direction = np.array(camera.GetDirectionOfProjection(), dtype=np.float64)
        direction_norm = np.linalg.norm(direction)
        if direction_norm <= 1e-6:
            return
        direction = direction / direction_norm

        up = np.array(camera.GetViewUp(), dtype=np.float64)
        up_norm = np.linalg.norm(up)
        if up_norm <= 1e-6:
            return
        up = up / up_norm

        right = np.cross(direction, up)
        right_norm = np.linalg.norm(right)
        if right_norm <= 1e-6:
            return
        right = right / right_norm

        distance = max(float(camera.GetDistance()), 1e-3)
        visible_height = 2.0 * distance * tan(radians(max(float(camera.GetViewAngle()), 1.0)) / 2.0)
        visible_width = visible_height * (float(request.canvas_width) / max(float(request.canvas_height), 1.0))
        delta = right * (-visible_width * (float(request.offset_x) / max(float(request.canvas_width), 1.0)))
        delta += up * (visible_height * (float(request.offset_y) / max(float(request.canvas_height), 1.0)))

        focal_point = np.array(camera.GetFocalPoint(), dtype=np.float64) + delta
        position = np.array(camera.GetPosition(), dtype=np.float64) + delta
        camera.SetFocalPoint(*focal_point.tolist())
        camera.SetPosition(*position.tolist())

    @staticmethod
    def _capture_image(session: SurfaceRenderSession) -> Image.Image:
        session.window_to_image.Modified()
        session.window_to_image.Update()

        image_data = session.window_to_image.GetOutput()
        width, height, _ = image_data.GetDimensions()
        vtk_scalars = image_data.GetPointData().GetScalars()
        array = vtk_to_numpy(vtk_scalars).reshape(height, width, 3)
        array = np.flipud(array)
        return Image.fromarray(array)


vtk_surface_renderer = VtkSurfaceRenderer(use_process=get_settings().vtk_render_process_enabled)
