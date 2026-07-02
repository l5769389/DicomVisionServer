from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from math import ceil, radians, tan
from threading import RLock
from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtkmodules.util.vtkConstants import VTK_FLOAT
from vtkmodules.vtkCommonCore import vtkObject
from vtkmodules.vtkCommonDataModel import vtkImageData, vtkPiecewiseFunction
from vtkmodules.vtkRenderingCore import (
    vtkColorTransferFunction,
    vtkRenderer,
    vtkRenderWindow,
    vtkVolume,
    vtkVolumeProperty,
    vtkWindowToImageFilter,
)
from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkSmartVolumeMapper

from app.core.logging import get_logger
from app.services.volume_rendering.contracts import VolumeRenderRequest
from app.services.volume_rendering.camera_math import (
    normalize_quaternion,
    normalize_vector,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
)


vtkObject.GlobalWarningDisplayOff()
logger = get_logger(__name__)
BACKGROUND_RGB = (0.0, 0.0, 0.0)
TRACKBALL_MOTION_FACTOR = 36.0
TRACKBALL_AZIMUTH_DEGREES_PER_VIEW_WIDTH = -20.0
TRACKBALL_ELEVATION_DEGREES_PER_VIEW_HEIGHT = 20.0
FAST_PREVIEW_IMAGE_SAMPLE_DISTANCE = 1.45
FINAL_RENDER_IMAGE_SAMPLE_DISTANCE = 1.0
FAST_PREVIEW_RAY_SAMPLE_FACTOR = 1.45
FINAL_RENDER_RAY_SAMPLE_FACTOR = 0.72
FAST_PREVIEW_VOLUME_MAX_DIMENSION = 144
VOLUME_SESSION_LIMIT = 8
VOLUME_PREVIEW_CACHE_LIMIT = 4


@dataclass
class VolumeRenderSession:
    image_data: vtkImageData
    mapper: vtkSmartVolumeMapper
    volume_property: vtkVolumeProperty
    volume_actor: vtkVolume
    renderer: vtkRenderer
    render_window: vtkRenderWindow
    window_to_image: vtkWindowToImageFilter
    color_func: vtkColorTransferFunction
    opacity_func: vtkPiecewiseFunction
    gradient_func: vtkPiecewiseFunction
    volume_token: tuple[object, tuple[int, ...], tuple[float, float, float]]
    canvas_size: tuple[int, int]
    base_position: tuple[float, float, float]
    base_focal_point: tuple[float, float, float]
    base_view_up: tuple[float, float, float]
    base_view_angle: float
    base_parallel_scale: float
    transfer_function_token: tuple[object, ...] | None
    sampling_token: tuple[object, ...] | None


class VtkVolumeRenderer:
    def __init__(self) -> None:
        self._sessions: OrderedDict[tuple[str, str], VolumeRenderSession] = OrderedDict()
        self._preview_volume_cache: OrderedDict[
            tuple[object, tuple[int, ...], tuple[float, float, float]],
            tuple[np.ndarray, tuple[float, float, float]],
        ] = OrderedDict()
        self._lock = RLock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vtk-render")

    def render(self, request: VolumeRenderRequest) -> Image.Image:
        return self._executor.submit(self._render_in_executor, request).result()

    def apply_trackball_camera_delta(
        self,
        request: VolumeRenderRequest,
        *,
        delta_x_pixels: float,
        delta_y_pixels: float,
    ) -> tuple[float, float, float, float]:
        return self._executor.submit(
            self._apply_trackball_camera_delta_in_executor,
            request,
            delta_x_pixels,
            delta_y_pixels,
        ).result()

    def drop_session(self, view_id: str) -> None:
        if not view_id:
            return
        self._executor.submit(self._drop_session_in_executor, view_id).result()

    def _render_in_executor(self, request: VolumeRenderRequest) -> Image.Image:
        total_started_at = perf_counter()
        if request.canvas_width <= 0 or request.canvas_height <= 0:
            raise ValueError("canvas size must be positive")
        if not request.view_id:
            raise ValueError("view_id is required")

        volume_started_at = perf_counter()
        volume = np.asarray(request.volume, dtype=np.float32)
        if volume.ndim != 3 or volume.size == 0:
            raise ValueError("volume must be a non-empty 3D array")
        volume_ms = (perf_counter() - volume_started_at) * 1000.0

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
            if request.fast_preview:
                source_shape = tuple(int(value) for value in volume.shape)
                render_shape = tuple(int(value) for value in session.image_data.GetDimensions())
                mapper_mode = self._get_mapper_last_used_render_mode(session.mapper)
                logger.debug(
                    "3d vtk preview timing view_id=%s token=%s session=%s source_shape=%s preview_image_dims=%s render_size=%sx%s mapper_mode=%s volume_ms=%.1f session_ms=%.1f configure_ms=%.1f vtk_render_ms=%.1f capture_ms=%.1f total_ms=%.1f",
                    request.view_id,
                    request.volume_token or "ndarray",
                    self._build_session_key(request.view_id, request.fast_preview)[1],
                    source_shape,
                    render_shape,
                    request.canvas_width,
                    request.canvas_height,
                    mapper_mode,
                    volume_ms,
                    session_ms,
                    configure_ms,
                    render_ms,
                    capture_ms,
                    (perf_counter() - total_started_at) * 1000.0,
                )
            return image

    @staticmethod
    def _get_mapper_last_used_render_mode(mapper: vtkSmartVolumeMapper) -> int | None:
        getter = getattr(mapper, "GetLastUsedRenderMode", None)
        if not callable(getter):
            return None
        try:
            return int(getter())
        except Exception:
            return None

    def _drop_session_in_executor(self, view_id: str) -> None:
        with self._lock:
            expired_keys = [key for key in self._sessions if key[0] == view_id]
            for key in expired_keys:
                session = self._sessions.pop(key, None)
                if session is None:
                    continue
                session.render_window.Finalize()

    def _apply_trackball_camera_delta_in_executor(
        self,
        request: VolumeRenderRequest,
        delta_x_pixels: float,
        delta_y_pixels: float,
    ) -> tuple[float, float, float, float]:
        if request.canvas_width <= 0 or request.canvas_height <= 0:
            raise ValueError("canvas size must be positive")
        volume = np.asarray(request.volume, dtype=np.float32)
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

    def _get_or_create_session(self, request: VolumeRenderRequest, volume: np.ndarray) -> VolumeRenderSession:
        volume_token = self._build_volume_token(volume, request.spacing_xyz, request.volume_token)
        session_key = self._build_session_key(request.view_id, request.fast_preview)
        session = self._sessions.get(session_key)
        if session is None or session.volume_token != volume_token:
            if session is not None:
                session.render_window.Finalize()
            session = self._create_session(volume, request.spacing_xyz, volume_token, request.fast_preview)
            self._sessions[session_key] = session
            self._evict_sessions_if_needed()
            return session
        self._sessions.move_to_end(session_key)
        return session

    def _evict_sessions_if_needed(self) -> None:
        while len(self._sessions) > VOLUME_SESSION_LIMIT:
            _, session = self._sessions.popitem(last=False)
            session.render_window.Finalize()

    def _create_session(
        self,
        volume: np.ndarray,
        spacing_xyz: tuple[float, float, float],
        volume_token: tuple[object, tuple[int, ...], tuple[float, float, float]],
        fast_preview: bool,
    ) -> VolumeRenderSession:
        render_volume, render_spacing_xyz = self._prepare_render_volume(
            volume,
            spacing_xyz,
            volume_token,
            fast_preview,
        )
        image_data = self._build_image_data(render_volume, render_spacing_xyz)

        mapper = vtkSmartVolumeMapper()
        mapper.SetInputData(image_data)
        mapper.SetBlendModeToComposite()
        if hasattr(mapper, "SetRequestedRenderModeToGPU"):
            mapper.SetRequestedRenderModeToGPU()
        if hasattr(mapper, "SetInteractiveAdjustSampleDistances"):
            mapper.SetInteractiveAdjustSampleDistances(0)
        if hasattr(mapper, "SetAutoAdjustSampleDistances"):
            mapper.SetAutoAdjustSampleDistances(0)

        color_func = vtkColorTransferFunction()
        opacity_func = vtkPiecewiseFunction()
        gradient_func = vtkPiecewiseFunction()

        volume_property = vtkVolumeProperty()
        volume_property.SetColor(color_func)
        volume_property.SetScalarOpacity(opacity_func)
        volume_property.SetGradientOpacity(gradient_func)
        volume_property.ShadeOn()
        volume_property.SetInterpolationTypeToLinear()
        volume_property.SetAmbient(0.18)
        volume_property.SetDiffuse(0.82)
        volume_property.SetSpecular(0.12)
        volume_property.SetSpecularPower(10.0)

        volume_actor = vtkVolume()
        volume_actor.SetMapper(mapper)
        volume_actor.SetProperty(volume_property)

        renderer = vtkRenderer()
        renderer.SetBackground(*BACKGROUND_RGB)
        renderer.AddVolume(volume_actor)

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

        return VolumeRenderSession(
            image_data=image_data,
            mapper=mapper,
            volume_property=volume_property,
            volume_actor=volume_actor,
            renderer=renderer,
            render_window=render_window,
            window_to_image=window_to_image,
            color_func=color_func,
            opacity_func=opacity_func,
            gradient_func=gradient_func,
            volume_token=volume_token,
            canvas_size=(0, 0),
            base_position=tuple(float(value) for value in camera.GetPosition()),
            base_focal_point=tuple(float(value) for value in camera.GetFocalPoint()),
            base_view_up=tuple(float(value) for value in camera.GetViewUp()),
            base_view_angle=float(camera.GetViewAngle()),
            base_parallel_scale=float(camera.GetParallelScale()),
            transfer_function_token=None,
            sampling_token=None,
        )

    def _configure_session(self, session: VolumeRenderSession, request: VolumeRenderRequest) -> None:
        if session.canvas_size != (request.canvas_width, request.canvas_height):
            session.render_window.SetSize(request.canvas_width, request.canvas_height)
            session.canvas_size = (request.canvas_width, request.canvas_height)

        transfer_function_token = self._build_transfer_function_token(
            request.window_width,
            request.window_center,
            request.volume_preset,
            request.volume_config,
        )
        if session.transfer_function_token != transfer_function_token:
            self._update_transfer_functions(
                session,
                request.window_width,
                request.window_center,
                request.volume_preset,
                request.volume_config,
                fast_preview=request.fast_preview,
            )
            session.transfer_function_token = transfer_function_token

        sampling_token = self._build_sampling_token(request)
        if session.sampling_token != sampling_token:
            self._update_sampling(session, request)
            session.sampling_token = sampling_token
        self._update_camera(session, request)

    @staticmethod
    def get_default_rotation_quaternion() -> tuple[float, float, float, float]:
        return (0.0, 0.0, 0.0, 1.0)

    @staticmethod
    def _set_base_camera_orientation(camera) -> None:
        focal_point = np.array(camera.GetFocalPoint(), dtype=np.float64)
        distance = max(float(camera.GetDistance()), 1e-3)
        forward = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        position = focal_point - forward * distance
        camera.SetPosition(*position.tolist())
        camera.SetFocalPoint(*focal_point.tolist())
        camera.SetViewUp(*up.tolist())
        camera.OrthogonalizeViewUp()

    @staticmethod
    def _build_volume_token(
        volume: np.ndarray,
        spacing_xyz: tuple[float, float, float],
        volume_token: str | None = None,
    ) -> tuple[object, tuple[int, ...], tuple[float, float, float]]:
        identity: object = str(volume_token) if volume_token else id(volume)
        return (identity, tuple(int(size) for size in volume.shape), tuple(float(value) for value in spacing_xyz))

    @staticmethod
    def _build_session_key(view_id: str, fast_preview: bool) -> tuple[str, str]:
        return (view_id, "preview" if fast_preview else "final")

    def _prepare_render_volume(
        self,
        volume: np.ndarray,
        spacing_xyz: tuple[float, float, float],
        volume_token: tuple[object, tuple[int, ...], tuple[float, float, float]],
        fast_preview: bool,
    ) -> tuple[np.ndarray, tuple[float, float, float]]:
        if not fast_preview:
            return volume, spacing_xyz

        cached = self._preview_volume_cache.get(volume_token)
        if cached is not None:
            self._preview_volume_cache.move_to_end(volume_token)
            return cached

        sampled, sampled_spacing_xyz = self._downsample_preview_volume(volume, spacing_xyz)
        self._preview_volume_cache[volume_token] = (sampled, sampled_spacing_xyz)
        while len(self._preview_volume_cache) > VOLUME_PREVIEW_CACHE_LIMIT:
            self._preview_volume_cache.popitem(last=False)
        return sampled, sampled_spacing_xyz

    @staticmethod
    def _downsample_preview_volume(
        volume: np.ndarray,
        spacing_xyz: tuple[float, float, float],
    ) -> tuple[np.ndarray, tuple[float, float, float]]:
        depth, height, width = volume.shape
        step = max(1, int(ceil(max(depth, height, width) / FAST_PREVIEW_VOLUME_MAX_DIMENSION)))
        if step == 1:
            return volume, spacing_xyz

        sampled = volume[::step, ::step, ::step]
        if min(sampled.shape) < 2:
            return volume, spacing_xyz

        spacing_x, spacing_y, spacing_z = spacing_xyz
        return sampled, (
            float(spacing_x) * float(step),
            float(spacing_y) * float(step),
            float(spacing_z) * float(step),
        )

    @classmethod
    def _build_transfer_function_token(
        cls,
        window_width: float,
        window_center: float,
        volume_preset: str,
        volume_config: dict[str, Any] | None,
    ) -> tuple[object, ...]:
        return (
            cls._token_float(window_width),
            cls._token_float(window_center),
            str(volume_preset or "bone").strip().lower(),
            cls._freeze_token_value(volume_config or {}),
        )

    @staticmethod
    def _build_sampling_token(request: VolumeRenderRequest) -> tuple[object, ...]:
        return (
            bool(request.fast_preview),
            tuple(round(abs(float(value)), 6) for value in request.spacing_xyz),
        )

    @staticmethod
    def _token_float(value: Any) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = 0.0
        return round(numeric, 6)

    @classmethod
    def _freeze_token_value(cls, value: Any) -> object:
        if isinstance(value, dict):
            return tuple(sorted((str(key), cls._freeze_token_value(item)) for key, item in value.items()))
        if isinstance(value, (list, tuple)):
            return tuple(cls._freeze_token_value(item) for item in value)
        if isinstance(value, float):
            return round(value, 6)
        if isinstance(value, (str, int, bool)) or value is None:
            return value
        return str(value)

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
            array_type=VTK_FLOAT,
        )
        vtk_array.SetName("Scalars")
        image_data.GetPointData().SetScalars(vtk_array)
        return image_data

    @staticmethod
    def _update_sampling(session: VolumeRenderSession, request: VolumeRenderRequest) -> None:
        if hasattr(session.mapper, "SetImageSampleDistance"):
            # Drag previews intentionally sample more coarsely. The final frame
            # is queued separately by the operation handler after interaction ends.
            session.mapper.SetImageSampleDistance(
                FAST_PREVIEW_IMAGE_SAMPLE_DISTANCE if request.fast_preview else FINAL_RENDER_IMAGE_SAMPLE_DISTANCE
            )
        if hasattr(session.mapper, "SetSampleDistance"):
            render_spacing_xyz = session.image_data.GetSpacing()
            min_spacing = max(1e-3, min(abs(float(value)) for value in render_spacing_xyz))
            factor = FAST_PREVIEW_RAY_SAMPLE_FACTOR if request.fast_preview else FINAL_RENDER_RAY_SAMPLE_FACTOR
            max_sample_distance = 3.5 if request.fast_preview else 1.25
            sample_distance = max(0.24, min(max_sample_distance, min_spacing * factor))
            session.mapper.SetSampleDistance(sample_distance)

    def _update_transfer_functions(
        self,
        session: VolumeRenderSession,
        window_width: float,
        window_center: float,
        volume_preset: str,
        volume_config: dict[str, Any] | None,
        *,
        fast_preview: bool = False,
    ) -> None:
        config = volume_config or {}
        preset = str(config.get("preset") or volume_preset or "bone").strip().lower()
        blend_mode = str(config.get("blendMode") or "composite").strip().lower()
        raw_layers = config.get("layers") if isinstance(config.get("layers"), list) else []
        layers = [layer for layer in raw_layers if isinstance(layer, dict) and layer.get("enabled")]
        lighting = config.get("lighting") if isinstance(config.get("lighting"), dict) else {}
        shading_enabled = bool(lighting.get("shading", blend_mode != "mip"))
        interpolation = str(lighting.get("interpolation") or "linear").strip().lower()
        ambient = self._clamp_unit_interval(lighting.get("ambient"), 0.16 if blend_mode != "mip" else 1.0)
        diffuse = self._clamp_unit_interval(lighting.get("diffuse"), 0.86 if blend_mode != "mip" else 0.0)
        specular = self._clamp_unit_interval(lighting.get("specular"), 0.18 if blend_mode != "mip" else 0.0)
        roughness = self._clamp_unit_interval(lighting.get("roughness"), 0.78 if blend_mode != "mip" else 1.0)

        if hasattr(session.mapper, "SetBlendModeToComposite"):
            session.mapper.SetBlendModeToComposite()
        if blend_mode == "mip" and hasattr(session.mapper, "SetBlendModeToMaximumIntensity"):
            session.mapper.SetBlendModeToMaximumIntensity()

        if interpolation == "nearest" and hasattr(session.volume_property, "SetInterpolationTypeToNearest"):
            session.volume_property.SetInterpolationTypeToNearest()
        elif interpolation == "cubic" and hasattr(session.volume_property, "SetInterpolationTypeToCubic"):
            session.volume_property.SetInterpolationTypeToCubic()
        else:
            session.volume_property.SetInterpolationTypeToLinear()

        if fast_preview:
            session.volume_property.ShadeOff()
            ambient = 1.0
            diffuse = 0.0
            specular = 0.0
            roughness = 1.0
        elif shading_enabled and layers and blend_mode != "mip":
            session.volume_property.ShadeOn()
        else:
            session.volume_property.ShadeOff()
        session.volume_property.SetAmbient(ambient)
        session.volume_property.SetDiffuse(diffuse)
        session.volume_property.SetSpecular(specular)
        session.volume_property.SetSpecularPower(max(1.0, 1.0 + (1.0 - roughness) * 59.0))

        session.color_func.RemoveAllPoints()
        session.opacity_func.RemoveAllPoints()
        session.gradient_func.RemoveAllPoints()

        combined_color_points, combined_opacity_points = self._build_foundation_transfer_points(
            preset=preset,
            window_width=window_width,
            window_center=window_center,
            overlay_active=bool(layers),
        )

        for layer in sorted(layers, key=lambda item: float(item.get("wl", 0.0))):
            layer_color_points, layer_opacity_points = self._build_layer_transfer_points(layer)
            combined_color_points.extend(layer_color_points)
            combined_opacity_points.extend(layer_opacity_points)

        if not combined_color_points:
            combined_color_points.append((0.0, 0.0, 0.0, 0.0))
        if not combined_opacity_points:
            combined_opacity_points.append((0.0, 0.0))

        for position, red, green, blue in sorted(combined_color_points, key=lambda item: item[0]):
            session.color_func.AddRGBPoint(position, red, green, blue)
        for position, opacity in sorted(combined_opacity_points, key=lambda item: item[0]):
            session.opacity_func.AddPoint(position, opacity)

        for position, opacity in self._build_gradient_opacity_points(preset, blend_mode, bool(layers)):
            session.gradient_func.AddPoint(position, opacity)

    def _build_foundation_transfer_points(
        self,
        *,
        preset: str,
        window_width: float,
        window_center: float,
        overlay_active: bool,
    ) -> tuple[list[tuple[float, float, float, float]], list[tuple[float, float]]]:
        width = max(float(window_width or 400.0), 1.0)
        center = float(window_center or 40.0)
        low = center - width / 2.0
        high = center + width / 2.0
        shoulder = center + width * 0.35
        tail = center + width * 0.75

        color_points = [
            (low - width * 0.45, 0.0, 0.0, 0.0),
            (low, 0.03, 0.03, 0.03),
            (center - width * 0.18, 0.07, 0.07, 0.07),
            (center, 0.22, 0.22, 0.22),
            (high, 0.78, 0.78, 0.78),
            (tail, 0.98, 0.98, 0.98),
        ]

        if preset == 'red':
            color_points.extend([
                (center + width * 0.22, 0.72, 0.26, 0.26),
                (tail, 0.95, 0.52, 0.52),
            ])
        elif preset == 'cardiac':
            color_points.extend([
                (center + width * 0.08, 0.44, 0.24, 0.2),
                (tail, 0.92, 0.86, 0.82),
            ])
        elif preset == 'muscle':
            color_points.extend([
                (center + width * 0.05, 0.30, 0.20, 0.18),
                (tail, 0.72, 0.58, 0.52),
            ])

        if preset == 'mip':
            opacity_points = [
                (low - width * 0.45, 0.0),
                (low, 0.0),
                (center, 0.0),
                (high, 0.002),
                (tail, 0.006),
            ]
            return color_points, opacity_points

        strength_scale = {
            'bone': 0.58,
            'aaa': 0.75,
            'red': 0.92,
            'cardiac': 0.82,
            'muscle': 0.78,
        }.get(preset, 0.8)
        overlay_factor = 0.34 if overlay_active else 1.0
        base_peak = 0.016 * strength_scale * overlay_factor
        shoulder_peak = 0.008 * strength_scale * overlay_factor

        opacity_points = [
            (low - width * 0.45, 0.0),
            (low, 0.0),
            (center - width * 0.12, 0.0002 * strength_scale * overlay_factor),
            (center, 0.0007 * strength_scale * overlay_factor),
            (shoulder, shoulder_peak),
            (tail, base_peak),
        ]
        return color_points, opacity_points

    def _build_layer_transfer_points(
        self,
        layer: dict[str, Any],
    ) -> tuple[list[tuple[float, float, float, float]], list[tuple[float, float]]]:
        key = str(layer.get('key') or '').strip()
        width = float(layer.get('ww', 1.0))
        center = float(layer.get('wl', 0.0))
        opacity = max(0.0, min(1.0, float(layer.get('opacity', 0.0))))
        low = center - width / 2.0
        high = center + width / 2.0
        start_rgb = self._hex_to_rgb(str(layer.get('colorStart') or '#ffffff'), (1.0, 1.0, 1.0))
        end_rgb = self._hex_to_rgb(str(layer.get('colorEnd') or '#ffffff'), start_rgb)
        mid_rgb = tuple((start_rgb[index] + end_rgb[index]) / 2.0 for index in range(3))

        if key == 'bone':
            highlight_rgb = (1.0, 1.0, 1.0)
            rise = center - width * 0.12
            shoulder = center + width * 0.16
            color_points = [
                (low, *start_rgb),
                (rise, *mid_rgb),
                (center, *highlight_rgb),
                (shoulder, *highlight_rgb),
                (high, *highlight_rgb),
            ]
            opacity_points = [
                (low, 0.0),
                (rise, 0.0),
                (center, opacity * 0.34),
                (shoulder, opacity * 0.86),
                (high, opacity),
            ]
            return color_points, opacity_points

        if key == 'blood':
            q1 = center - width * 0.14
            q2 = center + width * 0.18
            q3 = center + width * 0.42
            deep_rgb = (max(0.18, start_rgb[0] * 0.7), start_rgb[1] * 0.16, start_rgb[2] * 0.1)
            hot_rgb = (min(1.0, start_rgb[0] * 1.14), max(start_rgb[1], 0.12), max(start_rgb[2], 0.08))
            warm_rgb = tuple(min(1.0, end_rgb[index] * 0.94 + mid_rgb[index] * 0.06) for index in range(3))
            color_points = [
                (low, *deep_rgb),
                (q1, *start_rgb),
                (center, *start_rgb),
                (q2, *hot_rgb),
                (q3, *warm_rgb),
                (high, *end_rgb),
            ]
            opacity_points = [
                (low, 0.0),
                (q1, min(1.0, opacity * 0.72)),
                (center, min(1.0, opacity * 1.95)),
                (q2, min(1.0, opacity * 1.75)),
                (q3, min(1.0, opacity * 0.9)),
                (high, min(1.0, opacity * 0.22)),
            ]
            return color_points, opacity_points

        if key == 'lung':
            color_points = [
                (low, *start_rgb),
                (center - width * 0.18, *mid_rgb),
                (center, *end_rgb),
                (high, *end_rgb),
            ]
            opacity_points = [
                (low, 0.0),
                (center - width * 0.12, opacity * 0.16),
                (center, opacity * 0.72),
                (high, opacity * 0.28),
            ]
            return color_points, opacity_points

        color_points = [
            (low, *start_rgb),
            (center - width * 0.18, *mid_rgb),
            (center + width * 0.08, *end_rgb),
            (high, *end_rgb),
        ]
        opacity_points = [
            (low, 0.0),
            (center - width * 0.12, opacity * 0.18),
            (center, opacity * 0.6),
            (high, opacity * 0.9),
        ]
        return color_points, opacity_points

        if key == 'bone':
            knee = center + width * 0.12
            plateau = center + width * 0.35
            white_rgb = (1.0, 1.0, 1.0)
            color_points = [
                (low, *start_rgb),
                (center, *mid_rgb),
                (knee, *white_rgb),
                (high, *white_rgb),
                (plateau, *white_rgb),
            ]
            opacity_points = [
                (low, 0.0),
                (center, opacity * 0.35),
                (knee, opacity * 0.72),
                (high, opacity),
                (plateau, opacity),
            ]
            return color_points, opacity_points

        if key == 'lung':
            color_points = [
                (low, *start_rgb),
                (center - width * 0.18, *mid_rgb),
                (center, *end_rgb),
                (high, *end_rgb),
            ]
            opacity_points = [
                (low, 0.0),
                (center - width * 0.12, opacity * 0.2),
                (center, opacity * 0.8),
                (high, opacity * 0.35),
            ]
            return color_points, opacity_points

        color_points = [
            (low, *start_rgb),
            (center - width * 0.18, *mid_rgb),
            (center + width * 0.08, *end_rgb),
            (high, *end_rgb),
        ]
        opacity_points = [
            (low, 0.0),
            (center - width * 0.12, opacity * 0.25),
            (center, opacity * 0.72),
            (high, opacity * 0.9),
        ]
        return color_points, opacity_points

    @staticmethod
    def _build_gradient_opacity_points(
        preset: str,
        blend_mode: str,
        overlay_active: bool,
    ) -> list[tuple[float, float]]:
        if blend_mode == 'mip':
            return [
                (0.0, 0.0),
                (255.0, 0.0),
            ]

        if preset == 'bone':
            return [
                (0.0, 0.0),
                (18.0, 0.0),
                (54.0, 0.12 if not overlay_active else 0.18),
                (110.0, 0.34 if not overlay_active else 0.48),
                (210.0, 0.74 if not overlay_active else 0.86),
            ]

        if preset == 'aaa':
            return [
                (0.0, 0.0),
                (16.0, 0.0 if not overlay_active else 0.01),
                (48.0, 0.04 if not overlay_active else 0.10),
                (120.0, 0.16 if not overlay_active else 0.34),
                (220.0, 0.34 if not overlay_active else 0.76),
            ]

        if preset == 'cardiac':
            return [
                (0.0, 0.0),
                (18.0, 0.01),
                (60.0, 0.08),
                (120.0, 0.24),
                (220.0, 0.56),
            ]

        return [
            (0.0, 0.0),
            (18.0, 0.01),
            (72.0, 0.08),
            (180.0, 0.3),
        ]

    def _update_camera(self, session: VolumeRenderSession, request: VolumeRenderRequest) -> None:
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
        session: VolumeRenderSession,
        camera,
    ) -> tuple[float, float, float, float]:
        base_forward = self._normalize_vector(np.array(session.base_focal_point, dtype=np.float64) - np.array(session.base_position, dtype=np.float64))
        base_up = self._normalize_vector(np.array(session.base_view_up, dtype=np.float64))
        base_right = self._normalize_vector(np.cross(base_forward, base_up))
        current_forward = self._normalize_vector(np.array(camera.GetFocalPoint(), dtype=np.float64) - np.array(camera.GetPosition(), dtype=np.float64))
        current_up = self._normalize_vector(np.array(camera.GetViewUp(), dtype=np.float64))
        current_right = self._normalize_vector(np.cross(current_forward, current_up))

        base_basis = np.column_stack((base_right, base_up, -base_forward))
        current_basis = np.column_stack((current_right, current_up, -current_forward))
        rotation_matrix = current_basis @ base_basis.T
        return self._rotation_matrix_to_quaternion(rotation_matrix)

    @staticmethod
    def _clamp_unit_interval(value: Any, fallback: float) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return fallback
        return max(0.0, min(1.0, numeric))

    @staticmethod
    def _hex_to_rgb(color: str, fallback: tuple[float, float, float]) -> tuple[float, float, float]:
        text = str(color or '').strip()
        if len(text) == 7 and text.startswith('#'):
            try:
                return tuple(int(text[index:index + 2], 16) / 255.0 for index in (1, 3, 5))
            except ValueError:
                return fallback
        return fallback

    @staticmethod
    def _normalize_quaternion(quaternion: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        return normalize_quaternion(quaternion)

    @staticmethod
    def _normalize_vector(vector: np.ndarray) -> np.ndarray:
        return normalize_vector(vector)

    @staticmethod
    def _rotation_matrix_to_quaternion(matrix: np.ndarray) -> tuple[float, float, float, float]:
        return rotation_matrix_to_quaternion(matrix)

    @staticmethod
    def _quaternion_to_rotation_matrix(quaternion: tuple[float, float, float, float]) -> np.ndarray:
        return quaternion_to_rotation_matrix(quaternion)

    @staticmethod
    def _apply_pan(camera, renderer: vtkRenderer, request: VolumeRenderRequest) -> None:
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
    def _capture_image(session: VolumeRenderSession) -> Image.Image:
        session.window_to_image.Modified()
        session.window_to_image.Update()

        image_data = session.window_to_image.GetOutput()
        width, height, _ = image_data.GetDimensions()
        vtk_scalars = image_data.GetPointData().GetScalars()
        array = vtk_to_numpy(vtk_scalars).reshape(height, width, 3)
        array = np.flipud(array)
        return Image.fromarray(array)


vtk_volume_renderer = VtkVolumeRenderer()













