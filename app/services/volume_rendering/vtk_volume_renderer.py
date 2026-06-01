from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from math import radians, tan
from threading import RLock
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

from app.services.volume_rendering.contracts import VolumeRenderRequest


vtkObject.GlobalWarningDisplayOff()
BACKGROUND_RGB = (0.0, 0.0, 0.0)
TRACKBALL_MOTION_FACTOR = 36.0
TRACKBALL_AZIMUTH_DEGREES_PER_VIEW_WIDTH = -20.0
TRACKBALL_ELEVATION_DEGREES_PER_VIEW_HEIGHT = 20.0
FAST_PREVIEW_IMAGE_SAMPLE_DISTANCE = 1.9
FINAL_RENDER_IMAGE_SAMPLE_DISTANCE = 1.0
FAST_PREVIEW_RAY_SAMPLE_FACTOR = 1.45
FINAL_RENDER_RAY_SAMPLE_FACTOR = 0.72


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


class VtkVolumeRenderer:
    def __init__(self) -> None:
        self._sessions: dict[str, VolumeRenderSession] = {}
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
        if request.canvas_width <= 0 or request.canvas_height <= 0:
            raise ValueError("canvas size must be positive")
        if not request.view_id:
            raise ValueError("view_id is required")

        volume = np.asarray(request.volume, dtype=np.float32)
        if volume.ndim != 3 or volume.size == 0:
            raise ValueError("volume must be a non-empty 3D array")

        with self._lock:
            session = self._get_or_create_session(request, volume)
            self._configure_session(session, request)
            session.render_window.Render()
            return self._capture_image(session)

    def _drop_session_in_executor(self, view_id: str) -> None:
        with self._lock:
            session = self._sessions.pop(view_id, None)
            if session is None:
                return
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
        volume_token = self._build_volume_token(volume, request.spacing_xyz)
        session = self._sessions.get(request.view_id)
        if session is None or session.volume_token != volume_token:
            session = self._create_session(volume, request.spacing_xyz, volume_token)
            self._sessions[request.view_id] = session
        return session

    def _create_session(
        self,
        volume: np.ndarray,
        spacing_xyz: tuple[float, float, float],
        volume_token: tuple[object, tuple[int, ...], tuple[float, float, float]],
    ) -> VolumeRenderSession:
        image_data = self._build_image_data(volume, spacing_xyz)

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
        )

    def _configure_session(self, session: VolumeRenderSession, request: VolumeRenderRequest) -> None:
        if session.canvas_size != (request.canvas_width, request.canvas_height):
            session.render_window.SetSize(request.canvas_width, request.canvas_height)
            session.canvas_size = (request.canvas_width, request.canvas_height)

        self._update_transfer_functions(session, request.window_width, request.window_center, request.volume_preset, request.volume_config)
        self._update_sampling(session, request)
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
    ) -> tuple[object, tuple[int, ...], tuple[float, float, float]]:
        return (id(volume), tuple(int(size) for size in volume.shape), tuple(float(value) for value in spacing_xyz))

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
            min_spacing = max(1e-3, min(abs(float(value)) for value in request.spacing_xyz))
            factor = FAST_PREVIEW_RAY_SAMPLE_FACTOR if request.fast_preview else FINAL_RENDER_RAY_SAMPLE_FACTOR
            sample_distance = max(0.24, min(1.25, min_spacing * factor))
            session.mapper.SetSampleDistance(sample_distance)

    def _update_transfer_functions(
        self,
        session: VolumeRenderSession,
        window_width: float,
        window_center: float,
        volume_preset: str,
        volume_config: dict[str, Any] | None,
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

        if shading_enabled and layers and blend_mode != "mip":
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
        rotation_matrix = self._quaternion_to_rotation_matrix(request.rotation_quaternion)
        base_position = np.array(session.base_position, dtype=np.float64)
        base_focal_point = np.array(session.base_focal_point, dtype=np.float64)
        base_view_up = np.array(session.base_view_up, dtype=np.float64)
        relative_position = base_position - base_focal_point
        rotated_position = base_focal_point + rotation_matrix @ relative_position
        rotated_view_up = rotation_matrix @ base_view_up

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
        vector = np.asarray(quaternion, dtype=np.float64)
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            return (0.0, 0.0, 0.0, 1.0)
        vector /= norm
        return tuple(float(value) for value in vector)

    @staticmethod
    def _normalize_vector(vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            return np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        return vector / norm

    def _rotation_matrix_to_quaternion(self, matrix: np.ndarray) -> tuple[float, float, float, float]:
        trace = float(matrix[0, 0] + matrix[1, 1] + matrix[2, 2])
        if trace > 0.0:
            scale = np.sqrt(trace + 1.0) * 2.0
            w = 0.25 * scale
            x = (matrix[2, 1] - matrix[1, 2]) / scale
            y = (matrix[0, 2] - matrix[2, 0]) / scale
            z = (matrix[1, 0] - matrix[0, 1]) / scale
        elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            w = (matrix[2, 1] - matrix[1, 2]) / scale
            x = 0.25 * scale
            y = (matrix[0, 1] + matrix[1, 0]) / scale
            z = (matrix[0, 2] + matrix[2, 0]) / scale
        elif matrix[1, 1] > matrix[2, 2]:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            w = (matrix[0, 2] - matrix[2, 0]) / scale
            x = (matrix[0, 1] + matrix[1, 0]) / scale
            y = 0.25 * scale
            z = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            w = (matrix[1, 0] - matrix[0, 1]) / scale
            x = (matrix[0, 2] + matrix[2, 0]) / scale
            y = (matrix[1, 2] + matrix[2, 1]) / scale
            z = 0.25 * scale
        return self._normalize_quaternion((float(x), float(y), float(z), float(w)))

    def _quaternion_to_rotation_matrix(self, quaternion: tuple[float, float, float, float]) -> np.ndarray:
        x, y, z, w = self._normalize_quaternion(quaternion)
        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        wx = w * x
        wy = w * y
        wz = w * z
        return np.asarray(
            [
                [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
                [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
                [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
            ],
            dtype=np.float64,
        )

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













