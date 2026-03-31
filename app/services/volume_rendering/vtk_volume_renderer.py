from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from math import radians, tan
from threading import RLock

import numpy as np
from PIL import Image
from vtk import (
    VTK_FLOAT,
    vtkColorTransferFunction,
    vtkImageData,
    vtkPiecewiseFunction,
    vtkRenderer,
    vtkRenderWindow,
    vtkSmartVolumeMapper,
    vtkVolume,
    vtkVolumeProperty,
    vtkWindowToImageFilter,
)
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy


BACKGROUND_RGB = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class VolumeRenderRequest:
    view_id: str
    volume: np.ndarray
    spacing_xyz: tuple[float, float, float]
    canvas_width: int
    canvas_height: int
    window_width: float
    window_center: float
    zoom: float
    offset_x: float
    offset_y: float
    rotation_quaternion: tuple[float, float, float, float]
    fast_preview: bool = False


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
            motion_factor = 36.0
            delta_azimuth = -20.0 / max(float(request.canvas_width), 1.0)
            delta_elevation = 20.0 / max(float(request.canvas_height), 1.0)
            camera.Azimuth(float(delta_x_pixels) * delta_azimuth * motion_factor)
            camera.Elevation(float(delta_y_pixels) * delta_elevation * motion_factor)
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
            mapper.SetInteractiveAdjustSampleDistances(1)
        if hasattr(mapper, "SetAutoAdjustSampleDistances"):
            mapper.SetAutoAdjustSampleDistances(1)

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

        self._update_transfer_functions(session, request.window_width, request.window_center)
        self._update_sampling(session, request.fast_preview)
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
    def _update_sampling(session: VolumeRenderSession, fast_preview: bool) -> None:
        if hasattr(session.mapper, "SetImageSampleDistance"):
            session.mapper.SetImageSampleDistance(2.6 if fast_preview else 1.4)

    def _update_transfer_functions(
        self,
        session: VolumeRenderSession,
        window_width: float,
        window_center: float,
    ) -> None:
        ww = max(float(window_width), 1.0)
        wc = float(window_center)
        low = wc - ww / 2.0
        high = wc + ww / 2.0
        q1 = wc - ww * 0.2
        q2 = wc + ww * 0.08
        q3 = wc + ww * 0.32

        session.color_func.RemoveAllPoints()
        session.color_func.AddRGBPoint(low, 0.0, 0.0, 0.0)
        session.color_func.AddRGBPoint(q1, 0.55, 0.42, 0.40)
        session.color_func.AddRGBPoint(q2, 0.84, 0.74, 0.68)
        session.color_func.AddRGBPoint(q3, 0.96, 0.92, 0.88)
        session.color_func.AddRGBPoint(high, 1.0, 0.98, 0.95)

        session.opacity_func.RemoveAllPoints()
        session.opacity_func.AddPoint(low, 0.0)
        session.opacity_func.AddPoint(wc - ww * 0.12, 0.01)
        session.opacity_func.AddPoint(wc + ww * 0.02, 0.08)
        session.opacity_func.AddPoint(q3, 0.35)
        session.opacity_func.AddPoint(high, 0.7)

        session.gradient_func.RemoveAllPoints()
        session.gradient_func.AddPoint(0.0, 0.0)
        session.gradient_func.AddPoint(15.0, 0.04)
        session.gradient_func.AddPoint(60.0, 0.18)
        session.gradient_func.AddPoint(180.0, 0.62)

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
        return Image.fromarray(array, mode="RGB")


vtk_volume_renderer = VtkVolumeRenderer()



