import io
from copy import deepcopy
from dataclasses import dataclass, replace

import numpy as np
from fastapi import HTTPException
from PIL import Image
from pydicom.dataset import Dataset

from app.core import (
    DRAG_ACTION_END,
    DRAG_ACTION_MOVE,
    DRAG_ACTION_START,
    MPR_VIEWPORT_AXIAL,
    MPR_VIEWPORT_CORONAL,
    MPR_VIEWPORT_SAGITTAL,
    VIEW_OP_TYPE_CROSSHAIR,
    VIEW_OP_TYPE_PAN,
    VIEW_OP_TYPE_SET_SIZE,
    VIEW_OP_TYPE_WINDOW,
    VIEW_OP_TYPE_ZOOM,
    VIEW_OP_TYPE_ROTATE_3D,
    VIEW_OP_TYPE_VOLUME_CONFIG,
    WINDOW_DRAG_SENSITIVITY,
    WINDOW_WIDTH_MIN,
    ZOOM_DRAG_FACTOR_MIN,
    ZOOM_DRAG_SENSITIVITY,
    ZOOM_DRAG_SENSITIVITY_3D,
    ZOOM_MAX_3D,
    ZOOM_MIN_3D,
    VIEW_OP_TYPE_SCROLL,
)
from app.core.logging import get_logger
from app.models.viewer import InstanceRecord, SeriesRecord, ViewRecord
from app.schemas.dicom import CornerInfoPayload, CornerInfoRequest, CornerInfoResponse
from app.schemas.view import (
    ImageFormat,
    OperationAcceptedResponse,
    OrientationInfo,
    SliceInfo,
    MprCrosshairInfo,
    ViewHoverRequest,
    ViewHoverResponse,
    ViewImageResponse,
    ViewOperationRequest,
    ViewSetSizeRequest,
    WindowInfo,
    VolumeRenderConfig,
)
from app.services.dicom_cache import CachedDicom, dicom_cache
from app.services.layered_renderer import RenderContext, layered_renderer
from app.services.render_layers.render_context import CornerInfoOverlay, MprCrosshairOverlay, OrientationOverlay
from app.services.series_registry import series_registry
from app.services.viewport_transformer import viewport_transformer
from app.services.view_registry import view_registry
from app.services.viewer_operation_handlers import OperationRenderOutcome, handle_view_operation
from app.services.volume_rendering import VolumeRenderRequest, vtk_volume_renderer


logger = get_logger(__name__)

CROSSHAIR_HIT_RADIUS = 12.0


@dataclass(frozen=True)
class RenderedImageResult:
    meta: ViewImageResponse
    image_bytes: bytes


@dataclass(frozen=True)
class RenderPlan:
    render_view: ViewRecord
    render_ratio: float


class ViewerService:
    def __init__(self) -> None:
        self._volume_cache: dict[str, np.ndarray] = {}
        self._series_patient_transform_cache: dict[str, dict[str, object] | None] = {}
        self._logger = logger

    @staticmethod
    def _is_mpr_view_type(view_type: str) -> bool:
        return view_type in {"MPR", "AX", "COR", "SAG"}

    @staticmethod
    def _is_3d_view_type(view_type: str) -> bool:
        return view_type == "3D"

    def set_view_size(self, payload: ViewSetSizeRequest) -> OperationAcceptedResponse:
        if payload.op_type != VIEW_OP_TYPE_SET_SIZE:
            raise HTTPException(status_code=400, detail="opType must be setSize")

        view = view_registry.get(payload.view_id)
        view.width = payload.size.width
        view.height = payload.size.height
        logger.info(
            "set view size view_id=%s width=%s height=%s",
            view.view_id,
            view.width,
            view.height,
        )

        if not view.is_initialized:
            if self._is_mpr_view_type(view.view_type):
                self._initialize_mpr_viewport(view)
            elif self._is_3d_view_type(view.view_type):
                self._initialize_3d_viewport(view)
            else:
                self._initialize_viewport(view)
            view.is_initialized = True

        return OperationAcceptedResponse(message="View size updated", viewId=view.view_id)

    def render_view_by_id(
        self,
        view_id: str,
        *,
        image_format: ImageFormat = "png",
        fast_preview: bool = False,
    ) -> RenderedImageResult:
        view = view_registry.get(view_id)
        return self._render_by_view_type(view, image_format=image_format, fast_preview=fast_preview)

    def handle_view_operation(self, payload: ViewOperationRequest) -> OperationRenderOutcome:
        return handle_view_operation(self, payload)

    def handle_view_hover(self, payload: ViewHoverRequest) -> ViewHoverResponse:
        view = view_registry.get(payload.view_id)
        row, col = self._resolve_hover_row_col(view, payload.x, payload.y)
        return ViewHoverResponse(viewId=view.view_id, row=row, col=col)

    def get_series_corner_info(self, payload: CornerInfoRequest) -> CornerInfoResponse:
        series = series_registry.get(payload.series_id)
        _, reference_cached = self._get_reference_instance_and_cache(series)
        overlay = self._build_series_corner_info_overlay(
            series,
            reference_cached.dataset if reference_cached is not None else None,
        )
        return CornerInfoResponse(cornerInfo=self._serialize_corner_info_overlay(overlay))

    def _resolve_hover_row_col(self, view: ViewRecord, normalized_x: float, normalized_y: float) -> tuple[int, int]:
        if not view.width or not view.height or self._is_3d_view_type(view.view_type):
            return (0, 0)

        if self._is_mpr_view_type(view.view_type):
            series = series_registry.get(view.series_id)
            volume = self._get_series_volume(series)
            if not view.is_initialized:
                self._initialize_mpr_viewport(view)
                view.is_initialized = True
            target_viewport = self._resolve_mpr_viewport(view)
            plane_pixels, _, _ = self._extract_mpr_plane(view, volume, target_viewport)
            render_plan = self._build_render_plan_for_shape(view, *plane_pixels.shape[:2])
            image_transform = viewport_transformer.build_image_to_canvas_transform(
                image_width=plane_pixels.shape[1],
                image_height=plane_pixels.shape[0],
                canvas_width=render_plan.render_view.width or 0,
                canvas_height=render_plan.render_view.height or 0,
                view=render_plan.render_view,
            )
            return self._map_normalized_canvas_to_image_row_col(
                normalized_x,
                normalized_y,
                image_width=plane_pixels.shape[1],
                image_height=plane_pixels.shape[0],
                canvas_width=render_plan.render_view.width or 0,
                canvas_height=render_plan.render_view.height or 0,
                image_transform=image_transform,
            )

        series = series_registry.get(view.series_id)
        instance = series.instances[view.current_index]
        if not instance.sop_instance_uid:
            return (0, 0)
        cached = dicom_cache.get(instance.sop_instance_uid, instance.path)
        render_plan = self._build_render_plan_for_shape(view, *cached.source_pixels.shape[:2])
        image_transform = viewport_transformer.build_image_to_canvas_transform(
            image_width=cached.source_pixels.shape[1],
            image_height=cached.source_pixels.shape[0],
            canvas_width=render_plan.render_view.width or 0,
            canvas_height=render_plan.render_view.height or 0,
            view=render_plan.render_view,
        )
        return self._map_normalized_canvas_to_image_row_col(
            normalized_x,
            normalized_y,
            image_width=cached.source_pixels.shape[1],
            image_height=cached.source_pixels.shape[0],
            canvas_width=render_plan.render_view.width or 0,
            canvas_height=render_plan.render_view.height or 0,
            image_transform=image_transform,
        )

    def _map_normalized_canvas_to_image_row_col(
        self,
        normalized_x: float,
        normalized_y: float,
        *,
        image_width: int,
        image_height: int,
        canvas_width: int,
        canvas_height: int,
        image_transform,
    ) -> tuple[int, int]:
        if image_width <= 0 or image_height <= 0 or canvas_width <= 0 or canvas_height <= 0:
            return (0, 0)

        x = max(0.0, min(1.0, float(normalized_x)))
        y = max(0.0, min(1.0, float(normalized_y)))
        max_canvas_x = max(float(canvas_width) - 1e-6, 0.0)
        max_canvas_y = max(float(canvas_height) - 1e-6, 0.0)
        canvas_x = min(max(x * float(canvas_width), 0.0), max_canvas_x)
        canvas_y = min(max(y * float(canvas_height), 0.0), max_canvas_y)

        affine_matrix, offset = image_transform.inverse_components()
        source_point = affine_matrix @ np.asarray([canvas_x, canvas_y], dtype=np.float64) + offset
        source_x = float(source_point[0])
        source_y = float(source_point[1])

        if source_x < 0.0 or source_x >= float(image_width) or source_y < 0.0 or source_y >= float(image_height):
            return (0, 0)

        row = int(np.floor(source_y)) + 1
        col = int(np.floor(source_x)) + 1
        return (row, col)

    def _render_by_view_type(
        self,
        view: ViewRecord,
        image_format: ImageFormat = "png",
        *,
        fast_preview: bool = False,
    ) -> RenderedImageResult:
        if self._is_mpr_view_type(view.view_type):
            return self._render_mpr_view(view, image_format=image_format, fast_preview=fast_preview)
        if self._is_3d_view_type(view.view_type):
            return self._render_3d_view(view, image_format=image_format, fast_preview=fast_preview)
        return self._render_view(view, image_format=image_format, fast_preview=fast_preview)

    def _handle_scroll(self, view: ViewRecord, series: SeriesRecord, scroll: int) -> None:
        if not self._is_mpr_view_type(view.view_type):
            next_index = view.current_index + scroll
            view.current_index = max(0, min(next_index, len(series.instances) - 1))
            return

        volume = self._get_series_volume(series)
        depth, height, width = volume.shape
        target_viewport = self._resolve_mpr_viewport(view)
        if target_viewport == MPR_VIEWPORT_CORONAL:
            view.mpr_coronal_index = max(0, min(view.mpr_coronal_index + scroll, height - 1))
        elif target_viewport == MPR_VIEWPORT_SAGITTAL:
            view.mpr_sagittal_index = max(0, min(view.mpr_sagittal_index + scroll, width - 1))
        else:
            view.mpr_axial_index = max(0, min(view.mpr_axial_index + scroll, depth - 1))
        view.is_initialized = True

    def _initialize_viewport(self, view: ViewRecord) -> None:
        if not view.width or not view.height:
            raise HTTPException(status_code=400, detail="View size has not been set")

        series = series_registry.get(view.series_id)
        instance = series.instances[view.current_index]
        if not instance.sop_instance_uid:
            raise HTTPException(status_code=400, detail="DICOM instance does not contain SOPInstanceUID")

        cached = dicom_cache.get(instance.sop_instance_uid, instance.path)
        image_height, image_width = cached.source_pixels.shape[:2]
        view.zoom = viewport_transformer.calculate_contain_zoom(
            image_width=image_width,
            image_height=image_height,
            canvas_width=view.width,
            canvas_height=view.height,
        )
        view.offset_x = 0.0
        view.offset_y = 0.0
        view.window_width = cached.window_width or self._derive_default_window_width(cached)
        view.window_center = cached.window_center or self._derive_default_window_center(cached)
        self._reset_drag_state(view)
        logger.info(
            "viewport initialized view_id=%s image_width=%s image_height=%s zoom=%.4f ww=%s wl=%s",
            view.view_id,
            image_width,
            image_height,
            view.zoom,
            view.window_width,
            view.window_center,
        )

    def _initialize_mpr_viewport(self, view: ViewRecord) -> None:
        if not view.width or not view.height:
            raise HTTPException(status_code=400, detail="View size has not been set")

        series = series_registry.get(view.series_id)
        volume = self._get_series_volume(series)
        depth, height, width = volume.shape
        view.mpr_axial_index = depth // 2
        view.mpr_coronal_index = height // 2
        view.mpr_sagittal_index = width // 2
        view.current_index = view.mpr_axial_index
        view.offset_x = 0.0
        view.offset_y = 0.0

        first_instance = next((instance for instance in series.instances if instance.sop_instance_uid), None)
        if first_instance is not None and first_instance.sop_instance_uid:
            cached = dicom_cache.get(first_instance.sop_instance_uid, first_instance.path)
            view.window_width = cached.window_width or self._derive_default_window_width(cached)
            view.window_center = cached.window_center or self._derive_default_window_center(cached)
        else:
            pixel_min = float(np.min(volume))
            pixel_max = float(np.max(volume))
            view.window_width = max(WINDOW_WIDTH_MIN, pixel_max - pixel_min)
            view.window_center = (pixel_max + pixel_min) / 2.0

        plane_pixels, _, _ = self._extract_mpr_plane(view, volume)
        view.zoom = viewport_transformer.calculate_contain_zoom(
            image_width=plane_pixels.shape[1],
            image_height=plane_pixels.shape[0],
            canvas_width=view.width,
            canvas_height=view.height,
        )
        self._reset_drag_state(view)
        logger.info(
            "mpr viewport initialized view_id=%s volume=%s axial=%s coronal=%s sagittal=%s zoom=%.4f",
            view.view_id,
            volume.shape,
            view.mpr_axial_index,
            view.mpr_coronal_index,
            view.mpr_sagittal_index,
            view.zoom,
        )

    def _initialize_3d_viewport(self, view: ViewRecord) -> None:
        if not view.width or not view.height:
            raise HTTPException(status_code=400, detail="View size has not been set")

        series = series_registry.get(view.series_id)
        volume = self._get_series_volume(series)
        view.current_index = max(0, min(volume.shape[0] // 2, len(series.instances) - 1)) if series.instances else 0

        first_instance = next((instance for instance in series.instances if instance.sop_instance_uid), None)
        if first_instance is not None and first_instance.sop_instance_uid:
            cached = dicom_cache.get(first_instance.sop_instance_uid, first_instance.path)
            view.window_width = cached.window_width or self._derive_default_window_width(cached)
            view.window_center = cached.window_center or self._derive_default_window_center(cached)
        else:
            pixel_min = float(np.min(volume))
            pixel_max = float(np.max(volume))
            view.window_width = max(WINDOW_WIDTH_MIN, pixel_max - pixel_min)
            view.window_center = (pixel_max + pixel_min) / 2.0

        view.zoom = 1.0
        view.offset_x = 0.0
        view.offset_y = 0.0
        view.rotation_quaternion = vtk_volume_renderer.get_default_rotation_quaternion()
        view.volume_preset = "aaa"
        view.volume_render_config = self._create_default_volume_render_config("aaa")
        self._reset_drag_state(view)
        logger.info(
            "3d viewport initialized view_id=%s volume=%s zoom=%.4f ww=%s wl=%s",
            view.view_id,
            volume.shape,
            view.zoom,
            view.window_width,
            view.window_center,
        )

    @staticmethod
    def _hex_to_rgb(color: str, fallback: tuple[float, float, float]) -> tuple[float, float, float]:
        text = str(color or '').strip()
        if len(text) == 7 and text.startswith('#'):
            try:
                return tuple(int(text[index:index + 2], 16) / 255.0 for index in (1, 3, 5))
            except ValueError:
                return fallback
        return fallback

    def _create_default_volume_render_config(self, preset_value: str) -> dict[str, object]:
        preset = str(preset_value or 'aaa').strip().lower()
        if ':' in preset:
            preset = preset.split(':', 1)[1]

        layers = {
            'bone': {
                'key': 'bone',
                'label': '\u9aa8\u9abc',
                'enabled': False,
                'ww': 500.0,
                'wl': 400.0,
                'opacity': 1.0,
                'colorStart': '#ffffff',
                'colorEnd': '#ffffff',
            },
            'blood': {
                'key': 'blood',
                'label': '\u8840\u6db2',
                'enabled': False,
                'ww': 200.0,
                'wl': 220.0,
                'opacity': 0.2,
                'colorStart': '#d31b1b',
                'colorEnd': '#ffd54a',
            },
            'muscle': {
                'key': 'muscle',
                'label': '\u808c\u8089',
                'enabled': False,
                'ww': 320.0,
                'wl': 45.0,
                'opacity': 0.55,
                'colorStart': '#f2c7b8',
                'colorEnd': '#8a3426',
            },
            'softTissue': {
                'key': 'softTissue',
                'label': '\u8f6f\u7ec4\u7ec7',
                'enabled': False,
                'ww': 380.0,
                'wl': 55.0,
                'opacity': 0.32,
                'colorStart': '#f1d8c8',
                'colorEnd': '#b06b56',
            },
            'lung': {
                'key': 'lung',
                'label': '\u80ba',
                'enabled': False,
                'ww': 1500.0,
                'wl': -550.0,
                'opacity': 0.22,
                'colorStart': '#9fd8ff',
                'colorEnd': '#e5f6ff',
            },
            'custom': {
                'key': 'custom',
                'label': '\u81ea\u5b9a\u4e49',
                'enabled': False,
                'ww': 400.0,
                'wl': 40.0,
                'opacity': 0.3,
                'colorStart': '#7dd3fc',
                'colorEnd': '#f8fafc',
            },
        }
        blend_mode = 'composite'
        lighting = {
            'shading': True,
            'interpolation': 'linear',
            'ambient': 0.16,
            'diffuse': 0.86,
            'specular': 0.18,
            'roughness': 0.78,
        }

        if preset == 'aaa':
            layers['bone'].update({'enabled': True, 'ww': 500.0, 'wl': 400.0, 'opacity': 1.0, 'colorStart': '#ffffff', 'colorEnd': '#ffffff'})
            layers['blood'].update({'enabled': True, 'ww': 200.0, 'wl': 220.0, 'opacity': 0.2, 'colorStart': '#d31b1b', 'colorEnd': '#ffd54a'})
            lighting.update({'shading': True, 'interpolation': 'linear', 'ambient': 0.12, 'diffuse': 0.9, 'specular': 0.2, 'roughness': 0.74})
        elif preset == 'red':
            layers['bone'].update({'enabled': True, 'ww': 442.0, 'wl': 115.0, 'opacity': 1.0, 'colorStart': '#c31616', 'colorEnd': '#ff6666'})
            lighting.update({'shading': True, 'interpolation': 'linear', 'ambient': 0.14, 'diffuse': 0.88, 'specular': 0.16, 'roughness': 0.8})
        elif preset == 'cardiac':
            layers['bone'].update({'enabled': True, 'ww': 170.0, 'wl': 176.0, 'opacity': 0.9, 'colorStart': '#fff9f2', 'colorEnd': '#7f1720'})
            layers['blood'].update({'enabled': True, 'ww': 170.0, 'wl': 7.0, 'opacity': 0.3, 'colorStart': '#ffe082', 'colorEnd': '#ffb300'})
            lighting.update({'shading': True, 'interpolation': 'linear', 'ambient': 0.1, 'diffuse': 0.88, 'specular': 0.22, 'roughness': 0.72})
        elif preset == 'muscle':
            layers['muscle'].update({'enabled': True, 'ww': 280.0, 'wl': 40.0, 'opacity': 0.58, 'colorStart': '#f4cfbf', 'colorEnd': '#8c3d2e'})
            layers['softTissue'].update({'enabled': True, 'ww': 360.0, 'wl': 50.0, 'opacity': 0.28, 'colorStart': '#f3ddd1', 'colorEnd': '#9e6a5a'})
            lighting.update({'shading': True, 'interpolation': 'linear', 'ambient': 0.18, 'diffuse': 0.82, 'specular': 0.08, 'roughness': 0.9})
        elif preset == 'mip':
            blend_mode = 'mip'
            layers['bone'].update({'enabled': True, 'ww': 900.0, 'wl': 350.0, 'opacity': 0.35, 'colorStart': '#9a9a9a', 'colorEnd': '#ffffff'})
            layers['blood'].update({'enabled': True, 'ww': 260.0, 'wl': 200.0, 'opacity': 0.85, 'colorStart': '#f7f1b6', 'colorEnd': '#ffffff'})
            lighting.update({'shading': False, 'interpolation': 'linear', 'ambient': 1.0, 'diffuse': 0.0, 'specular': 0.0, 'roughness': 1.0})
        else:
            preset = 'aaa'
            layers['bone'].update({'enabled': True, 'ww': 500.0, 'wl': 400.0, 'opacity': 1.0, 'colorStart': '#ffffff', 'colorEnd': '#ffffff'})
            layers['blood'].update({'enabled': True, 'ww': 200.0, 'wl': 220.0, 'opacity': 0.2, 'colorStart': '#d31b1b', 'colorEnd': '#ffd54a'})
            lighting.update({'shading': True, 'interpolation': 'linear', 'ambient': 0.12, 'diffuse': 0.9, 'specular': 0.2, 'roughness': 0.74})

        return {
            'preset': preset,
            'blendMode': blend_mode,
            'layers': list(layers.values()),
            'lighting': lighting,
        }

    def _normalize_volume_render_config(self, value: VolumeRenderConfig | dict[str, object] | None, fallback_preset: str) -> dict[str, object]:
        fallback = self._create_default_volume_render_config(fallback_preset)
        if value is None:
            return fallback

        if isinstance(value, VolumeRenderConfig):
            payload = value.model_dump(by_alias=True)
        else:
            payload = dict(value)

        preset = str(payload.get('preset') or fallback['preset']).strip().lower()
        normalized = self._create_default_volume_render_config(preset)
        normalized['blendMode'] = 'mip' if payload.get('blendMode') == 'mip' else 'composite'

        incoming_layers = payload.get('layers') if isinstance(payload.get('layers'), list) else []
        layer_map = {str(layer['key']): layer for layer in normalized['layers'] if isinstance(layer, dict)}

        for entry in incoming_layers:
            if not isinstance(entry, dict):
                continue
            layer = layer_map.get(str(entry.get('key') or ''))
            if layer is None:
                continue
            layer['label'] = str(entry.get('label') or layer['label'])
            layer['enabled'] = bool(entry.get('enabled', layer['enabled']))
            layer['ww'] = max(1.0, float(entry.get('ww', layer['ww'])))
            layer['wl'] = float(entry.get('wl', layer['wl']))
            layer['opacity'] = self._normalize_unit_interval(entry.get('opacity'), float(layer['opacity']))
            layer['colorStart'] = self._normalize_hex_color(str(entry.get('colorStart') or layer['colorStart']), str(layer['colorStart']))
            layer['colorEnd'] = self._normalize_hex_color(str(entry.get('colorEnd') or layer['colorEnd']), str(layer['colorEnd']))

        lighting_payload = payload.get('lighting') if isinstance(payload.get('lighting'), dict) else {}
        lighting = normalized.get('lighting') if isinstance(normalized.get('lighting'), dict) else {}
        lighting['shading'] = bool(lighting_payload.get('shading', lighting.get('shading', True)))
        lighting['interpolation'] = self._normalize_volume_interpolation(
            str(lighting_payload.get('interpolation') or lighting.get('interpolation') or 'linear')
        )
        lighting['ambient'] = self._normalize_unit_interval(lighting_payload.get('ambient'), float(lighting.get('ambient', 0.18)))
        lighting['diffuse'] = self._normalize_unit_interval(lighting_payload.get('diffuse'), float(lighting.get('diffuse', 0.82)))
        lighting['specular'] = self._normalize_unit_interval(lighting_payload.get('specular'), float(lighting.get('specular', 0.12)))
        lighting['roughness'] = self._normalize_unit_interval(lighting_payload.get('roughness'), float(lighting.get('roughness', 0.85)))
        normalized['lighting'] = lighting

        return normalized

    @staticmethod
    def _normalize_hex_color(value: str, fallback: str) -> str:
        text = str(value or '').strip().lower()
        if len(text) == 7 and text.startswith('#') and all(ch in '0123456789abcdef' for ch in text[1:]):
            return text
        return fallback

    @staticmethod
    def _normalize_unit_interval(value: object, fallback: float) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return fallback
        return max(0.0, min(1.0, numeric))

    @staticmethod
    def _normalize_volume_interpolation(value: str) -> str:
        normalized = str(value or 'linear').strip().lower()
        if normalized in {'nearest', 'linear', 'cubic'}:
            return normalized
        return 'linear'

    def _reset_view(self, view: ViewRecord) -> None:
        view.hor_flip = False
        view.ver_flip = False

        if self._is_mpr_view_type(view.view_type):
            self._initialize_mpr_viewport(view)
        elif self._is_3d_view_type(view.view_type):
            self._initialize_3d_viewport(view)
        else:
            self._initialize_viewport(view)

        view.is_initialized = True

    def _render_3d_view(
        self,
        view: ViewRecord,
        image_format: ImageFormat = "png",
        *,
        fast_preview: bool = False,
    ) -> RenderedImageResult:
        if not view.width or not view.height:
            raise HTTPException(status_code=400, detail="View size has not been set")

        series = series_registry.get(view.series_id)
        volume = self._get_series_volume(series)
        if not view.is_initialized:
            self._initialize_3d_viewport(view)
            view.is_initialized = True

        spacing_xyz = self._get_3d_spacing_xyz(series)
        image = vtk_volume_renderer.render(
            VolumeRenderRequest(
                view_id=view.view_id,
                volume=volume,
                spacing_xyz=spacing_xyz,
                canvas_width=view.width,
                canvas_height=view.height,
                window_width=float(view.window_width or WINDOW_WIDTH_MIN),
                window_center=float(view.window_center or 0.0),
                zoom=float(view.zoom),
                offset_x=float(view.offset_x),
                offset_y=float(view.offset_y),
                rotation_quaternion=tuple(float(value) for value in view.rotation_quaternion),
                volume_preset=str(view.volume_preset or "aaa"),
                volume_config=view.volume_render_config,
                fast_preview=fast_preview,
            )
        )

        corner_info = self._build_slice_corner_info_overlay(
            view,
            series,
            None,
            current_index=view.current_index,
            total_slices=max(1, volume.shape[0]),
            viewport_label="3D VR",
        )

        return RenderedImageResult(
            meta=ViewImageResponse(
                slice_info=SliceInfo(current=view.current_index, total=max(1, volume.shape[0])),
                window_info=WindowInfo(ww=view.window_width, wl=view.window_center),
                imageFormat=image_format,
                viewId=view.view_id,
                cornerInfo=self._serialize_corner_info_overlay(corner_info),
                orientation=self._build_3d_orientation_overlay(view),
                volumePreset=str(view.volume_preset or "aaa"),
                volumeConfig=view.volume_render_config,
            ),
            image_bytes=self._encode_image(image, image_format),
        )

    def _render_view(
        self,
        view: ViewRecord,
        image_format: ImageFormat = "png",
        *,
        fast_preview: bool = False,
    ) -> RenderedImageResult:
        if not view.width or not view.height:
            raise HTTPException(status_code=400, detail="View size has not been set")

        series = series_registry.get(view.series_id)
        instance = series.instances[view.current_index]
        if not instance.sop_instance_uid:
            raise HTTPException(status_code=400, detail="DICOM instance does not contain SOPInstanceUID")

        cached = dicom_cache.get(instance.sop_instance_uid, instance.path)
        render_plan = self._build_render_plan_for_shape(view, *cached.source_pixels.shape[:2])
        image_transform = viewport_transformer.build_image_to_canvas_transform(
            image_width=cached.source_pixels.shape[1],
            image_height=cached.source_pixels.shape[0],
            canvas_width=render_plan.render_view.width or 0,
            canvas_height=render_plan.render_view.height or 0,
            view=render_plan.render_view,
        )
        slice_corner_info = self._build_slice_corner_info_overlay(
            view,
            series,
            cached.dataset,
            current_index=view.current_index,
            total_slices=len(series.instances),
            viewport_label="Stack",
        )
        context = RenderContext(
            view=render_plan.render_view,
            source_pixels=cached.source_pixels,
            pixel_min=cached.pixel_min,
            pixel_max=cached.pixel_max,
            instance=instance,
            cached=cached,
            image_transform=image_transform,
            corner_info=None,
            orientation=None,
        )

        if fast_preview:
            image = self._render_fast_preview(context)
        else:
            image = layered_renderer.render(context)

        logger.debug(
            "render completed view_id=%s index=%s viewport=%sx%s render=%sx%s ratio=%.4f zoom=%.4f ww=%s wl=%s image_format=%s fast_window=%s",
            view.view_id,
            view.current_index,
            view.width,
            view.height,
            render_plan.render_view.width,
            render_plan.render_view.height,
            render_plan.render_ratio,
            view.zoom,
            view.window_width,
            view.window_center,
            image_format,
            fast_preview,
        )

        return RenderedImageResult(
            meta=ViewImageResponse(
                slice_info=SliceInfo(current=view.current_index, total=len(series.instances)),
                window_info=WindowInfo(ww=view.window_width, wl=view.window_center),
                imageFormat=image_format,
                viewId=view.view_id,
                cornerInfo=self._serialize_corner_info_overlay(slice_corner_info),
                orientation=self._serialize_orientation_overlay(
                    self._build_stack_orientation_overlay(render_plan.render_view, cached.dataset)
                ),
            ),
            image_bytes=self._encode_image(image, image_format),
        )

    def _render_mpr_view(
        self,
        view: ViewRecord,
        image_format: ImageFormat = "png",
        *,
        fast_preview: bool = False,
    ) -> RenderedImageResult:
        if not view.width or not view.height:
            raise HTTPException(status_code=400, detail="View size has not been set")

        series = series_registry.get(view.series_id)
        volume = self._get_series_volume(series)
        if not view.is_initialized:
            self._initialize_mpr_viewport(view)
            view.is_initialized = True

        target_viewport = self._resolve_mpr_viewport(view)
        plane_pixels, current, total = self._extract_mpr_plane(view, volume, target_viewport)
        render_plan = self._build_render_plan_for_shape(view, *plane_pixels.shape[:2])
        image_transform = viewport_transformer.build_image_to_canvas_transform(
            image_width=plane_pixels.shape[1],
            image_height=plane_pixels.shape[0],
            canvas_width=render_plan.render_view.width or 0,
            canvas_height=render_plan.render_view.height or 0,
            view=render_plan.render_view,
        )
        plane_min = float(np.min(plane_pixels))
        plane_max = float(np.max(plane_pixels))
        mpr_crosshair_overlay = self._build_mpr_crosshair_overlay(view, volume.shape, plane_pixels.shape, image_transform)
        reference_instance, reference_cached = self._get_reference_instance_and_cache(series)
        slice_corner_info = self._build_slice_corner_info_overlay(
            view,
            series,
            reference_cached.dataset if reference_cached is not None else None,
            current_index=current,
            total_slices=total,
            viewport_label=self._build_mpr_viewport_label(target_viewport),
        )
        context = RenderContext(
            view=render_plan.render_view,
            source_pixels=plane_pixels,
            pixel_min=plane_min,
            pixel_max=plane_max,
            image_transform=image_transform,
            instance=reference_instance,
            cached=reference_cached,
            mpr_viewport=target_viewport,
            mpr_crosshair=None,
            corner_info=None,
            orientation=None,
        )
        if fast_preview:
            image = self._render_fast_mpr_preview(context)
        else:
            image = layered_renderer.render(context)

        return RenderedImageResult(
            meta=ViewImageResponse(
                slice_info=SliceInfo(current=current, total=total),
                window_info=WindowInfo(ww=view.window_width, wl=view.window_center),
                imageFormat=image_format,
                viewId=view.view_id,
                mpr_crosshair=self._build_mpr_crosshair_info(mpr_crosshair_overlay),
                cornerInfo=self._serialize_corner_info_overlay(slice_corner_info),
                orientation=self._serialize_orientation_overlay(
                    self._build_mpr_orientation_overlay(render_plan.render_view, target_viewport)
                ),
            ),
            image_bytes=self._encode_image(image, image_format),
        )

    @staticmethod
    def _render_fast_mpr_preview(context: RenderContext) -> Image.Image:
        return ViewerService._render_fast_preview(context)

    @staticmethod
    def _render_fast_preview(context: RenderContext) -> Image.Image:
        image = ViewerService._render_fast_grayscale_image(
            source_pixels=context.source_pixels,
            pixel_min=context.pixel_min,
            pixel_max=context.pixel_max,
            render_view=context.view,
            image_transform=context.image_transform,
        ).convert("RGBA")
        return layered_renderer.composite_overlays(image, context)

    @staticmethod
    def _render_fast_grayscale_image(
        source_pixels: np.ndarray,
        pixel_min: float,
        pixel_max: float,
        render_view: ViewRecord,
        image_transform,
    ) -> Image.Image:
        base_pixels = ViewerService._window_array(
            source_pixels,
            render_view.window_width,
            render_view.window_center,
            pixel_min=pixel_min,
            pixel_max=pixel_max,
        )
        transformed = viewport_transformer.apply_affine_array(
            base_pixels,
            render_view.width or 0,
            render_view.height or 0,
            image_transform,
            order=1,
            cval=0.0,
        )
        return Image.fromarray(transformed, mode="L")

    def _build_render_plan_for_shape(self, view: ViewRecord, image_height: int, image_width: int) -> RenderPlan:
        render_ratio = self._resolve_render_ratio_for_shape(view, image_height, image_width)
        if render_ratio >= 0.999:
            return RenderPlan(render_view=view, render_ratio=1.0)

        render_width = max(1, int(round((view.width or 1) * render_ratio)))
        render_height = max(1, int(round((view.height or 1) * render_ratio)))
        scaled_transform = replace(
            view.transform,
            zoom=view.zoom * render_ratio,
            offset_x=view.offset_x * render_ratio,
            offset_y=view.offset_y * render_ratio,
        )
        render_view = replace(
            view,
            width=render_width,
            height=render_height,
            transform=scaled_transform,
        )
        return RenderPlan(render_view=render_view, render_ratio=render_ratio)

    @staticmethod
    def _resolve_render_ratio_for_shape(view: ViewRecord, image_height: int, image_width: int) -> float:
        if not view.width or not view.height:
            return 1.0

        if view.width <= image_width or view.height <= image_height:
            return 1.0

        contain_zoom = viewport_transformer.calculate_contain_zoom(
            image_width=image_width,
            image_height=image_height,
            canvas_width=view.width,
            canvas_height=view.height,
        )
        if view.zoom > contain_zoom:
            return 1.0

        width_ratio = image_width / view.width
        height_ratio = image_height / view.height
        return max(width_ratio, height_ratio)

    @staticmethod
    def _get_mpr_plane_shape(volume_shape: tuple[int, int, int], viewport_key: str) -> tuple[int, int]:
        depth, height, width = volume_shape
        if viewport_key == MPR_VIEWPORT_CORONAL:
            return depth, width
        if viewport_key == MPR_VIEWPORT_SAGITTAL:
            return depth, height
        return height, width

    def _extract_mpr_plane(
        self,
        view: ViewRecord,
        volume: np.ndarray,
        viewport_key: str | None = None,
    ) -> tuple[np.ndarray, int, int]:
        depth, height, width = volume.shape
        target_viewport = viewport_key or self._resolve_mpr_viewport(view)
        if target_viewport == MPR_VIEWPORT_CORONAL:
            index = max(0, min(view.mpr_coronal_index, height - 1))
            # 浠?3D volume 鍒囧嚭鏉ョ殑涓€寮?2D 鍒囬潰
            # np.flipud(...)锛氭妸杩欏紶鍒囬潰涓婁笅缈昏繃鏉?
            plane = np.flipud(volume[:, index, :])
            return plane.astype(np.float32), index, height
        if target_viewport == MPR_VIEWPORT_SAGITTAL:
            index = max(0, min(view.mpr_sagittal_index, width - 1))
            plane = np.flipud(volume[:, :, index])
            return plane.astype(np.float32), index, width
        index = max(0, min(view.mpr_axial_index, depth - 1))
        view.current_index = index
        plane = volume[index, :, :]
        return plane.astype(np.float32), index, depth

    @staticmethod
    def _clamp_3d_zoom(zoom: float) -> float:
        return min(max(float(zoom), ZOOM_MIN_3D), ZOOM_MAX_3D)

    def _get_3d_spacing_xyz(self, series: SeriesRecord) -> tuple[float, float, float]:
        transform = self._get_series_patient_transform(series)
        if transform is not None:
            axis_vectors = transform.get("axis_vectors")
            if isinstance(axis_vectors, tuple) and len(axis_vectors) == 3:
                spacing = tuple(max(float(np.linalg.norm(axis_vectors[index])), 1e-3) for index in (2, 1, 0))
                return spacing

        reference_instance, reference_cached = self._get_reference_instance_and_cache(series)
        dataset = reference_cached.dataset if reference_cached is not None else None
        pixel_spacing = getattr(dataset, "PixelSpacing", None) if dataset is not None else None
        slice_spacing = self._estimate_slice_spacing([], np.array([1.0, 0.0, 0.0], dtype=np.float64), dataset)
        if pixel_spacing is not None and len(pixel_spacing) >= 2:
            try:
                row_spacing = max(abs(float(pixel_spacing[0])), 1e-3)
                col_spacing = max(abs(float(pixel_spacing[1])), 1e-3)
                return (col_spacing, row_spacing, max(slice_spacing, 1e-3))
            except (TypeError, ValueError):
                pass
        return (1.0, 1.0, 1.0)

    def _get_series_volume(self, series: SeriesRecord) -> np.ndarray:
        cached_volume = self._volume_cache.get(series.series_id)
        if cached_volume is not None:
            return cached_volume

        slice_entries: list[tuple[np.ndarray, np.ndarray | None, np.ndarray | None]] = []
        for instance in series.instances:
            if not instance.sop_instance_uid:
                continue
            cached = dicom_cache.get(instance.sop_instance_uid, instance.path)
            dataset = cached.dataset
            orientation = self._get_dataset_orientation(dataset)
            position = self._get_dataset_position(dataset)
            slice_entries.append((cached.source_pixels, orientation, position))

        if not slice_entries:
            raise HTTPException(status_code=400, detail="Series does not contain readable pixel data")

        first_shape = slice_entries[0][0].shape
        if any(item[0].shape != first_shape for item in slice_entries):
            raise HTTPException(status_code=400, detail="MPR requires a series with consistent slice dimensions")

        volume = self._build_standardized_volume(slice_entries)
        self._volume_cache[series.series_id] = volume
        return volume

    @staticmethod
    def _get_dataset_orientation(dataset) -> np.ndarray | None:
        value = getattr(dataset, "ImageOrientationPatient", None)
        if value is None or len(value) < 6:
            return None
        try:
            orientation = np.asarray([float(item) for item in value[:6]], dtype=np.float64)
        except (TypeError, ValueError):
            self._series_patient_transform_cache[series.series_id] = None
            return None
        return orientation if np.all(np.isfinite(orientation)) else None

    @staticmethod
    def _get_dataset_position(dataset) -> np.ndarray | None:
        value = getattr(dataset, "ImagePositionPatient", None)
        if value is None or len(value) < 3:
            return None
        try:
            position = np.asarray([float(item) for item in value[:3]], dtype=np.float64)
        except (TypeError, ValueError):
            self._series_patient_transform_cache[series.series_id] = None
            return None
        return position if np.all(np.isfinite(position)) else None

    @staticmethod
    def _normalize_vector(vector: np.ndarray) -> np.ndarray | None:
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-6:
            return None
        return vector / norm

    def _build_standardized_volume(
        self,
        slice_entries: list[tuple[np.ndarray, np.ndarray | None, np.ndarray | None]],
    ) -> np.ndarray:
        orientation = next((item[1] for item in slice_entries if item[1] is not None), None)
        if orientation is None:
            return np.stack([item[0] for item in slice_entries], axis=0).astype(np.float32)

        row_direction = self._normalize_vector(orientation[:3])
        column_direction = self._normalize_vector(orientation[3:6])
        if row_direction is None or column_direction is None:
            return np.stack([item[0] for item in slice_entries], axis=0).astype(np.float32)

        slice_direction = self._normalize_vector(np.cross(row_direction, column_direction))
        if slice_direction is None:
            return np.stack([item[0] for item in slice_entries], axis=0).astype(np.float32)

        positions = [item[2] for item in slice_entries]
        if any(position is None for position in positions):
            ordered_entries = slice_entries
        else:
            ordered_entries = sorted(
                slice_entries,
                key=lambda item: float(np.dot(item[2], slice_direction)) if item[2] is not None else 0.0,
            )

        raw_volume = np.stack([item[0] for item in ordered_entries], axis=0).astype(np.float32)
        raw_axis_vectors = (slice_direction, column_direction, row_direction)
        patient_axes: list[int] = []
        axis_signs: list[int] = []

        for vector in raw_axis_vectors:
            patient_axis = int(np.argmax(np.abs(vector)))
            if patient_axis in patient_axes:
                logger.warning("falling back to non-standardized volume because orientation axes are not orthogonal enough")
                return raw_volume
            patient_axes.append(patient_axis)
            axis_signs.append(1 if vector[patient_axis] >= 0 else -1)

        transpose_order = [patient_axes.index(2), patient_axes.index(1), patient_axes.index(0)]
        canonical_signs = [
            axis_signs[patient_axes.index(2)],
            axis_signs[patient_axes.index(1)],
            axis_signs[patient_axes.index(0)],
        ]
        volume = np.transpose(raw_volume, axes=transpose_order)
        for axis, sign in enumerate(canonical_signs):
            if sign < 0:
                volume = np.flip(volume, axis=axis)

        logger.info(
            "standardized MPR volume shape=%s raw_axes=%s canonical_signs=%s row_dir=%s col_dir=%s slice_dir=%s",
            volume.shape,
            patient_axes,
            canonical_signs,
            np.round(row_direction, 4).tolist(),
            np.round(column_direction, 4).tolist(),
            np.round(slice_direction, 4).tolist(),
        )
        return volume.astype(np.float32, copy=False)

    def _handle_drag_pan(self, view: ViewRecord, payload: ViewOperationRequest) -> None:
        if payload.action_type == DRAG_ACTION_START:
            view.drag_origin_offset_x = view.offset_x
            view.drag_origin_offset_y = view.offset_y
            return

        if payload.action_type == DRAG_ACTION_MOVE:
            base_offset_x = view.drag_origin_offset_x if view.drag_origin_offset_x is not None else view.offset_x
            base_offset_y = view.drag_origin_offset_y if view.drag_origin_offset_y is not None else view.offset_y
            view.offset_x = float(base_offset_x) + float(payload.x or 0.0)
            view.offset_y = float(base_offset_y) + float(payload.y or 0.0)
            view.is_initialized = True
            return

        if payload.action_type == DRAG_ACTION_END:
            view.drag_origin_offset_x = None
            view.drag_origin_offset_y = None

    def _handle_drag_rotate_3d(self, view: ViewRecord, payload: ViewOperationRequest) -> None:
        if payload.action_type not in {DRAG_ACTION_START, DRAG_ACTION_MOVE, DRAG_ACTION_END}:
            return
        if payload.x is None or payload.y is None or not view.width or not view.height:
            return

        if payload.action_type == DRAG_ACTION_START:
            view.drag_origin_arcball_x = float(payload.x)
            view.drag_origin_arcball_y = float(payload.y)
            return

        if payload.action_type == DRAG_ACTION_END:
            view.drag_origin_arcball_x = None
            view.drag_origin_arcball_y = None
            return

        previous_x = view.drag_origin_arcball_x
        previous_y = view.drag_origin_arcball_y
        current_x = float(payload.x)
        current_y = float(payload.y)
        view.drag_origin_arcball_x = current_x
        view.drag_origin_arcball_y = current_y
        if previous_x is None or previous_y is None:
            return

        delta_x_pixels = (current_x - previous_x) * float(view.width)
        delta_y_pixels = (current_y - previous_y) * float(view.height)
        if abs(delta_x_pixels) < 0.01 and abs(delta_y_pixels) < 0.01:
            return

        series = series_registry.get(view.series_id)
        volume = self._get_series_volume(series)
        spacing_xyz = self._get_3d_spacing_xyz(series)
        view.rotation_quaternion = vtk_volume_renderer.apply_trackball_camera_delta(
            VolumeRenderRequest(
                view_id=view.view_id,
                volume=volume,
                spacing_xyz=spacing_xyz,
                canvas_width=view.width,
                canvas_height=view.height,
                window_width=float(view.window_width or WINDOW_WIDTH_MIN),
                window_center=float(view.window_center or 0.0),
                zoom=float(view.zoom),
                offset_x=float(view.offset_x),
                offset_y=float(view.offset_y),
                rotation_quaternion=tuple(float(value) for value in view.rotation_quaternion),
                volume_preset=str(view.volume_preset or "aaa"),
                volume_config=view.volume_render_config,
                fast_preview=True,
            ),
            delta_x_pixels=delta_x_pixels,
            delta_y_pixels=delta_y_pixels,
        )
        view.is_initialized = True

    def _handle_volume_config(self, view: ViewRecord, payload: ViewOperationRequest) -> None:
        if not self._is_3d_view_type(view.view_type):
            return
        view.volume_render_config = self._normalize_volume_render_config(payload.volume_config, view.volume_preset)
        view.volume_preset = str(view.volume_render_config.get('preset', view.volume_preset or 'aaa'))
        view.is_initialized = True

    def _handle_volume_preset(self, view: ViewRecord, payload: ViewOperationRequest) -> None:
        if not self._is_3d_view_type(view.view_type):
            return

        preset_value = (payload.sub_op_type or "aaa").strip().lower()
        if ":" in preset_value:
            preset_value = preset_value.split(":", 1)[1]

        preset_aliases = {
            "aaa": "aaa",
            "red": "red",
            "cardiac": "cardiac",
            "cardiac-muscle": "cardiac",
            "cardiac_muscle": "cardiac",
            "cardiac muscle": "cardiac",
            "muscle": "muscle",
            "mip": "mip",
        }
        view.volume_preset = preset_aliases.get(preset_value, "aaa")
        view.volume_render_config = self._create_default_volume_render_config(view.volume_preset)
        view.is_initialized = True

    def _handle_drag_zoom(self, view: ViewRecord, payload: ViewOperationRequest) -> None:
        if payload.action_type == DRAG_ACTION_START:
            view.drag_origin_zoom = view.zoom
            return

        if payload.action_type == DRAG_ACTION_MOVE:
            base_zoom = view.drag_origin_zoom if view.drag_origin_zoom is not None else view.zoom
            delta_y = float(payload.y or 0.0)
            zoom_sensitivity = ZOOM_DRAG_SENSITIVITY_3D if self._is_3d_view_type(view.view_type) else ZOOM_DRAG_SENSITIVITY
            zoom_factor = 1.0 - delta_y * zoom_sensitivity
            zoom_factor = max(ZOOM_DRAG_FACTOR_MIN, zoom_factor)
            next_zoom = viewport_transformer.clamp_zoom(float(base_zoom) * zoom_factor)
            if self._is_3d_view_type(view.view_type):
                next_zoom = self._clamp_3d_zoom(next_zoom)
            view.zoom = next_zoom
            view.is_initialized = True
            return

        if payload.action_type == DRAG_ACTION_END:
            view.drag_origin_zoom = None

    def _handle_drag_window(self, view: ViewRecord, payload: ViewOperationRequest) -> None:
        if payload.action_type == DRAG_ACTION_START:
            view.drag_origin_window_width = view.window_width
            view.drag_origin_window_center = view.window_center
            view.drag_origin_volume_render_config = None
            return

        if payload.action_type == DRAG_ACTION_MOVE:
            base_ww = view.drag_origin_window_width if view.drag_origin_window_width is not None else view.window_width
            base_wl = view.drag_origin_window_center if view.drag_origin_window_center is not None else view.window_center
            base_ww = max(WINDOW_WIDTH_MIN, float(base_ww or WINDOW_WIDTH_MIN))
            base_wl = float(base_wl or 0.0)
            delta_x = float(payload.x or 0.0)
            delta_y = float(payload.y or 0.0)
            view.window_width = max(WINDOW_WIDTH_MIN, base_ww + delta_x * WINDOW_DRAG_SENSITIVITY)
            view.window_center = base_wl - delta_y * WINDOW_DRAG_SENSITIVITY
            view.is_initialized = True
            return

        if payload.action_type == DRAG_ACTION_END:
            view.drag_origin_window_width = None
            view.drag_origin_window_center = None
            view.drag_origin_volume_render_config = None

    def _get_mpr_group_views(self, view: ViewRecord) -> list[ViewRecord]:
        if view.view_group is None:
            return [view]
        group_views = view_registry.list_view_group(view.view_group.group_id)
        return group_views or [view]

    def _handle_mpr_crosshair(self, view: ViewRecord, payload: ViewOperationRequest) -> bool:
        if payload.x is None or payload.y is None:
            return False
        if not self._is_mpr_view_type(view.view_type):
            return False
        if not view.width or not view.height:
            raise HTTPException(status_code=400, detail="View size has not been set")

        volume = self._get_series_volume(series_registry.get(view.series_id))
        target_viewport = self._resolve_mpr_viewport(view)
        plane_shape = self._get_mpr_plane_shape(volume.shape, target_viewport)
        image_transform = viewport_transformer.build_image_to_canvas_transform(
            image_width=plane_shape[1],
            image_height=plane_shape[0],
            canvas_width=view.width,
            canvas_height=view.height,
            view=view,
        )
        crosshair_info = self._build_mpr_crosshair_info(
            self._build_mpr_crosshair_overlay(view, volume.shape, plane_shape, image_transform)
        )

        if payload.action_type == DRAG_ACTION_START:
            view.mpr_crosshair_drag_active = True
            return False

        if payload.action_type == DRAG_ACTION_END:
            was_dragging = view.mpr_crosshair_drag_active
            view.mpr_crosshair_drag_active = False
            return was_dragging

        if payload.action_type != DRAG_ACTION_MOVE or not view.mpr_crosshair_drag_active:
            return False

        overlay = self._build_mpr_crosshair_overlay(view, volume.shape, plane_shape, image_transform)
        if overlay.image_width <= 0 or overlay.image_height <= 0:
            return False

        canvas_x = overlay.image_left + float(payload.x) * float(overlay.image_width)
        canvas_y = overlay.image_top + float(payload.y) * float(overlay.image_height)
        image_x, image_y = self._canvas_to_image_coordinates(image_transform, canvas_x, canvas_y)
        depth, height, width = volume.shape
        previous_indices = (view.mpr_axial_index, view.mpr_coronal_index, view.mpr_sagittal_index)

        if target_viewport == MPR_VIEWPORT_CORONAL:
            view.mpr_sagittal_index = max(0, min(int(np.floor(image_x)), width - 1))
            view.mpr_axial_index = max(0, min(depth - 1 - int(np.floor(image_y)), depth - 1))
        elif target_viewport == MPR_VIEWPORT_SAGITTAL:
            view.mpr_coronal_index = max(0, min(int(np.floor(image_x)), height - 1))
            view.mpr_axial_index = max(0, min(depth - 1 - int(np.floor(image_y)), depth - 1))
        else:
            view.mpr_sagittal_index = max(0, min(int(np.floor(image_x)), width - 1))
            view.mpr_coronal_index = max(0, min(int(np.floor(image_y)), height - 1))

        current_indices = (view.mpr_axial_index, view.mpr_coronal_index, view.mpr_sagittal_index)
        if current_indices == previous_indices:
            return False

        view.current_index = view.mpr_axial_index
        view.is_initialized = True
        return True

    @staticmethod
    def _build_mpr_crosshair_info(overlay: MprCrosshairOverlay) -> MprCrosshairInfo | None:
        if overlay.center_x is None or overlay.center_y is None:
            return None

        normalized_radius = (
            CROSSHAIR_HIT_RADIUS / float(min(overlay.width, overlay.height))
            if min(overlay.width, overlay.height) > 0
            else 0.0
        )
        return MprCrosshairInfo(
            centerX=(
                float(overlay.center_x) / float(overlay.width)
                if overlay.width > 0
                else 0.0
            ),
            centerY=(
                float(overlay.center_y) / float(overlay.height)
                if overlay.height > 0
                else 0.0
            ),
            hitRadius=normalized_radius,
            horizontalPosition=(
                float(overlay.horizontal_position) / float(overlay.height)
                if overlay.horizontal_position is not None and overlay.height > 0
                else None
            ),
            verticalPosition=(
                float(overlay.vertical_position) / float(overlay.width)
                if overlay.vertical_position is not None and overlay.width > 0
                else None
            ),
        )

    @staticmethod
    def _is_point_near_mpr_crosshair_center(
        crosshair_info: MprCrosshairInfo | None,
        canvas_x: float,
        canvas_y: float,
    ) -> bool:
        if crosshair_info is None:
            return False

        delta_x = canvas_x - crosshair_info.center_x
        delta_y = canvas_y - crosshair_info.center_y
        return delta_x * delta_x + delta_y * delta_y <= crosshair_info.hit_radius * crosshair_info.hit_radius

    @staticmethod
    def _canvas_to_image_coordinates(image_transform, canvas_x: float, canvas_y: float) -> tuple[float, float]:
        inverse = np.linalg.inv(image_transform.matrix)
        point = inverse @ np.array([canvas_x, canvas_y, 1.0], dtype=np.float64)
        return float(point[0]), float(point[1])

    @staticmethod
    def _build_mpr_crosshair_overlay(
        view: ViewRecord,
        volume_shape: tuple[int, int, int],
        plane_shape: tuple[int, int],
        image_transform,
    ) -> MprCrosshairOverlay:
        depth, _, _ = volume_shape
        plane_height, plane_width = plane_shape
        canvas_width = view.width or plane_width
        canvas_height = view.height or plane_height
        target_viewport = ViewerService._resolve_mpr_viewport(view)
        is_active = view.mpr_active_viewport == target_viewport
        line_alpha = 255

        def with_alpha(rgb: tuple[int, int, int], alpha: int) -> tuple[int, int, int, int]:
            return rgb[0], rgb[1], rgb[2], alpha

        axial_color = with_alpha((255, 0, 0), line_alpha)
        coronal_color = with_alpha((0, 255, 0), line_alpha)
        sagittal_color = with_alpha((0, 102, 255), line_alpha)

        def image_to_canvas(image_x: float, image_y: float) -> tuple[float, float]:
            point = image_transform.matrix @ np.array([image_x, image_y, 1.0], dtype=np.float64)
            return float(point[0]), float(point[1])

        top_left_x, top_left_y = image_to_canvas(0.0, 0.0)
        top_right_x, top_right_y = image_to_canvas(float(plane_width), 0.0)
        bottom_left_x, bottom_left_y = image_to_canvas(0.0, float(plane_height))
        bottom_right_x, bottom_right_y = image_to_canvas(float(plane_width), float(plane_height))
        image_left = min(top_left_x, top_right_x, bottom_left_x, bottom_right_x)
        image_top = min(top_left_y, top_right_y, bottom_left_y, bottom_right_y)
        image_right = max(top_left_x, top_right_x, bottom_left_x, bottom_right_x)
        image_bottom = max(top_left_y, top_right_y, bottom_left_y, bottom_right_y)
        image_width = image_right - image_left
        image_height = image_bottom - image_top

        if target_viewport == MPR_VIEWPORT_CORONAL:
            center_x, center_y = image_to_canvas(float(view.mpr_sagittal_index) + 0.5, float(depth - 1 - view.mpr_axial_index) + 0.5)
            _, horizontal_position = image_to_canvas(0.0, float(depth - 1 - view.mpr_axial_index) + 0.5)
            vertical_position, _ = image_to_canvas(float(view.mpr_sagittal_index) + 0.5, 0.0)
            return MprCrosshairOverlay(
                width=canvas_width,
                height=canvas_height,
                image_left=image_left,
                image_top=image_top,
                image_width=image_width,
                image_height=image_height,
                horizontal_position=horizontal_position,
                horizontal_color=axial_color,
                vertical_position=vertical_position,
                vertical_color=sagittal_color,
                center_x=center_x,
                center_y=center_y,
                is_active=is_active,
            )
        if target_viewport == MPR_VIEWPORT_SAGITTAL:
            center_x, center_y = image_to_canvas(float(view.mpr_coronal_index) + 0.5, float(depth - 1 - view.mpr_axial_index) + 0.5)
            _, horizontal_position = image_to_canvas(0.0, float(depth - 1 - view.mpr_axial_index) + 0.5)
            vertical_position, _ = image_to_canvas(float(view.mpr_coronal_index) + 0.5, 0.0)
            return MprCrosshairOverlay(
                width=canvas_width,
                height=canvas_height,
                image_left=image_left,
                image_top=image_top,
                image_width=image_width,
                image_height=image_height,
                horizontal_position=horizontal_position,
                horizontal_color=axial_color,
                vertical_position=vertical_position,
                vertical_color=coronal_color,
                center_x=center_x,
                center_y=center_y,
                is_active=is_active,
            )
        center_x, center_y = image_to_canvas(float(view.mpr_sagittal_index) + 0.5, float(view.mpr_coronal_index) + 0.5)
        _, horizontal_position = image_to_canvas(0.0, float(view.mpr_coronal_index) + 0.5)
        vertical_position, _ = image_to_canvas(float(view.mpr_sagittal_index) + 0.5, 0.0)
        return MprCrosshairOverlay(
            width=canvas_width,
            height=canvas_height,
            image_left=image_left,
            image_top=image_top,
            image_width=image_width,
            image_height=image_height,
            horizontal_position=horizontal_position,
            horizontal_color=coronal_color,
            vertical_position=vertical_position,
            vertical_color=sagittal_color,
            center_x=center_x,
            center_y=center_y,
            is_active=is_active,
        )

    @staticmethod
    def _get_reference_instance_and_cache(series: SeriesRecord) -> tuple[InstanceRecord | None, CachedDicom | None]:
        for instance in series.instances:
            if not instance.sop_instance_uid:
                continue
            return instance, dicom_cache.get(instance.sop_instance_uid, instance.path)
        return None, None

    def _build_series_corner_info_overlay(
        self,
        series: SeriesRecord,
        dataset: Dataset | None,
    ) -> CornerInfoOverlay:
        manufacturer = self._safe_text(getattr(dataset, "Manufacturer", None))
        manufacturer_model = self._safe_text(getattr(dataset, "ManufacturerModelName", None))
        station_name = self._safe_text(getattr(dataset, "StationName", None))
        institution_name = self._safe_text(getattr(dataset, "InstitutionName", None))
        study_description = self._safe_text(getattr(dataset, "StudyDescription", None))
        exam_text = self._first_non_empty(
            study_description,
            self._safe_text(getattr(dataset, "StudyID", None)),
            self._safe_text(series.series_description),
        )
        series_number = self._safe_text(getattr(dataset, "SeriesNumber", None))
        patient_name = self._safe_text(getattr(dataset, "PatientName", None))
        patient_id = self._first_non_empty(self._safe_text(getattr(dataset, "PatientID", None)), self._safe_text(series.patient_id))
        patient_sex = self._safe_text(getattr(dataset, "PatientSex", None))
        patient_age = self._safe_text(getattr(dataset, "PatientAge", None))
        acquisition_date = self._first_non_empty(
            self._format_dicom_date(getattr(dataset, "AcquisitionDate", None)),
            self._format_dicom_date(getattr(dataset, "ContentDate", None)),
            self._format_dicom_date(getattr(dataset, "StudyDate", None)),
        )
        acquisition_time = self._first_non_empty(
            self._format_dicom_time(getattr(dataset, "AcquisitionTime", None)),
            self._format_dicom_time(getattr(dataset, "ContentTime", None)),
            self._format_dicom_time(getattr(dataset, "StudyTime", None)),
        )
        kv = self._format_number(getattr(dataset, "KVP", None), suffix="kV")
        ma = self._format_number(getattr(dataset, "XRayTubeCurrent", None), suffix="mA")
        thickness = self._format_number(getattr(dataset, "SliceThickness", None), suffix="mm")

        vendor_line = self._join_non_empty(" / ", manufacturer, manufacturer_model)
        patient_meta = self._join_non_empty(" ", patient_id, self._join_non_empty(" / ", patient_sex, patient_age))
        technique_parts = [part for part in (kv, ma) if part]

        top_left = tuple(
            line
            for line in (
                vendor_line,
                station_name,
                institution_name,
                exam_text,
                f"Se: {series_number}" if series_number else None,
            )
            if line
        )
        top_right = tuple(
            line
            for line in (
                patient_name,
                patient_meta,
            )
            if line
        )
        bottom_left = tuple(
            line
            for line in (
                " ".join(technique_parts) if technique_parts else None,
                thickness,
                self._join_non_empty(" ", acquisition_date, acquisition_time),
            )
            if line
        )
        return CornerInfoOverlay(
            top_left=top_left,
            top_right=top_right,
            bottom_left=bottom_left,
            bottom_right=tuple(),
        )

    def _build_slice_corner_info_overlay(
        self,
        view: ViewRecord,
        series: SeriesRecord,
        dataset: Dataset | None,
        *,
        current_index: int,
        total_slices: int,
        viewport_label: str,
    ) -> CornerInfoOverlay:
        zoom = self._format_number(view.zoom, precision=2, suffix="x")
        physical_location = self._build_physical_location_label(view, series, dataset, current_index, viewport_label)
        top_left = tuple(
            line
            for line in (
                self._join_non_empty("  ", viewport_label, physical_location),
                f"Im: {current_index + 1}/{total_slices}" if total_slices > 0 else None,
            )
            if line
        )
        top_right = tuple()
        bottom_left = tuple(
            line
            for line in (
                self._build_window_label(view.window_width, view.window_center),
            )
            if line
        )
        bottom_right = tuple(
            line
            for line in (
                f"Zoom:{zoom}" if zoom else None,
                f"X:{int(round(view.offset_x))} Y:{int(round(view.offset_y))}",
            )
            if line
        )
        return CornerInfoOverlay(
            top_left=top_left,
            top_right=top_right,
            bottom_left=bottom_left,
            bottom_right=bottom_right,
        )

    @staticmethod
    def _serialize_corner_info_overlay(overlay: CornerInfoOverlay) -> CornerInfoPayload:
        return CornerInfoPayload(
            topLeft=list(overlay.top_left),
            topRight=list(overlay.top_right),
            bottomLeft=list(overlay.bottom_left),
            bottomRight=list(overlay.bottom_right),
        )

    @staticmethod
    def _serialize_orientation_overlay(overlay: OrientationOverlay | None) -> OrientationInfo:
        return OrientationInfo(
            top=overlay.top if overlay is not None else None,
            right=overlay.right if overlay is not None else None,
            bottom=overlay.bottom if overlay is not None else None,
            left=overlay.left if overlay is not None else None,
            volumeQuaternion=getattr(overlay, "volume_quaternion", None) if overlay is not None else None,
        )

    def _build_3d_orientation_overlay(self, view: ViewRecord) -> OrientationInfo:
        quaternion = self._normalize_quaternion(tuple(float(value) for value in view.rotation_quaternion))
        return OrientationInfo(
            top=None,
            right=None,
            bottom=None,
            left=None,
            volumeQuaternion=quaternion,
        )

    @staticmethod
    def _normalize_quaternion(quaternion: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        vector = np.asarray(quaternion, dtype=np.float64)
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            return (0.0, 0.0, 0.0, 1.0)
        vector /= norm
        return tuple(float(value) for value in vector)

    def _build_physical_location_label(
        self,
        view: ViewRecord,
        series: SeriesRecord,
        dataset: Dataset | None,
        current_index: int,
        viewport_label: str,
    ) -> str | None:
        label = viewport_label.lower()
        if self._is_mpr_view_type(view.view_type):
            transform = self._get_series_patient_transform(series)
            if transform is not None:
                shape = transform["shape"]
                x_index = max(0, min(view.mpr_sagittal_index, int(shape[2]) - 1))
                y_index = max(0, min(view.mpr_coronal_index, int(shape[1]) - 1))
                z_index = max(0, min(view.mpr_axial_index, int(shape[0]) - 1))
                patient_point = (
                    transform["origin"]
                    + transform["axis_vectors"][0] * float(z_index)
                    + transform["axis_vectors"][1] * float(y_index)
                    + transform["axis_vectors"][2] * float(x_index)
                )
                if label.startswith("cor"):
                    return self._format_oriented_mm(float(patient_point[1]), positive="P", negative="A")
                if label.startswith("sag"):
                    return self._format_oriented_mm(float(patient_point[0]), positive="L", negative="R")
                if label.startswith("ax"):
                    return self._format_oriented_mm(float(patient_point[2]), positive="S", negative="I")
                return self._join_non_empty(
                    " ",
                    self._format_oriented_mm(float(patient_point[0]), positive="L", negative="R"),
                    self._format_oriented_mm(float(patient_point[1]), positive="P", negative="A"),
                    self._format_oriented_mm(float(patient_point[2]), positive="S", negative="I"),
                )

        position = self._get_dataset_position(dataset)
        if position is None:
            return None
        if label.startswith("stack") or label.startswith("ax"):
            return self._format_oriented_mm(float(position[2]), positive="S", negative="I")
        if label.startswith("cor"):
            return self._format_oriented_mm(float(position[1]), positive="P", negative="A")
        if label.startswith("sag"):
            return self._format_oriented_mm(float(position[0]), positive="L", negative="R")
        return self._join_non_empty(
            " ",
            self._format_oriented_mm(float(position[0]), positive="L", negative="R"),
            self._format_oriented_mm(float(position[1]), positive="P", negative="A"),
            self._format_oriented_mm(float(position[2]), positive="S", negative="I"),
        )

    def _get_series_patient_transform(self, series: SeriesRecord) -> dict[str, object] | None:
        cached_transform = self._series_patient_transform_cache.get(series.series_id, Ellipsis)
        if cached_transform is not Ellipsis:
            return cached_transform

        slice_entries: list[tuple[np.ndarray, np.ndarray | None, np.ndarray | None, Dataset]] = []
        for instance in series.instances:
            if not instance.sop_instance_uid:
                continue
            cached = dicom_cache.get(instance.sop_instance_uid, instance.path)
            dataset = cached.dataset
            slice_entries.append((
                cached.source_pixels,
                self._get_dataset_orientation(dataset),
                self._get_dataset_position(dataset),
                dataset,
            ))

        if not slice_entries:
            self._series_patient_transform_cache[series.series_id] = None
            return None

        orientation = next((item[1] for item in slice_entries if item[1] is not None), None)
        if orientation is None:
            self._series_patient_transform_cache[series.series_id] = None
            return None

        row_direction = self._normalize_vector(orientation[:3])
        column_direction = self._normalize_vector(orientation[3:6])
        if row_direction is None or column_direction is None:
            self._series_patient_transform_cache[series.series_id] = None
            return None

        slice_direction = self._normalize_vector(np.cross(row_direction, column_direction))
        if slice_direction is None:
            self._series_patient_transform_cache[series.series_id] = None
            return None

        positions = [item[2] for item in slice_entries]
        if any(position is None for position in positions):
            ordered_entries = slice_entries
        else:
            ordered_entries = sorted(
                slice_entries,
                key=lambda item: float(np.dot(item[2], slice_direction)) if item[2] is not None else 0.0,
            )

        first_dataset = ordered_entries[0][3]
        pixel_spacing = getattr(first_dataset, "PixelSpacing", None)
        if pixel_spacing is None or len(pixel_spacing) < 2:
            self._series_patient_transform_cache[series.series_id] = None
            return None

        try:
            row_spacing = abs(float(pixel_spacing[0]))
            col_spacing = abs(float(pixel_spacing[1]))
        except (TypeError, ValueError):
            self._series_patient_transform_cache[series.series_id] = None
            return None

        ordered_positions = [item[2] for item in ordered_entries if item[2] is not None]
        slice_spacing = self._estimate_slice_spacing(ordered_positions, slice_direction, first_dataset)

        raw_axis_vectors = (slice_direction, column_direction, row_direction)
        raw_axis_steps = (slice_spacing, row_spacing, col_spacing)
        raw_lengths = (
            len(ordered_entries),
            int(getattr(first_dataset, "Rows", 0) or 0),
            int(getattr(first_dataset, "Columns", 0) or 0),
        )
        if any(length <= 0 for length in raw_lengths):
            return None

        patient_axes: list[int] = []
        axis_signs: list[int] = []
        for vector in raw_axis_vectors:
            patient_axis = int(np.argmax(np.abs(vector)))
            if patient_axis in patient_axes:
                self._series_patient_transform_cache[series.series_id] = None
                return None
            patient_axes.append(patient_axis)
            axis_signs.append(1 if vector[patient_axis] >= 0 else -1)

        transpose_order = [patient_axes.index(2), patient_axes.index(1), patient_axes.index(0)]
        canonical_signs = [
            axis_signs[patient_axes.index(2)],
            axis_signs[patient_axes.index(1)],
            axis_signs[patient_axes.index(0)],
        ]

        origin = np.asarray(ordered_entries[0][2], dtype=np.float64)
        for canonical_axis, raw_axis in enumerate(transpose_order):
            if canonical_signs[canonical_axis] < 0:
                origin = origin + raw_axis_vectors[raw_axis] * raw_axis_steps[raw_axis] * float(raw_lengths[raw_axis] - 1)

        axis_vectors = tuple(
            raw_axis_vectors[raw_axis] * raw_axis_steps[raw_axis] * float(canonical_signs[canonical_axis])
            for canonical_axis, raw_axis in enumerate(transpose_order)
        )
        shape = tuple(raw_lengths[raw_axis] for raw_axis in transpose_order)
        result = {
            "origin": origin,
            "axis_vectors": axis_vectors,
            "shape": shape,
        }
        self._series_patient_transform_cache[series.series_id] = result
        return result

    @staticmethod
    def _estimate_slice_spacing(
        positions: list[np.ndarray],
        slice_direction: np.ndarray,
        dataset: Dataset | None,
    ) -> float:
        if len(positions) >= 2:
            projected = sorted(float(np.dot(position, slice_direction)) for position in positions)
            diffs = [abs(projected[index] - projected[index - 1]) for index in range(1, len(projected))]
            diffs = [diff for diff in diffs if diff > 1e-6]
            if diffs:
                return float(np.median(diffs))
        slice_thickness = getattr(dataset, "SliceThickness", None) if dataset is not None else None
        try:
            thickness = abs(float(slice_thickness))
            if thickness > 0:
                return thickness
        except (TypeError, ValueError):
            pass
        return 1.0

    @staticmethod
    def _format_oriented_mm(value: float, *, positive: str, negative: str) -> str:
        orientation = positive if float(value) >= 0 else negative
        magnitude = abs(float(value))
        return f"{orientation} {magnitude:.2f}mm"

    def _build_stack_orientation_overlay(self, view: ViewRecord, dataset: Dataset | None) -> OrientationOverlay | None:
        orientation = self._get_dataset_orientation(dataset)
        if orientation is None:
            self._series_patient_transform_cache[series.series_id] = None
            return None

        row_direction = self._normalize_vector(orientation[:3])
        column_direction = self._normalize_vector(orientation[3:6])
        if row_direction is None or column_direction is None:
            self._series_patient_transform_cache[series.series_id] = None
            return None

        x_vector = row_direction * (-1.0 if view.hor_flip else 1.0)
        y_vector = column_direction * (-1.0 if view.ver_flip else 1.0)
        return OrientationOverlay(
            top=self._orientation_text_for_vector(-y_vector),
            right=self._orientation_text_for_vector(x_vector),
            bottom=self._orientation_text_for_vector(y_vector),
            left=self._orientation_text_for_vector(-x_vector),
        )

    def _build_mpr_orientation_overlay(self, view: ViewRecord, viewport_key: str) -> OrientationOverlay:
        if viewport_key == MPR_VIEWPORT_CORONAL:
            x_vector = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
            y_vector = np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
        elif viewport_key == MPR_VIEWPORT_SAGITTAL:
            x_vector = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
            y_vector = np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
        else:
            x_vector = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
            y_vector = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)

        if view.hor_flip:
            x_vector = -x_vector
        if view.ver_flip:
            y_vector = -y_vector

        return OrientationOverlay(
            top=self._orientation_text_for_vector(-y_vector),
            right=self._orientation_text_for_vector(x_vector),
            bottom=self._orientation_text_for_vector(y_vector),
            left=self._orientation_text_for_vector(-x_vector),
        )

    @staticmethod
    def _build_mpr_viewport_label(viewport_key: str) -> str:
        if viewport_key == MPR_VIEWPORT_CORONAL:
            return "CORONAL"
        if viewport_key == MPR_VIEWPORT_SAGITTAL:
            return "SAGITTAL"
        return "AXIAL"

    @staticmethod
    def _build_window_label(window_width: float | None, window_center: float | None) -> str | None:
        ww = ViewerService._format_number(window_width, precision=0)
        wl = ViewerService._format_number(window_center, precision=0)
        if ww is None and wl is None:
            return None
        return f"W: {ww or '-'} L: {wl or '-'}"

    @staticmethod
    def _safe_text(value) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _first_non_empty(*values: str | None) -> str | None:
        for value in values:
            if value:
                return value
        return None

    @staticmethod
    def _join_non_empty(separator: str, *values: str | None) -> str | None:
        parts = [value for value in values if value]
        if not parts:
            return None
        return separator.join(parts)

    @staticmethod
    def _format_number(value, *, precision: int = 1, suffix: str = "") -> str | None:
        if value is None:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            text = str(value).strip()
            return f"{text}{suffix}" if text else None
        if precision <= 0:
            rendered = str(int(round(numeric)))
        else:
            rendered = f"{numeric:.{precision}f}".rstrip("0").rstrip(".")
        return f"{rendered}{suffix}"

    @staticmethod
    def _format_dicom_date(value) -> str | None:
        text = ViewerService._safe_text(value)
        if not text or len(text) != 8 or not text.isdigit():
            return text
        return f"{text[:4]}.{text[4:6]}.{text[6:8]}"

    @staticmethod
    def _format_dicom_time(value) -> str | None:
        text = ViewerService._safe_text(value)
        if not text:
            return None
        digits = ''.join(ch for ch in text if ch.isdigit())
        if len(digits) < 6:
            return text
        return f"{digits[:2]}:{digits[2:4]}:{digits[4:6]}"

    @staticmethod
    def _orientation_text_for_vector(vector: np.ndarray | None) -> str | None:
        if vector is None:
            return None
        axis_map = (
            (0, "L", "R"),
            (1, "P", "A"),
            (2, "S", "I"),
        )
        components: list[tuple[float, str]] = []
        for axis_index, positive_label, negative_label in axis_map:
            component = float(vector[axis_index])
            magnitude = abs(component)
            if magnitude < 0.2:
                continue
            label = positive_label if component >= 0 else negative_label
            components.append((magnitude, label))
        if not components:
            return None
        components.sort(key=lambda item: item[0], reverse=True)
        return ''.join(label for _, label in components[:3])

    @staticmethod
    def _window_array(
        pixels: np.ndarray,
        window_width: float | None,
        window_center: float | None,
        *,
        pixel_min: float | None = None,
        pixel_max: float | None = None,
    ) -> np.ndarray:
        lower_bound = float(np.min(pixels)) if pixel_min is None else float(pixel_min)
        upper_bound = float(np.max(pixels)) if pixel_max is None else float(pixel_max)

        if window_width is not None and window_width > 0 and window_center is not None:
            lower = window_center - window_width / 2.0
            upper = window_center + window_width / 2.0
        else:
            lower = lower_bound
            upper = upper_bound

        scale = upper - lower
        if scale <= 0:
            return np.zeros(pixels.shape, dtype=np.uint8)

        normalized = np.asarray(pixels, dtype=np.float32).copy()
        np.clip(normalized, lower, upper, out=normalized)
        normalized -= lower
        normalized *= 255.0 / scale
        return normalized.astype(np.uint8, copy=False)

    @staticmethod
    def _resolve_mpr_viewport(view: ViewRecord) -> str:
        if view.view_type == "COR":
            return MPR_VIEWPORT_CORONAL
        if view.view_type == "SAG":
            return MPR_VIEWPORT_SAGITTAL
        return MPR_VIEWPORT_AXIAL

    @staticmethod
    def _derive_default_window_width(cached: CachedDicom) -> float:
        return max(WINDOW_WIDTH_MIN, cached.pixel_max - cached.pixel_min)


    @staticmethod
    def _derive_default_window_center(cached: CachedDicom) -> float:
        return (cached.pixel_max + cached.pixel_min) / 2.0

    @staticmethod
    def _reset_drag_state(view: ViewRecord) -> None:
        view.drag_origin_zoom = None
        view.drag_origin_offset_x = None
        view.drag_origin_offset_y = None
        view.drag_origin_window_width = None
        view.drag_origin_window_center = None
        view.drag_origin_rotation_quaternion = None
        view.drag_origin_arcball_x = None
        view.drag_origin_arcball_y = None

    @staticmethod
    def _encode_image(image: Image.Image, image_format: ImageFormat) -> bytes:
        output = io.BytesIO()
        if image_format == "jpeg":
            image.convert("RGB").save(output, format="JPEG", quality=20)
        else:
            image.save(output, format="PNG")
        return output.getvalue()


viewer_service = ViewerService()



















