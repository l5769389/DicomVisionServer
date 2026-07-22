from __future__ import annotations

"""Hover, measurement, annotation, and operation entry points."""

import math

from app.models.measurement import MeasurementMetrics
from app.services.viewer.shared import *  # noqa: F403


ALIGNMENT_MEASUREMENT_TOOL_TYPES = frozenset({"alignment-horizontal", "alignment-vertical"})
ALIGNMENT_SPACING_UNAVAILABLE_LABEL = "DICOM Pixel Spacing unavailable"
ALIGNMENT_UNSUPPORTED_VIEW_LABEL = "Available only in 2D CT views"


class ViewerInteractionMixin:
    def handle_view_operation(
        self,
        payload: ViewOperationRequest,
        workspace_id: str | None = None,
    ) -> OperationRenderOutcome:
        return handle_view_operation(self, payload, workspace_id=workspace_id)

    def handle_view_hover(
        self,
        payload: ViewHoverRequest,
        workspace_id: str | None = None,
    ) -> ViewHoverResponse:
        view = compat.view_registry.get(payload.view_id, workspace_id=workspace_id)
        row, col, pixel_value, value_label, value_unit = self._resolve_hover_sample_for_workspace(
            view,
            payload.x,
            payload.y,
            workspace_id=workspace_id,
        )
        return ViewHoverResponse(
            viewId=view.view_id,
            row=row,
            col=col,
            pixelValue=pixel_value,
            valueLabel=value_label,
            valueUnit=value_unit,
            displayText=self._format_hover_display_text(row, col, pixel_value, value_unit),
        )

    @staticmethod
    def _format_hover_display_text(
        row: int,
        col: int,
        pixel_value: float | None,
        value_unit: str | None,
    ) -> str | None:
        """Build the authoritative cursor readout sent to clients."""

        if row < 1 or col < 1 or pixel_value is None or not np.isfinite(pixel_value):
            return None
        if str(value_unit or "").strip().upper() == "HU":
            value_text = str(int(round(float(pixel_value))))
        else:
            value_text = f"{float(pixel_value):.3f}".rstrip("0").rstrip(".")
        unit_text = str(value_unit or "").strip()
        return f"X:{col:>4d} Y:{row:>4d} {value_text:>6}{f' {unit_text}' if unit_text else ''}"

    def build_mpr_state_update_payload(
        self,
        view_id: str,
        *,
        workspace_id: str | None = None,
        mpr_revision: int | None = None,
    ) -> dict[str, object] | None:
        return self.build_mpr_state_update_payloads(
            (view_id,),
            workspace_id=workspace_id,
            mpr_revision=mpr_revision,
        ).get(view_id)

    def build_mpr_state_update_payloads(
        self,
        view_ids: tuple[str, ...],
        *,
        workspace_id: str | None = None,
        mpr_revision: int | None = None,
    ) -> dict[str, dict[str, object]]:
        grouped_views: OrderedDict[tuple[str, str], list[ViewRecord]] = OrderedDict()
        for view_id in dict.fromkeys(view_ids):
            view = compat.view_registry.get(view_id, workspace_id=workspace_id)
            if not self._is_mpr_view_type(view.view_type):
                continue
            group_key = view.view_group.group_id if view.view_group is not None else view.view_id
            grouped_views.setdefault((str(group_key), view.series_id), []).append(view)

        payloads: dict[str, dict[str, object]] = {}
        for views in grouped_views.values():
            if not views:
                continue
            source_view = views[0]
            series = compat.series_registry.get(source_view.series_id, workspace_id=workspace_id)
            volume = self._get_series_volume(series)
            for view in views:
                ensure_view_size(view)
                if not view.is_initialized:
                    self._initialize_mpr_viewport(view)
                    view.is_initialized = True
            pose_context = self._build_mpr_pose_context(source_view, volume.shape, series=series)
            for view in views:
                payload = self._build_mpr_state_update_payload_from_context(
                    view,
                    volume_shape=volume.shape,
                    pose_context=pose_context,
                    mpr_revision=mpr_revision,
                )
                if payload is not None:
                    payloads[view.view_id] = payload
        return payloads

    def _build_mpr_state_update_payload_from_context(
        self,
        view: ViewRecord,
        *,
        volume_shape: tuple[int, int, int],
        pose_context: MprPoseContext,
        mpr_revision: int | None = None,
    ) -> dict[str, object] | None:
        if not self._is_mpr_view_type(view.view_type):
            return None

        ensure_view_size(view)

        target_viewport = self._resolve_mpr_viewport(view)
        target_plane_pose = pose_context.poses[target_viewport]
        pixel_aspect_x, pixel_aspect_y = self._get_mpr_display_aspect_xy_from_pose(target_plane_pose)
        full_plane_height, full_plane_width = target_plane_pose.output_shape
        render_plan = self._build_render_plan_for_shape(
            view,
            full_plane_height,
            full_plane_width,
            pixel_aspect_x=pixel_aspect_x,
            pixel_aspect_y=pixel_aspect_y,
        )
        metadata_image_transform = compat.viewport_transformer.build_image_to_canvas_transform(
            image_width=full_plane_width,
            image_height=full_plane_height,
            canvas_width=render_plan.render_view.width or 0,
            canvas_height=render_plan.render_view.height or 0,
            view=render_plan.render_view,
            pixel_aspect_x=pixel_aspect_x,
            pixel_aspect_y=pixel_aspect_y,
        )
        current, total = self._get_mpr_viewport_index_info(
            view,
            volume_shape,
            target_viewport,
            cursor=pose_context.cursor,
            geometry=pose_context.geometry,
        )
        if target_viewport == MPR_VIEWPORT_AXIAL:
            view.current_index = current
        mpr_crosshair_overlay = self._build_mpr_crosshair_overlay(
            render_plan.render_view,
            volume_shape,
            target_plane_pose.output_shape,
            metadata_image_transform,
            pose_context=pose_context,
        )
        frame_payload = self._build_mpr_frame_payload(pose_context.cursor, pose_context.geometry)
        cursor_payload = self._build_mpr_cursor_payload(pose_context.cursor)
        plane_payload = self._build_mpr_plane_payload(
            view,
            target_viewport,
            plane_pose=target_plane_pose,
            geometry=pose_context.geometry,
            image_transform=metadata_image_transform,
        )
        crosshair_payload = self._build_mpr_crosshair_info(mpr_crosshair_overlay)
        payload: dict[str, object] = {
            "viewId": view.view_id,
            "slice_info": SliceInfo(current=current, total=total).model_dump(by_alias=True),
            "mprRevision": mpr_revision if mpr_revision is not None else self._get_mpr_revision(view.view_group),
            "mprCrosshairMode": self._get_mpr_crosshair_mode(view.view_group),
        }
        if frame_payload is not None:
            payload["mprFrame"] = frame_payload.model_dump(by_alias=True)
        if cursor_payload is not None:
            payload["mprCursor"] = cursor_payload.model_dump(by_alias=True)
        if plane_payload is not None:
            payload["mprPlane"] = plane_payload.model_dump(by_alias=True)
        if crosshair_payload is not None:
            payload["mpr_crosshair"] = crosshair_payload.model_dump(by_alias=True)
        return payload

    def get_series_corner_info(
        self,
        payload: CornerInfoRequest,
        workspace_id: str | None = None,
    ) -> CornerInfoResponse:
        series = compat.series_registry.get(payload.series_id, workspace_id=workspace_id)
        _, reference_cached = self._get_reference_instance_and_cache(series)
        overlay = self._build_series_corner_info_overlay(
            series,
            reference_cached.dataset if reference_cached is not None else None,
        )
        return CornerInfoResponse(cornerInfo=self._serialize_corner_info_overlay(overlay))

    def analyze_mtf(
        self,
        payload: ViewMtfAnalyzeRequest,
        workspace_id: str | None = None,
    ) -> ViewMtfAnalyzeResponse:
        compat.view_registry.get(payload.view_id, workspace_id=workspace_id)
        return self._mtf_analysis_service.analyze(payload)

    def analyze_qa_water(
        self,
        payload: ViewQaWaterAnalyzeRequest,
        workspace_id: str | None = None,
    ) -> ViewQaWaterAnalyzeResponse:
        compat.view_registry.get(payload.view_id, workspace_id=workspace_id)
        return self._water_phantom_qa_service.analyze(payload)

    def _resolve_hover_row_col(self, view: ViewRecord, normalized_x: float, normalized_y: float) -> tuple[int, int]:
        return self._resolve_hover_row_col_for_workspace(view, normalized_x, normalized_y)

    def _resolve_hover_row_col_for_workspace(
        self,
        view: ViewRecord,
        normalized_x: float,
        normalized_y: float,
        workspace_id: str | None = None,
    ) -> tuple[int, int]:
        row, col, _, _, _ = self._resolve_hover_sample_for_workspace(
            view,
            normalized_x,
            normalized_y,
            workspace_id=workspace_id,
        )
        return (row, col)

    def _resolve_hover_sample_for_workspace(
        self,
        view: ViewRecord,
        normalized_x: float,
        normalized_y: float,
        workspace_id: str | None = None,
    ) -> tuple[int, int, float | None, str | None, str | None]:
        if not view.width or not view.height or self._is_3d_view_type(view.view_type):
            return (0, 0, None, None, None)

        source_pixels, dataset, modality, value_label, value_unit = self._get_hover_source_context(
            view,
            workspace_id=workspace_id,
        )
        if source_pixels.ndim < 2 or source_pixels.shape[0] <= 0 or source_pixels.shape[1] <= 0:
            return (0, 0, None, value_label, value_unit)
        image_width, image_height, image_transform, canvas_width, canvas_height = self._build_hover_mapping_context(
            view,
            workspace_id=workspace_id,
            source_dimensions=(int(source_pixels.shape[1]), int(source_pixels.shape[0])),
        )
        row, col = map_normalized_canvas_to_image_row_col(
            normalized_x,
            normalized_y,
            image_width=image_width,
            image_height=image_height,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            image_transform=image_transform,
        )
        if row < 1 or col < 1:
            return (0, 0, None, value_label, value_unit)

        sample = np.asarray(source_pixels)[row - 1, col - 1]
        if np.ndim(sample) != 0:
            return (row, col, None, value_label, value_unit)
        pixel_value = float(sample)
        if not np.isfinite(pixel_value):
            return (row, col, None, value_label, value_unit)

        # source_pixels are presentation-inverted for MONOCHROME1. Recover the
        # modality value used for quantitative cursor readout.
        if str(getattr(dataset, "PhotometricInterpretation", "") or "").upper() == "MONOCHROME1":
            pixel_value = -pixel_value
        if modality == "CT":
            value_label = "CT"
            value_unit = "HU"
        return (row, col, pixel_value, value_label, value_unit)

    def _get_hover_source_context(
        self,
        view: ViewRecord,
        workspace_id: str | None = None,
    ) -> tuple[np.ndarray, Any | None, str, str, str | None]:
        series = compat.series_registry.get(view.series_id, workspace_id=workspace_id)
        modality = str(series.modality or "").strip().upper()
        instance_index = max(0, min(int(view.current_index), len(series.instances) - 1)) if series.instances else 0
        instance = series.instances[instance_index] if series.instances else None
        cached = (
            compat.dicom_cache.get(instance.sop_instance_uid, instance.path)
            if instance is not None and instance.sop_instance_uid
            else None
        )
        dataset = cached.dataset if cached is not None else None

        if self._is_mpr_view_type(view.view_type):
            volume = self._get_series_volume(series)
            if not view.is_initialized:
                self._initialize_mpr_viewport(view)
                view.is_initialized = True
            target_viewport = self._resolve_mpr_viewport(view)
            source_pixels, _, _ = self._extract_mpr_plane(view, volume, target_viewport)
        elif self._is_pet_series(series):
            pet_display = self._build_fusion_pet_display_volume(series, self._get_series_volume(series), view.pet_unit)
            index = max(0, min(int(view.current_index), pet_display.volume.shape[0] - 1))
            source_pixels = np.asarray(pet_display.volume[index], dtype=np.float32)
            return source_pixels, dataset, modality, "PET", pet_display.unit
        elif cached is not None:
            source_pixels = cached.source_pixels
        else:
            source_pixels = np.empty((0, 0), dtype=np.float32)

        if modality == "CT":
            return np.asarray(source_pixels), dataset, modality, "CT", "HU"
        unit = str(getattr(dataset, "Units", "") or getattr(dataset, "RescaleType", "") or "").strip() or None
        return np.asarray(source_pixels), dataset, modality, modality or "Value", unit

    def _build_hover_mapping_context(
        self,
        view: ViewRecord,
        workspace_id: str | None = None,
        source_dimensions: tuple[int, int] | None = None,
    ) -> tuple[int, int, Any, int, int]:
        """Prepare the source-image dimensions and inverse transform used for hover lookup."""

        image_width, image_height = source_dimensions or self._get_hover_source_dimensions(
            view,
            workspace_id=workspace_id,
        )
        pixel_aspect_x = 1.0
        pixel_aspect_y = 1.0
        if self._is_mpr_view_type(view.view_type):
            series = compat.series_registry.get(view.series_id, workspace_id=workspace_id)
            target_viewport = self._resolve_mpr_viewport(view)
            volume = self._get_series_volume(series)
            pose_context = self._build_mpr_pose_context(view, volume.shape, series=series)
            pixel_aspect_x, pixel_aspect_y = self._get_mpr_display_aspect_xy_from_pose(
                pose_context.poses[target_viewport]
            )
        render_plan = self._build_render_plan_for_shape(
            view,
            image_height,
            image_width,
            pixel_aspect_x=pixel_aspect_x,
            pixel_aspect_y=pixel_aspect_y,
        )
        image_transform = compat.viewport_transformer.build_image_to_canvas_transform(
            image_width=image_width,
            image_height=image_height,
            canvas_width=render_plan.render_view.width or 0,
            canvas_height=render_plan.render_view.height or 0,
            view=render_plan.render_view,
            pixel_aspect_x=pixel_aspect_x,
            pixel_aspect_y=pixel_aspect_y,
        )
        return (
            image_width,
            image_height,
            image_transform,
            render_plan.render_view.width or 0,
            render_plan.render_view.height or 0,
        )

    def _get_hover_source_dimensions(self, view: ViewRecord, workspace_id: str | None = None) -> tuple[int, int]:
        if self._is_mpr_view_type(view.view_type):
            series = compat.series_registry.get(view.series_id, workspace_id=workspace_id)
            volume = self._get_series_volume(series)
            if not view.is_initialized:
                self._initialize_mpr_viewport(view)
                view.is_initialized = True
            target_viewport = self._resolve_mpr_viewport(view)
            plane_pixels, _, _ = self._extract_mpr_plane(view, volume, target_viewport)
            return (int(plane_pixels.shape[1]), int(plane_pixels.shape[0]))

        series = compat.series_registry.get(view.series_id, workspace_id=workspace_id)
        instance = series.instances[view.current_index]
        if not instance.sop_instance_uid:
            return (0, 0)
        cached = compat.dicom_cache.get(instance.sop_instance_uid, instance.path)
        return (int(cached.source_pixels.shape[1]), int(cached.source_pixels.shape[0]))

    def _resolve_normalized_point_to_image_point(
        self,
        view: ViewRecord,
        normalized_x: float,
        normalized_y: float,
    ) -> MeasurementPoint:
        image_width, image_height, image_transform, canvas_width, canvas_height = self._build_hover_mapping_context(view)
        if image_width <= 0 or image_height <= 0 or canvas_width <= 0 or canvas_height <= 0:
            raise HTTPException(status_code=400, detail="View is not ready for measurement")

        x = max(0.0, min(1.0, float(normalized_x)))
        y = max(0.0, min(1.0, float(normalized_y)))
        max_canvas_x = max(float(canvas_width) - 1e-6, 0.0)
        max_canvas_y = max(float(canvas_height) - 1e-6, 0.0)
        canvas_x = min(max(x * float(canvas_width), 0.0), max_canvas_x)
        canvas_y = min(max(y * float(canvas_height), 0.0), max_canvas_y)

        affine_matrix, offset = image_transform.inverse_components()
        source_point = affine_matrix @ np.asarray([canvas_x, canvas_y], dtype=np.float64) + offset
        return MeasurementPoint(x=float(source_point[0]), y=float(source_point[1]))

    def _resolve_measurement_source_context(
        self,
        view: ViewRecord,
    ) -> tuple[np.ndarray, tuple[float, float] | None, MeasurementSliceContext]:
        if self._is_mpr_view_type(view.view_type):
            series = compat.series_registry.get(view.series_id)
            volume = self._get_series_volume(series)
            target_viewport = self._resolve_mpr_viewport(view)
            plane_pixels, current_index, _ = self._extract_mpr_plane(view, volume, target_viewport)
            pose_context = self._build_mpr_pose_context(view, volume.shape, series=series)
            return (
                plane_pixels,
                self._get_mpr_spacing_xy_from_pose(pose_context.poses[target_viewport]),
                MeasurementSliceContext(kind="mpr", slice_index=current_index),
            )

        series = compat.series_registry.get(view.series_id)
        instance = series.instances[view.current_index]
        if not instance.sop_instance_uid:
            raise HTTPException(status_code=400, detail="DICOM instance does not contain SOPInstanceUID")
        cached = compat.dicom_cache.get(instance.sop_instance_uid, instance.path)
        return (
            cached.source_pixels,
            self._get_stack_spacing_xy(cached.dataset),
            MeasurementSliceContext(kind="stack", slice_index=view.current_index, sop_instance_uid=instance.sop_instance_uid),
        )

    def _resolve_mpr_measurement_plane_pose(self, view: ViewRecord) -> PlanePose:
        series = compat.series_registry.get(view.series_id)
        volume = self._get_series_volume(series)
        target_viewport = self._resolve_mpr_viewport(view)
        return self._build_mpr_pose_context(view, volume.shape, series=series).poses[target_viewport]

    def _resolve_mpr_measurement_model_transform(
        self,
        view: ViewRecord,
        *,
        fallback_pivot_world: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        group = view.view_group
        if group is None:
            return np.eye(3, dtype=np.float64), np.asarray(fallback_pivot_world, dtype=np.float64)
        return (
            self._get_mpr_model_rotation_matrix(group),
            self._get_mpr_model_rotation_pivot_world(group, fallback_pivot_world),
        )

    def _capture_mpr_measurement_world_points(
        self,
        view: ViewRecord,
        image_points: tuple[MeasurementPoint, ...],
    ) -> tuple[tuple[float, float, float], ...]:
        if not self._is_mpr_view_type(view.view_type):
            return ()

        plane_pose = self._resolve_mpr_measurement_plane_pose(view)
        rotation, pivot = self._resolve_mpr_measurement_model_transform(
            view,
            fallback_pivot_world=plane_pose.cursor_center_world,
        )
        inverse_rotation = rotation.T
        world_points: list[tuple[float, float, float]] = []
        for point in image_points:
            displayed_world = plane_image_point_to_world(plane_pose, (point.x, point.y))
            source_world = pivot + inverse_rotation @ (displayed_world - pivot)
            world_points.append(tuple(float(value) for value in source_world))
        return tuple(world_points)

    def _project_mpr_measurement_to_current_plane(
        self,
        view: ViewRecord,
        measurement: MeasurementRecord,
    ) -> MeasurementRecord:
        if not measurement.world_points:
            return measurement

        plane_pose = self._resolve_mpr_measurement_plane_pose(view)
        rotation, pivot = self._resolve_mpr_measurement_model_transform(
            view,
            fallback_pivot_world=plane_pose.cursor_center_world,
        )
        projected_points: list[MeasurementPoint] = []
        for source_world_value in measurement.world_points:
            source_world = np.asarray(source_world_value, dtype=np.float64)
            displayed_world = pivot + rotation @ (source_world - pivot)
            x, y = world_point_to_plane_image(plane_pose, displayed_world)
            projected_points.append(MeasurementPoint(x=x, y=y))

        next_points = tuple(projected_points)
        source_pixels, spacing_xy, current_context = self._resolve_measurement_source_context(view)
        metrics, label_lines = build_measurement_metrics(
            measurement.tool_type,
            next_points,
            source_pixels,
            spacing_xy,
        )
        next_context = (
            MeasurementSliceContext(
                kind=measurement.slice_context.kind,
                slice_index=current_context.slice_index,
                sop_instance_uid=measurement.slice_context.sop_instance_uid,
            )
            if measurement.scope == "series"
            else measurement.slice_context
        )
        return replace(
            measurement,
            points=next_points,
            slice_context=next_context,
            metrics=metrics,
            label_anchor=next_points[1],
            label_lines=label_lines,
        )

    @staticmethod
    def _resolve_measurement_tool_type(payload: ViewOperationRequest) -> str | None:
        tool_type = str(payload.sub_op_type or "").strip().lower()
        return tool_type if tool_type in MEASUREMENT_TOOL_TYPES else None

    def _resolve_measurement_image_points(
        self,
        view: ViewRecord,
        payload: ViewOperationRequest,
    ) -> tuple[MeasurementPoint, ...]:
        return tuple(
            self._resolve_normalized_point_to_image_point(view, point.x, point.y)
            for point in (payload.points or [])
        )

    @staticmethod
    def _is_empty_measurement(tool_type: str, points: tuple[MeasurementPoint, ...]) -> bool:
        if tool_type in {"curve", "freeform"}:
            return len(points) < get_measurement_point_requirement(tool_type).min_points
        if tool_type == "angle" or len(points) < 2:
            return False
        start, end = points[:2]
        return abs(end.x - start.x) < 1e-3 and abs(end.y - start.y) < 1e-3

    @staticmethod
    def _serialize_measurement_metrics(metrics) -> dict[str, float | str | None]:
        return {
            "length": metrics.length,
            "width": metrics.width,
            "height": metrics.height,
            "area": metrics.area,
            "angleDegrees": metrics.angle_degrees,
            "mean": metrics.mean,
            "sd": metrics.standard_deviation,
            "min": metrics.minimum,
            "max": metrics.maximum,
            "unit": metrics.unit,
            "areaUnit": metrics.area_unit,
        }

    def _build_measurement_preview_payload(
        self,
        *,
        view: ViewRecord,
        viewport_key: str,
        tool_type: str,
        slice_index: int,
        label_lines: tuple[str, ...] | list[str] = (),
        metrics=None,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "viewId": view.view_id,
            "viewportKey": viewport_key,
            "toolType": tool_type,
            "labelLines": list(label_lines),
            "sliceIndex": slice_index,
        }
        if metrics is not None:
            payload["metrics"] = self._serialize_measurement_metrics(metrics)
        return payload

    def _build_measurement_preview(self, view: ViewRecord, payload: ViewOperationRequest) -> dict[str, object] | None:
        tool_type = self._resolve_measurement_tool_type(payload)
        if tool_type is None or not payload.points:
            return None

        viewport_key = payload.viewport_key or self._resolve_measurement_viewport_key(view)
        if self._is_alignment_measurement(tool_type) and not self._is_supported_alignment_view(view):
            return self._build_measurement_preview_payload(
                view=view,
                viewport_key=viewport_key,
                tool_type=tool_type,
                slice_index=view.current_index,
                label_lines=(ALIGNMENT_UNSUPPORTED_VIEW_LABEL,),
            )

        image_points = self._resolve_measurement_image_points(view, payload)
        source_pixels, spacing_xy, slice_context = self._resolve_measurement_source_context(view)

        if tool_type == "angle" and len(image_points) < get_measurement_point_requirement(tool_type).min_points:
            return self._build_measurement_preview_payload(
                view=view,
                viewport_key=viewport_key,
                tool_type=tool_type,
                slice_index=slice_context.slice_index,
            )

        if not has_required_measurement_points(tool_type, len(image_points)):
            return None

        if self._is_empty_measurement(tool_type, image_points):
            return self._build_measurement_preview_payload(
                view=view,
                viewport_key=viewport_key,
                tool_type=tool_type,
                slice_index=slice_context.slice_index,
            )

        if self._is_alignment_measurement(tool_type) and not self._has_alignment_physical_context(view, spacing_xy):
            return self._build_measurement_preview_payload(
                view=view,
                viewport_key=viewport_key,
                tool_type=tool_type,
                slice_index=slice_context.slice_index,
                label_lines=(ALIGNMENT_SPACING_UNAVAILABLE_LABEL,),
            )

        metrics, label_lines = build_measurement_metrics(tool_type, image_points, source_pixels, spacing_xy)
        return self._build_measurement_preview_payload(
            view=view,
            viewport_key=viewport_key,
            tool_type=tool_type,
            slice_index=slice_context.slice_index,
            label_lines=label_lines,
            metrics=metrics,
        )

    def _handle_measurement(self, view: ViewRecord, payload: ViewOperationRequest) -> bool:
        tool_type = self._resolve_measurement_tool_type(payload)
        if tool_type is None:
            raise HTTPException(status_code=400, detail="Unsupported measurement tool type")
        if not payload.points:
            raise HTTPException(status_code=400, detail="Measurement points are required")

        if self._is_alignment_measurement(tool_type) and not self._is_supported_alignment_view(view):
            raise HTTPException(status_code=400, detail=ALIGNMENT_UNSUPPORTED_VIEW_LABEL)

        if not has_required_measurement_points(tool_type, len(payload.points)):
            return False

        image_points = self._resolve_measurement_image_points(view, payload)

        if self._is_empty_measurement(tool_type, image_points):
            return False

        source_pixels, spacing_xy, slice_context = self._resolve_measurement_source_context(view)
        slice_context = self._with_operation_slice_index(slice_context, payload.slice_index)
        if self._is_alignment_measurement(tool_type) and not self._has_alignment_physical_context(view, spacing_xy):
            metrics, label_lines = self._build_alignment_spacing_unavailable_metrics()
        else:
            metrics, label_lines = build_measurement_metrics(tool_type, image_points, source_pixels, spacing_xy)
        world_points = self._capture_mpr_measurement_world_points(view, image_points)

        label_anchor = image_points[1] if tool_type != "angle" else image_points[1]
        measurement_id = str(payload.measurement_id or "").strip() or str(uuid4())
        next_measurement = MeasurementRecord(
            measurement_id=measurement_id,
            tool_type=tool_type,
            points=image_points,
            slice_context=slice_context,
            metrics=metrics,
            label_anchor=label_anchor,
            world_points=world_points,
            label_lines=label_lines,
            scope=self._normalize_drawing_scope(payload.scope),
        )
        existing_index = next(
            (index for index, measurement in enumerate(view.measurements) if measurement.measurement_id == measurement_id),
            None,
        )
        if existing_index is None:
            view.measurements.append(next_measurement)
        else:
            view.measurements[existing_index] = next_measurement
        view.is_initialized = True
        return True

    @staticmethod
    def _delete_measurement(view: ViewRecord, measurement_id: str | None) -> bool:
        target_measurement_id = str(measurement_id or "").strip()
        if not target_measurement_id:
            return False

        existing_count = len(view.measurements)
        if not existing_count:
            return False

        view.measurements = [
            measurement for measurement in view.measurements if measurement.measurement_id != target_measurement_id
        ]
        if len(view.measurements) == existing_count:
            return False

        view.is_initialized = True
        return True

    @staticmethod
    def _clear_measurements(view: ViewRecord) -> bool:
        if not view.measurements:
            return False

        view.measurements = []
        view.is_initialized = True
        return True

    @staticmethod
    def _resolve_annotation_tool_type(payload: ViewOperationRequest) -> str | None:
        tool_type = str(payload.tool_type or payload.sub_op_type or "").strip().lower()
        return tool_type if tool_type in {"arrow"} else None

    def _resolve_annotation_image_points(
        self,
        view: ViewRecord,
        payload: ViewOperationRequest,
    ) -> tuple[MeasurementPoint, ...]:
        return tuple(
            self._resolve_normalized_point_to_image_point(view, point.x, point.y)
            for point in (payload.points or [])
        )

    @staticmethod
    def _is_empty_annotation(points: tuple[MeasurementPoint, ...]) -> bool:
        if len(points) < 2:
            return True
        start, end = points[:2]
        return abs(end.x - start.x) < 1e-3 and abs(end.y - start.y) < 1e-3

    @staticmethod
    def _normalize_annotation_size(value: str | None) -> str:
        size = str(value or "").strip().lower()
        return size if size in {"sm", "md", "lg"} else "md"

    @staticmethod
    def _normalize_annotation_color(value: str | None) -> str:
        color = str(value or "").strip()
        return color or "#ffd166"

    @staticmethod
    def _normalize_drawing_scope(value: str | None) -> DrawingScope:
        return "series" if str(value or "").strip().lower() == "series" else "image"

    @staticmethod
    def _with_operation_slice_index(
        slice_context: MeasurementSliceContext,
        slice_index: int | None,
    ) -> MeasurementSliceContext:
        if slice_index is None:
            return slice_context
        return MeasurementSliceContext(
            kind=slice_context.kind,
            slice_index=max(0, int(slice_index)),
            sop_instance_uid=slice_context.sop_instance_uid,
        )

    def _handle_annotation(self, view: ViewRecord, payload: ViewOperationRequest) -> bool:
        tool_type = self._resolve_annotation_tool_type(payload)
        if tool_type is None:
            raise HTTPException(status_code=400, detail="Unsupported annotation tool type")
        if not payload.points:
            raise HTTPException(status_code=400, detail="Annotation points are required")

        image_points = self._resolve_annotation_image_points(view, payload)
        if self._is_empty_annotation(image_points):
            return False

        _, _, slice_context = self._resolve_measurement_source_context(view)
        slice_context = self._with_operation_slice_index(slice_context, payload.slice_index)
        annotation_id = str(payload.annotation_id or payload.measurement_id or "").strip() or str(uuid4())
        next_annotation = AnnotationRecord(
            annotation_id=annotation_id,
            tool_type=tool_type,
            points=image_points,
            slice_context=slice_context,
            text=str(payload.text or ""),
            color=self._normalize_annotation_color(payload.color),
            size=self._normalize_annotation_size(payload.size),
            scope=self._normalize_drawing_scope(payload.scope),
        )
        existing_index = next(
            (index for index, annotation in enumerate(view.annotations) if annotation.annotation_id == annotation_id),
            None,
        )
        if existing_index is None:
            view.annotations.append(next_annotation)
        else:
            view.annotations[existing_index] = next_annotation
        view.is_initialized = True
        return True

    @staticmethod
    def _delete_annotation(view: ViewRecord, annotation_id: str | None) -> bool:
        target_annotation_id = str(annotation_id or "").strip()
        if not target_annotation_id or not view.annotations:
            return False

        existing_count = len(view.annotations)
        view.annotations = [
            annotation for annotation in view.annotations if annotation.annotation_id != target_annotation_id
        ]
        if len(view.annotations) == existing_count:
            return False

        view.is_initialized = True
        return True

    @staticmethod
    def _clear_annotations(view: ViewRecord) -> bool:
        if not view.annotations:
            return False

        view.annotations = []
        view.is_initialized = True
        return True

    def _build_visible_measurements(self, view: ViewRecord) -> tuple[MeasurementRecord, ...]:
        if not view.measurements:
            return ()

        current_slice = self._resolve_current_measurement_slice_index(view)
        visible: list[MeasurementRecord] = []
        for measurement in view.measurements:
            if measurement.slice_context.kind == "stack":
                if not self._is_mpr_view_type(view.view_type) and (
                    measurement.scope == "series" or measurement.slice_context.slice_index == current_slice
                ):
                    visible.append(self._with_current_series_measurement_metrics(view, measurement))
                continue
            if self._is_mpr_view_type(view.view_type) and (
                measurement.scope == "series" or measurement.slice_context.slice_index == current_slice
            ):
                try:
                    projected = self._project_mpr_measurement_to_current_plane(view, measurement)
                except Exception:
                    logger.debug("Failed to reproject MPR measurement", exc_info=True)
                    projected = measurement
                visible.append(self._with_current_series_measurement_metrics(view, projected))
        return tuple(visible)

    def _with_current_series_measurement_metrics(
        self,
        view: ViewRecord,
        measurement: MeasurementRecord,
    ) -> MeasurementRecord:
        if measurement.scope != "series":
            return measurement

        try:
            source_pixels, spacing_xy, current_context = self._resolve_measurement_source_context(view)
            if self._is_alignment_measurement(measurement.tool_type) and not self._has_alignment_physical_context(view, spacing_xy):
                metrics, label_lines = self._build_alignment_spacing_unavailable_metrics()
            else:
                metrics, label_lines = build_measurement_metrics(
                    measurement.tool_type,
                    measurement.points,
                    source_pixels,
                    spacing_xy,
                )
        except Exception:
            logger.debug(
                "Failed to refresh series-scope measurement metrics for current slice",
                exc_info=True,
            )
            return measurement

        return replace(
            measurement,
            metrics=metrics,
            label_lines=label_lines,
            slice_context=MeasurementSliceContext(
                kind=measurement.slice_context.kind,
                slice_index=current_context.slice_index,
                sop_instance_uid=(
                    current_context.sop_instance_uid
                    if measurement.slice_context.kind == "stack"
                    else measurement.slice_context.sop_instance_uid
                ),
            ),
        )

    def _build_visible_annotations(self, view: ViewRecord) -> tuple[AnnotationRecord, ...]:
        if not view.annotations:
            return ()

        current_slice = self._resolve_current_measurement_slice_index(view)
        visible: list[AnnotationRecord] = []
        for annotation in view.annotations:
            if annotation.slice_context.kind == "stack":
                if not self._is_mpr_view_type(view.view_type) and (
                    annotation.scope == "series" or annotation.slice_context.slice_index == current_slice
                ):
                    visible.append(annotation)
                continue
            if self._is_mpr_view_type(view.view_type) and (
                annotation.scope == "series" or annotation.slice_context.slice_index == current_slice
            ):
                visible.append(annotation)
        return tuple(visible)

    @staticmethod
    def _serialize_measurements(
        measurements: tuple[Any, ...],
        *,
        image_transform: Any,
        canvas_width: int,
        canvas_height: int,
    ) -> list[MeasurementOverlayPayload]:
        if canvas_width <= 0 or canvas_height <= 0:
            return []

        matrix = image_transform.matrix
        width = max(float(canvas_width), 1.0)
        height = max(float(canvas_height), 1.0)

        def serialize_point(point: MeasurementPoint) -> dict[str, float]:
            projected = matrix @ np.asarray([point.x, point.y, 1.0], dtype=np.float64)
            return {
                "x": float(projected[0]) / width,
                "y": float(projected[1]) / height,
            }

        return [
            MeasurementOverlayPayload(
                measurementId=measurement.measurement_id,
                toolType=measurement.tool_type,
                points=[serialize_point(point) for point in measurement.points],
                labelLines=list(measurement.label_lines),
                scope=getattr(measurement, "scope", "image"),
                sliceIndex=getattr(measurement.slice_context, "slice_index", None),
            )
            for measurement in measurements
        ]

    def _build_visible_presentation_measurements(
        self,
        series: SeriesRecord,
        instance: InstanceRecord,
    ) -> tuple[PresentationMeasurementRecord, ...]:
        if not instance.sop_instance_uid:
            return ()

        presentation_states = series.presentation_states_by_sop_uid.get(str(instance.sop_instance_uid), [])
        return tuple(
            measurement
            for presentation_state in presentation_states
            for measurement in presentation_state.measurements
        )

    def _build_visible_presentation_annotations(
        self,
        series: SeriesRecord,
        instance: InstanceRecord,
    ) -> tuple[PresentationAnnotationRecord, ...]:
        if not instance.sop_instance_uid:
            return ()

        presentation_states = series.presentation_states_by_sop_uid.get(str(instance.sop_instance_uid), [])
        return tuple(
            annotation
            for presentation_state in presentation_states
            for annotation in presentation_state.annotations
        )

    @staticmethod
    def _serialize_annotations(
        annotations: tuple[Any, ...],
        *,
        image_transform: Any,
        canvas_width: int,
        canvas_height: int,
    ) -> list[AnnotationOverlayPayload]:
        if canvas_width <= 0 or canvas_height <= 0:
            return []

        matrix = image_transform.matrix
        width = max(float(canvas_width), 1.0)
        height = max(float(canvas_height), 1.0)

        def serialize_point(point: MeasurementPoint) -> dict[str, float]:
            projected = matrix @ np.asarray([point.x, point.y, 1.0], dtype=np.float64)
            return {
                "x": float(projected[0]) / width,
                "y": float(projected[1]) / height,
            }

        return [
            AnnotationOverlayPayload(
                annotationId=annotation.annotation_id,
                toolType=annotation.tool_type,
                points=[serialize_point(point) for point in annotation.points],
                text=annotation.text,
                color=annotation.color,
                size=annotation.size,
                scope=getattr(annotation, "scope", "image"),
                sliceIndex=getattr(annotation.slice_context, "slice_index", None),
            )
            for annotation in annotations
        ]

    def _resolve_current_measurement_slice_index(self, view: ViewRecord) -> int:
        if not self._is_mpr_view_type(view.view_type):
            return int(view.current_index)
        target_viewport = self._resolve_mpr_viewport(view)
        if target_viewport == MPR_VIEWPORT_CORONAL:
            return int(view.mpr_coronal_index)
        if target_viewport == MPR_VIEWPORT_SAGITTAL:
            return int(view.mpr_sagittal_index)
        return int(view.mpr_axial_index)

    def _resolve_measurement_viewport_key(self, view: ViewRecord) -> str:
        if not self._is_mpr_view_type(view.view_type):
            return "single"
        return self._resolve_mpr_viewport(view)

    @staticmethod
    def _get_stack_spacing_xy(dataset: Dataset | None) -> tuple[float, float] | None:
        pixel_spacing = getattr(dataset, "PixelSpacing", None) if dataset is not None else None
        if pixel_spacing is None or len(pixel_spacing) < 2:
            return None
        try:
            row_spacing = abs(float(pixel_spacing[0]))
            col_spacing = abs(float(pixel_spacing[1]))
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(value) and value > 0.0 for value in (row_spacing, col_spacing)):
            return None
        return (col_spacing, row_spacing)

    @staticmethod
    def _is_alignment_measurement(tool_type: str) -> bool:
        return tool_type in ALIGNMENT_MEASUREMENT_TOOL_TYPES

    def _has_alignment_physical_context(
        self,
        view: ViewRecord,
        spacing_xy: tuple[float, float] | None,
    ) -> bool:
        """Return whether a supported 2D CT view has authoritative pixel spacing."""

        return self._is_supported_alignment_view(view) and has_valid_physical_spacing(spacing_xy)

    @staticmethod
    def _is_supported_alignment_view(view: ViewRecord) -> bool:
        if view.view_type != "Stack":
            return False
        try:
            series = compat.series_registry.get(view.series_id, workspace_id=view.workspace_id)
        except Exception:
            return False
        return str(series.modality or "").strip().upper() == "CT"

    @staticmethod
    def _build_alignment_spacing_unavailable_metrics() -> tuple[MeasurementMetrics, tuple[str, ...]]:
        """Keep the user-drawn reference line but never emit a numeric pseudo-result."""

        return (
            MeasurementMetrics(unit="px", area_unit="px2"),
            (ALIGNMENT_SPACING_UNAVAILABLE_LABEL,),
        )

    def _get_mpr_spacing_xy(
        self,
        series: SeriesRecord,
        viewport_key: str,
        plane_state: MprObliquePlaneState | None = None,
    ) -> tuple[float, float] | None:
        if plane_state is not None:
            transform = self._get_series_patient_transform(series)
            if transform is not None:
                return (
                    transform.spacing_for_direction(plane_state.col),
                    transform.spacing_for_direction(plane_state.row),
                )
        spacing_x, spacing_y, spacing_z = self._get_3d_spacing_xyz(series)
        if viewport_key == MPR_VIEWPORT_CORONAL:
            return (spacing_x, spacing_z)
        if viewport_key == MPR_VIEWPORT_SAGITTAL:
            return (spacing_y, spacing_z)
        return (spacing_x, spacing_y)

    def _get_mpr_display_aspect_xy(
        self,
        series: SeriesRecord,
        viewport_key: str,
        plane_state: MprObliquePlaneState | None = None,
    ) -> tuple[float, float]:
        spacing_xy = self._get_mpr_spacing_xy(series, viewport_key, plane_state)
        if spacing_xy is None:
            return (1.0, 1.0)
        return (
            max(abs(float(spacing_xy[0])), 1e-6),
            max(abs(float(spacing_xy[1])), 1e-6),
        )

    @staticmethod
    def _get_mpr_spacing_xy_from_pose(plane_pose: PlanePose) -> tuple[float, float]:
        return (
            max(abs(float(plane_pose.pixel_spacing_col_mm)), 1e-6),
            max(abs(float(plane_pose.pixel_spacing_row_mm)), 1e-6),
        )

    @staticmethod
    def _get_mpr_display_aspect_xy_from_pose(plane_pose: PlanePose) -> tuple[float, float]:
        return compat.ViewerService._get_mpr_spacing_xy_from_pose(plane_pose)

    @staticmethod
    def _get_display_aspect_xy_from_spacing(spacing_xy: tuple[float, float] | None) -> tuple[float, float]:
        if spacing_xy is None:
            return (1.0, 1.0)
        try:
            spacing_x = abs(float(spacing_xy[0]))
            spacing_y = abs(float(spacing_xy[1]))
        except (TypeError, ValueError, IndexError):
            return (1.0, 1.0)
        if not np.isfinite(spacing_x) or spacing_x <= 0.0:
            spacing_x = 1.0
        if not np.isfinite(spacing_y) or spacing_y <= 0.0:
            spacing_y = 1.0
        return (max(spacing_x, 1e-6), max(spacing_y, 1e-6))
