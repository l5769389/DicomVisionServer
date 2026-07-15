from __future__ import annotations

"""Volume caching and interactive operation handlers."""

from app.services.viewer.shared import *  # noqa: F403


class ViewerOperationsMixin:
    @staticmethod
    def _normalize_render_3d_mode(value: object) -> str:
        return "surface" if str(value or "").strip().lower() == "surface" else "volume"

    def _resolve_representative_stack_index(self, series: SeriesRecord) -> int:
        instance_count = len(series.instances)
        if instance_count <= 1:
            return 0

        cached_entry = self._series_representative_slice_cache.get(series.series_id)
        if cached_entry is not None and cached_entry[0] == instance_count:
            return max(0, min(int(cached_entry[1]), instance_count - 1))

        sample_indexes = build_representative_sample_indexes(instance_count)
        midpoint = (instance_count - 1) / 2.0
        best_index = int(round(midpoint))
        best_score = -1.0
        readable_indexes: list[int] = []

        for index in sample_indexes:
            instance = series.instances[index]
            if not instance.sop_instance_uid:
                continue
            readable_indexes.append(index)
            try:
                cached = compat.dicom_cache.get(instance.sop_instance_uid, instance.path)
            except HTTPException:
                readable_indexes.pop()
                continue

            score = score_representative_pixels(cached.source_pixels)
            if score > best_score or (abs(score - best_score) <= 1e-6 and abs(index - midpoint) < abs(best_index - midpoint)):
                best_score = score
                best_index = index

        if best_score <= 1e-6 and readable_indexes:
            best_index = min(readable_indexes, key=lambda index: abs(index - midpoint))

        best_index = max(0, min(best_index, instance_count - 1))
        self._series_representative_slice_cache[series.series_id] = (instance_count, best_index)
        logger.info(
            "representative stack slice resolved series_id=%s index=%s total=%s score=%.4f",
            series.series_id,
            best_index,
            instance_count,
            max(best_score, 0.0),
        )
        return best_index

    def _get_3d_spacing_xyz(self, series: SeriesRecord) -> tuple[float, float, float]:
        transform = self._get_series_patient_transform(series)
        if transform is not None:
            return transform.spacing_xyz()

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

    def _get_series_volume(
        self,
        series: SeriesRecord,
        *,
        progress_callback: ViewRenderProgressCallback | None = None,
    ) -> np.ndarray:
        volume_cache_key = self._build_series_volume_cache_key(series)
        cached_volume = self._get_cached_series_volume(volume_cache_key)
        if cached_volume is not None:
            self._emit_render_progress(
                progress_callback,
                "volume",
                progress_percent=70,
                loaded_count=len(series.instances),
                total_count=len(series.instances),
            )
            return cached_volume

        build_lock = self._get_series_volume_build_lock(volume_cache_key)
        if build_lock.locked():
            self._emit_render_progress(progress_callback, "waiting", progress_percent=8)

        with build_lock:
            cached_volume = self._get_cached_series_volume(volume_cache_key)
            if cached_volume is not None:
                self._emit_render_progress(
                    progress_callback,
                    "volume",
                    progress_percent=70,
                    loaded_count=len(series.instances),
                    total_count=len(series.instances),
                )
                return cached_volume

            started_at = perf_counter()
            volume = self._build_series_volume(series, progress_callback=progress_callback)
            stored_volume = self._store_series_volume(volume_cache_key, volume)
            self._emit_render_progress(
                progress_callback,
                "volume",
                progress_percent=70,
                loaded_count=len(series.instances),
                total_count=len(series.instances),
            )
            logger.info(
                "series volume built series_id=%s cache_key=%s shape=%s bytes=%s elapsed_ms=%.1f",
                series.series_id,
                volume_cache_key,
                stored_volume.shape,
                int(stored_volume.nbytes),
                (perf_counter() - started_at) * 1000.0,
            )
            return stored_volume

    @staticmethod
    def _build_series_volume_cache_key(series: SeriesRecord) -> str:
        cached_key = getattr(series, "volume_cache_key", None)
        if cached_key:
            return str(cached_key)

        content_keys = [
            compat.dicom_cache.build_instance_content_key(instance.sop_instance_uid, instance.path)
            for instance in series.instances
            if instance.sop_instance_uid
        ]
        digest = hashlib.sha256("\n".join(content_keys).encode("utf-8")).hexdigest()
        volume_cache_key = f"volume::{digest}"
        try:
            series.volume_cache_key = volume_cache_key
        except Exception:
            pass
        return volume_cache_key

    def _build_series_volume(
        self,
        series: SeriesRecord,
        *,
        progress_callback: ViewRenderProgressCallback | None = None,
    ) -> np.ndarray:
        slice_entries: list[tuple[np.ndarray, np.ndarray | None, np.ndarray | None]] = []
        readable_total = sum(1 for instance in series.instances if instance.sop_instance_uid)
        loaded_count = 0
        last_progress_percent = -1

        for instance in series.instances:
            if not instance.sop_instance_uid:
                continue
            cached = compat.dicom_cache.get(instance.sop_instance_uid, instance.path)
            dataset = cached.dataset
            orientation = self._get_dataset_orientation(dataset)
            position = self._get_dataset_position(dataset)
            slice_entries.append((cached.source_pixels, orientation, position))
            loaded_count += 1

            if readable_total:
                progress_percent = 10 + int((loaded_count / readable_total) * 55)
                if progress_percent != last_progress_percent:
                    self._emit_render_progress(
                        progress_callback,
                        "volume",
                        progress_percent=progress_percent,
                        loaded_count=loaded_count,
                        total_count=readable_total,
                    )
                    last_progress_percent = progress_percent

        if not slice_entries:
            raise HTTPException(status_code=400, detail="Series does not contain readable pixel data")

        first_shape = slice_entries[0][0].shape
        if any(item[0].shape != first_shape for item in slice_entries):
            raise HTTPException(status_code=400, detail="MPR requires a series with consistent slice dimensions")

        self._emit_render_progress(
            progress_callback,
            "normalize",
            progress_percent=66,
            loaded_count=loaded_count,
            total_count=readable_total,
        )
        return self._build_standardized_volume(slice_entries)

    def _get_series_volume_build_lock(self, series_id: str) -> Any:
        return self._series_volume_cache.get_build_lock(series_id)

    def _get_cached_series_volume(self, series_id: str) -> np.ndarray | None:
        return self._series_volume_cache.get(series_id)

    def _store_series_volume(self, series_id: str, volume: np.ndarray) -> np.ndarray:
        return self._series_volume_cache.store(series_id, volume)

    def _handle_series_volume_cache_evict(self, series_id: str, volume: np.ndarray) -> None:
        self._series_volume_geometry_cache.pop(series_id, None)
        self._series_patient_transform_cache.pop(series_id, None)
        self._series_representative_slice_cache.pop(series_id, None)
        self._volume_render_preprocess_cache.clear()
        logger.debug("volume cache evict series_id=%s bytes=%s", series_id, int(volume.nbytes))

    def get_volume_cache_stats(self) -> dict[str, int]:
        return self._series_volume_cache.stats()

    @staticmethod
    def _get_dataset_orientation(dataset) -> np.ndarray | None:
        return get_dataset_orientation(dataset)

    @staticmethod
    def _get_dataset_position(dataset) -> np.ndarray | None:
        return get_dataset_position(dataset)

    @staticmethod
    def _normalize_vector(vector: np.ndarray) -> np.ndarray | None:
        return normalize_vector(vector)

    def _build_standardized_volume(
        self,
        slice_entries: list[tuple[np.ndarray, np.ndarray | None, np.ndarray | None]],
    ) -> np.ndarray:
        return build_standardized_volume(slice_entries, logger=self._logger)

    @staticmethod
    def _is_authoritative_drag_end(payload: ViewOperationRequest) -> bool:
        return (
            payload.action_type == DRAG_ACTION_END
            and payload.interaction_id is not None
            and payload.canvas_width is not None
            and payload.canvas_height is not None
        )

    def _handle_drag_pan(self, view: ViewRecord, payload: ViewOperationRequest) -> None:
        if payload.action_type == DRAG_ACTION_START:
            view.drag_origin_offset_x = view.offset_x
            view.drag_origin_offset_y = view.offset_y
            return

        should_apply_delta = payload.action_type == DRAG_ACTION_MOVE or self._is_authoritative_drag_end(payload)
        if should_apply_delta:
            base_offset_x = view.drag_origin_offset_x if view.drag_origin_offset_x is not None else view.offset_x
            base_offset_y = view.drag_origin_offset_y if view.drag_origin_offset_y is not None else view.offset_y
            delta_x = float(payload.x or 0.0)
            delta_y = float(payload.y or 0.0)
            if payload.canvas_width is not None and float(payload.canvas_width) > 0 and view.width:
                delta_x *= float(view.width) / float(payload.canvas_width)
            if payload.canvas_height is not None and float(payload.canvas_height) > 0 and view.height:
                delta_y *= float(view.height) / float(payload.canvas_height)
            view.offset_x = float(base_offset_x) + delta_x
            view.offset_y = float(base_offset_y) + delta_y
            view.is_initialized = True
            if payload.action_type == DRAG_ACTION_MOVE:
                return

        if payload.action_type == DRAG_ACTION_END:
            view.drag_origin_offset_x = None
            view.drag_origin_offset_y = None

    def _handle_mpr_model_rotate_3d(self, view: ViewRecord, payload: ViewOperationRequest) -> bool:
        if not self._is_mpr_view_type(view.view_type) or view.view_group is None:
            return False
        if payload.action_type not in {DRAG_ACTION_START, DRAG_ACTION_MOVE, DRAG_ACTION_END}:
            return False
        if payload.x is None or payload.y is None or not view.width or not view.height:
            if payload.action_type == DRAG_ACTION_END:
                was_dragging = view.drag_origin_arcball_x is not None
                view.drag_origin_arcball_x = None
                view.drag_origin_arcball_y = None
                return was_dragging
            return False

        group = view.view_group
        series = compat.series_registry.get(view.series_id)
        volume_shape = self._get_series_volume(series).shape
        pose_context = self._build_mpr_pose_context(view, volume_shape, series=series)
        active_viewport = self._resolve_mpr_viewport(view)
        active_plane = pose_context.poses[active_viewport]
        plane_shape = active_plane.output_shape
        pixel_aspect_x, pixel_aspect_y = self._get_mpr_display_aspect_xy_from_pose(active_plane)
        image_transform = compat.viewport_transformer.build_image_to_canvas_transform(
            image_width=int(plane_shape[1]),
            image_height=int(plane_shape[0]),
            canvas_width=int(view.width or 0),
            canvas_height=int(view.height or 0),
            view=view,
            pixel_aspect_x=pixel_aspect_x,
            pixel_aspect_y=pixel_aspect_y,
        )
        pointer_angle_rad = self._resolve_mpr_rotation_pointer_angle(
            view,
            active_plane,
            image_transform,
            float(payload.x),
            float(payload.y),
        )
        group.active_viewport = active_viewport

        if payload.action_type == DRAG_ACTION_START:
            view.drag_origin_arcball_x = pointer_angle_rad
            view.drag_origin_arcball_y = None
            if group.mpr_model_rotation_pivot_world is None:
                self._set_mpr_model_rotation_pivot_world(group, active_plane.cursor_center_world)
            return False

        previous_angle_rad = view.drag_origin_arcball_x
        if payload.action_type == DRAG_ACTION_END:
            view.drag_origin_arcball_x = None
            view.drag_origin_arcball_y = None
        elif pointer_angle_rad is not None:
            view.drag_origin_arcball_x = pointer_angle_rad

        if previous_angle_rad is None:
            if pointer_angle_rad is not None and payload.action_type != DRAG_ACTION_END:
                view.drag_origin_arcball_x = pointer_angle_rad
            return False
        if pointer_angle_rad is None:
            return payload.action_type == DRAG_ACTION_END

        delta_angle_rad = self._normalize_screen_full_turn_delta(
            float(pointer_angle_rad) - float(previous_angle_rad)
        )
        if abs(delta_angle_rad) < 1e-6:
            return payload.action_type == DRAG_ACTION_END

        self._apply_mpr_model_rotation_delta(
            view.view_group,
            active_plane,
            screen_angle_delta_rad=delta_angle_rad,
        )
        view.is_initialized = True
        return True

    def _apply_mpr_model_rotation_delta(
        self,
        group: ViewGroupRecord,
        active_plane: PlanePose,
        *,
        screen_angle_delta_rad: float,
    ) -> None:
        rotation_axis_world = mpr_geometry.normalize_oblique_vector(
            np.asarray(active_plane.normal_world, dtype=np.float64),
            fallback=(1.0, 0.0, 0.0),
        )
        delta_rotation = axis_angle_rotation_matrix(rotation_axis_world, float(screen_angle_delta_rad))
        self._set_mpr_model_rotation_matrix(
            group,
            delta_rotation @ self._get_mpr_model_rotation_matrix(group),
            pivot_world=active_plane.cursor_center_world,
        )

    @staticmethod
    def _resolve_rotate_3d_canvas_point(
        view: ViewRecord,
        payload: ViewOperationRequest,
    ) -> tuple[float, float, float, float] | None:
        canvas_values = (
            payload.canvas_x,
            payload.canvas_y,
            payload.canvas_width,
            payload.canvas_height,
        )
        if all(value is not None and np.isfinite(float(value)) for value in canvas_values):
            canvas_x = float(payload.canvas_x)
            canvas_y = float(payload.canvas_y)
            canvas_width = float(payload.canvas_width)
            canvas_height = float(payload.canvas_height)
            if canvas_width > 0.0 and canvas_height > 0.0:
                return canvas_x, canvas_y, canvas_width, canvas_height

        if payload.x is None or payload.y is None or not view.width or not view.height:
            return None
        if not np.isfinite(float(payload.x)) or not np.isfinite(float(payload.y)):
            return None

        canvas_width = float(view.width)
        canvas_height = float(view.height)
        return (
            float(payload.x) * canvas_width,
            float(payload.y) * canvas_height,
            canvas_width,
            canvas_height,
        )

    def _handle_drag_rotate_3d(self, view: ViewRecord, payload: ViewOperationRequest) -> None:
        if payload.action_type not in {DRAG_ACTION_START, DRAG_ACTION_MOVE, DRAG_ACTION_END}:
            return

        if payload.action_type == DRAG_ACTION_START:
            point = self._resolve_rotate_3d_canvas_point(view, payload)
            if point is None:
                return
            canvas_x, canvas_y, canvas_width, canvas_height = point
            control_point = resolve_direct_model_trackball_control_point(
                canvas_x=canvas_x,
                canvas_y=canvas_y,
                canvas_width=canvas_width,
                canvas_height=canvas_height,
            )
            view.drag_origin_arcball_x = control_point[0]
            view.drag_origin_arcball_y = control_point[1]
            view.drag_origin_arcball_z = control_point[2]
            view.drag_origin_rotation_quaternion = tuple(float(value) for value in view.rotation_quaternion)
            return

        point = self._resolve_rotate_3d_canvas_point(view, payload)
        if point is None:
            if payload.action_type == DRAG_ACTION_END:
                view.drag_origin_arcball_x = None
                view.drag_origin_arcball_y = None
                view.drag_origin_arcball_z = None
                view.drag_origin_rotation_quaternion = None
            return

        current_x, current_y, canvas_width, canvas_height = point
        current_control_point = resolve_direct_model_trackball_control_point(
            canvas_x=current_x,
            canvas_y=current_y,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
        )
        origin_x = view.drag_origin_arcball_x
        origin_y = view.drag_origin_arcball_y
        origin_z = view.drag_origin_arcball_z
        origin_quaternion = view.drag_origin_rotation_quaternion
        if origin_x is None or origin_y is None or origin_z is None or origin_quaternion is None:
            if payload.action_type == DRAG_ACTION_MOVE:
                view.drag_origin_arcball_x = current_control_point[0]
                view.drag_origin_arcball_y = current_control_point[1]
                view.drag_origin_arcball_z = current_control_point[2]
                view.drag_origin_rotation_quaternion = tuple(float(value) for value in view.rotation_quaternion)
            return

        origin_control_point = (float(origin_x), float(origin_y), float(origin_z))
        control_delta = np.asarray(current_control_point, dtype=np.float64) - np.asarray(origin_control_point, dtype=np.float64)
        if float(np.linalg.norm(control_delta)) >= 1e-6:
            view.rotation_quaternion = compat.apply_direct_model_trackball_control_points_to_quaternion(
                tuple(float(value) for value in origin_quaternion),
                origin_control_point=origin_control_point,
                current_control_point=current_control_point,
            )
            view.is_initialized = True

        if payload.action_type == DRAG_ACTION_END:
            view.drag_origin_arcball_x = None
            view.drag_origin_arcball_y = None
            view.drag_origin_arcball_z = None
            view.drag_origin_rotation_quaternion = None
            return

    def _handle_anatomical_orientation(self, view: ViewRecord, payload: ViewOperationRequest) -> bool:
        if not self._is_3d_view_type(view.view_type):
            return False
        sub_op_type = str(payload.sub_op_type or "").strip()
        if not sub_op_type.lower().startswith("orientation:"):
            return False
        quaternion = anatomical_orientation_quaternion(sub_op_type.split(":", 1)[1])
        if quaternion is None:
            return False
        view.rotation_quaternion = quaternion
        self._reset_drag_state(view)
        view.is_initialized = True
        return True

    def _handle_volume_config(self, view: ViewRecord, payload: ViewOperationRequest) -> None:
        if not self._is_3d_view_type(view.view_type):
            return
        view.volume_render_config = normalize_volume_render_config(payload.volume_config, view.volume_preset)
        view.volume_preset = str(view.volume_render_config.get("preset", view.volume_preset or "bone"))
        view.volume_render_config_source = "manual"
        view.volume_render_config_token = None
        view.is_initialized = True

    def _handle_volume_preset(self, view: ViewRecord, payload: ViewOperationRequest) -> None:
        if not self._is_3d_view_type(view.view_type):
            return

        view.volume_preset = normalize_volume_preset_name(payload.sub_op_type or "bone")
        view.volume_render_config = create_default_volume_render_config(view.volume_preset)
        view.volume_render_config_source = "preset"
        view.volume_render_config_token = None
        view.is_initialized = True

    def _handle_render_3d_mode(self, view: ViewRecord, payload: ViewOperationRequest) -> None:
        if not self._is_3d_view_type(view.view_type):
            return
        view.render_3d_mode = self._normalize_render_3d_mode(payload.render_3d_mode or payload.sub_op_type)
        if view.surface_render_config is None:
            view.surface_render_config = create_default_surface_render_config("bone")
            view.surface_render_config_source = "preset"
            view.surface_render_config_token = None
        view.is_initialized = True

    def _handle_surface_config(self, view: ViewRecord, payload: ViewOperationRequest) -> None:
        if not self._is_3d_view_type(view.view_type):
            return
        sub_op_type = str(payload.sub_op_type or "").strip()
        is_preset_operation = sub_op_type.startswith("surfacePreset")
        if is_preset_operation:
            preset_value = sub_op_type
            if ":" not in preset_value and payload.surface_config is not None:
                preset_value = str(payload.surface_config.preset or preset_value)
            preset = normalize_surface_preset_name(preset_value)
            view.surface_render_config = create_default_surface_render_config(preset)
            view.surface_render_config_source = "preset"
            view.surface_render_config_token = None
        else:
            view.surface_render_config = normalize_surface_render_config(payload.surface_config, "bone")
            view.surface_render_config_source = "manual"
            view.surface_render_config_token = None
        view.render_3d_mode = "surface"
        view.is_initialized = True

    def _handle_volume_render_options(self, view: ViewRecord, payload: ViewOperationRequest) -> bool:
        if not self._is_3d_view_type(view.view_type):
            return False
        next_remove_bed: bool | None = payload.remove_bed
        if payload.volume_render_options is not None:
            next_remove_bed = bool(payload.volume_render_options.remove_bed)
        if next_remove_bed is None:
            return False
        view.volume_remove_bed = bool(next_remove_bed)
        view.is_initialized = True
        return True

    def _handle_volume_clip(self, view: ViewRecord, payload: ViewOperationRequest) -> bool:
        if not self._is_3d_view_type(view.view_type):
            return False
        sub_op = str(payload.sub_op_type or "").strip().lower()
        if sub_op == "reset":
            view.volume_clip_mode = None
            view.volume_clip_points = ()
            view.volume_clip_rotation_quaternion = tuple(float(value) for value in view.rotation_quaternion)
            view.is_initialized = True
            return True
        if sub_op not in {"inside", "outside"}:
            return False
        points = tuple(
            (max(0.0, min(1.0, float(point.x))), max(0.0, min(1.0, float(point.y))))
            for point in (payload.points or [])
        )
        if len(points) < 3:
            return False
        view.volume_clip_mode = sub_op
        view.volume_clip_points = points
        view.volume_clip_rotation_quaternion = tuple(float(value) for value in view.rotation_quaternion)
        view.is_initialized = True
        return True

    @staticmethod
    def _build_volume_render_options_response(view: ViewRecord) -> dict[str, object]:
        clip: dict[str, object] | None = None
        if view.volume_clip_mode in {"inside", "outside"} and len(view.volume_clip_points) >= 3:
            clip = {
                "mode": view.volume_clip_mode,
                "points": [
                    {"x": float(point[0]), "y": float(point[1])}
                    for point in view.volume_clip_points
                ],
            }
        return {
            "removeBed": bool(view.volume_remove_bed),
            "clip": clip,
        }

    def _handle_drag_zoom(self, view: ViewRecord, payload: ViewOperationRequest) -> None:
        if payload.action_type == DRAG_ACTION_START:
            view.drag_origin_zoom = view.zoom
            view.drag_origin_offset_x = view.offset_x
            view.drag_origin_offset_y = view.offset_y
            return

        should_apply_delta = payload.action_type == DRAG_ACTION_MOVE or self._is_authoritative_drag_end(payload)
        if should_apply_delta:
            base_zoom = view.drag_origin_zoom if view.drag_origin_zoom is not None else view.zoom
            delta_y = float(payload.y or 0.0)
            if payload.canvas_height is not None and float(payload.canvas_height) > 0:
                normalized_delta_y = delta_y / float(payload.canvas_height)
                zoom_factor = float(np.exp(-normalized_delta_y * ZOOM_DRAG_LOG_SENSITIVITY))
            else:
                zoom_sensitivity = ZOOM_DRAG_SENSITIVITY_3D if self._is_3d_view_type(view.view_type) else ZOOM_DRAG_SENSITIVITY
                zoom_factor = max(ZOOM_DRAG_FACTOR_MIN, 1.0 - delta_y * zoom_sensitivity)
            next_zoom = compat.viewport_transformer.clamp_zoom(float(base_zoom) * zoom_factor)
            if self._is_3d_view_type(view.view_type):
                next_zoom = self._clamp_3d_zoom(next_zoom)
            view.zoom = next_zoom
            self._apply_zoom_anchor_offset(view, payload, float(base_zoom), float(next_zoom))
            view.is_initialized = True
            if payload.action_type == DRAG_ACTION_MOVE:
                return

        if payload.action_type == DRAG_ACTION_END:
            view.drag_origin_zoom = None
            view.drag_origin_offset_x = None
            view.drag_origin_offset_y = None

    @staticmethod
    def _apply_zoom_anchor_offset(
        view: ViewRecord,
        payload: ViewOperationRequest,
        base_zoom: float,
        next_zoom: float,
    ) -> None:
        if (
            payload.anchor_x is None
            or payload.anchor_y is None
            or payload.canvas_width is None
            or payload.canvas_height is None
            or float(payload.canvas_width) <= 0
            or float(payload.canvas_height) <= 0
            or not view.width
            or not view.height
            or abs(float(base_zoom)) <= 1e-9
        ):
            return

        anchor_canvas_x = float(payload.anchor_x) * float(view.width) / float(payload.canvas_width)
        anchor_canvas_y = float(payload.anchor_y) * float(view.height) / float(payload.canvas_height)
        anchor_from_center_x = anchor_canvas_x - float(view.width) * 0.5
        anchor_from_center_y = anchor_canvas_y - float(view.height) * 0.5
        base_offset_x = view.drag_origin_offset_x if view.drag_origin_offset_x is not None else view.offset_x
        base_offset_y = view.drag_origin_offset_y if view.drag_origin_offset_y is not None else view.offset_y
        zoom_ratio = float(next_zoom) / float(base_zoom)
        view.offset_x = zoom_ratio * float(base_offset_x) + (1.0 - zoom_ratio) * anchor_from_center_x
        view.offset_y = zoom_ratio * float(base_offset_y) + (1.0 - zoom_ratio) * anchor_from_center_y

    def _handle_drag_window(self, view: ViewRecord, payload: ViewOperationRequest) -> None:
        if payload.action_type == DRAG_ACTION_START:
            view.drag_origin_window_width = view.window_width
            view.drag_origin_window_center = view.window_center
            view.drag_origin_volume_render_config = None
            return

        should_apply_delta = payload.action_type == DRAG_ACTION_MOVE or self._is_authoritative_drag_end(payload)
        if should_apply_delta:
            base_ww = view.drag_origin_window_width if view.drag_origin_window_width is not None else view.window_width
            base_wl = view.drag_origin_window_center if view.drag_origin_window_center is not None else view.window_center
            base_ww = float(base_ww or 0.0)
            base_wl = float(base_wl or 0.0)
            delta_x = float(payload.x or 0.0)
            delta_y = float(payload.y or 0.0)
            sensitivity = self._resolve_window_drag_sensitivity(base_ww)
            view.window_width = base_ww + delta_x * sensitivity
            view.window_center = base_wl - delta_y * sensitivity
            view.is_initialized = True
            if payload.action_type == DRAG_ACTION_MOVE:
                return

        if payload.action_type == DRAG_ACTION_END:
            view.drag_origin_window_width = None
            view.drag_origin_window_center = None
            view.drag_origin_volume_render_config = None

    @staticmethod
    def _handle_pseudocolor(view: ViewRecord, payload: ViewOperationRequest) -> bool:
        next_preset = normalize_pseudocolor_preset(payload.pseudocolor_preset)
        if view.pseudocolor_preset == next_preset:
            return False
        view.pseudocolor_preset = next_preset
        return True

    def _get_mpr_group_views(self, view: ViewRecord) -> list[ViewRecord]:
        if view.view_group is None:
            return [view]
        group_views = compat.view_registry.list_view_group(view.view_group.group_id)
        return group_views or [view]

    @staticmethod
    def _resolve_window_drag_sensitivity(window_width: float | None) -> float:
        width = abs(float(window_width or 0.0))
        if not np.isfinite(width) or width <= 0:
            return 1.0
        scaled = width / max(float(WINDOW_DRAG_REFERENCE_WIDTH), 1.0)
        return max(float(WINDOW_DRAG_MIN_SENSITIVITY), min(float(WINDOW_DRAG_SENSITIVITY), scaled))

    def _get_group_views(self, view: ViewRecord) -> list[ViewRecord]:
        if view.view_group is None:
            return [view]
        group_views = compat.view_registry.list_view_group(view.view_group.group_id, workspace_id=view.workspace_id)
        return group_views or [view]

    @staticmethod
    def _resolve_fusion_pane_role(view: ViewRecord) -> str:
        return view.fusion_pane_role or FUSION_VIEW_TYPE_TO_PANE_ROLE.get(view.view_type, FUSION_PANE_OVERLAY_AXIAL)

    @staticmethod
    def _build_fusion_viewport_label(role: str) -> str:
        if role == FUSION_PANE_CT_AXIAL:
            return "CT Axial"
        if role == FUSION_PANE_PET_AXIAL:
            return "PET Axial"
        if role == FUSION_PANE_PET_CORONAL_MIP:
            return "PET Coronal MIP"
        return "PET/CT"

    @staticmethod
    def _build_fusion_corner_viewport_label(role: str) -> str:
        if role == FUSION_PANE_PET_CORONAL_MIP:
            return "MIP"
        return "Axial"

    @staticmethod
    def _is_fusion_pet_display_role(role: str) -> bool:
        return role in {FUSION_PANE_PET_AXIAL, FUSION_PANE_PET_CORONAL_MIP}

    @staticmethod
    def _set_pet_window_range(
        view: ViewRecord,
        *,
        min_value: float = 0.0,
        max_value: float,
    ) -> bool:
        if not np.isfinite(float(min_value)) or not np.isfinite(float(max_value)):
            raise HTTPException(status_code=400, detail="PET window range must be finite")
        next_low = float(min_value)
        next_high = float(max_value)
        if next_high <= next_low:
            raise HTTPException(status_code=400, detail="PET window max must be greater than min")
        next_width = max(1e-6, next_high - next_low)
        next_center = (next_low + next_high) / 2.0
        if (
            view.window_width is not None
            and view.window_center is not None
            and abs(float(view.window_width) - next_width) <= 1e-6
            and abs(float(view.window_center) - next_center) <= 1e-6
        ):
            return False
        view.window_width = next_width
        view.window_center = next_center
        return True

    def _set_fusion_pet_window_range(
        self,
        group: ViewGroupRecord,
        *,
        min_value: float = 0.0,
        max_value: float,
    ) -> bool:
        next_low = float(min_value) if np.isfinite(float(min_value)) else 0.0
        next_high = float(max_value) if np.isfinite(float(max_value)) else next_low + 1.0
        if next_high <= next_low:
            next_high = next_low + 1e-6
        next_width = max(1e-6, next_high - next_low)
        next_center = (next_low + next_high) / 2.0
        if (
            group.fusion_pet_window.window_width is not None
            and group.fusion_pet_window.window_center is not None
            and abs(float(group.fusion_pet_window.window_width) - next_width) <= 1e-6
            and abs(float(group.fusion_pet_window.window_center) - next_center) <= 1e-6
        ):
            return False
        group.fusion_pet_window.window_width = next_width
        group.fusion_pet_window.window_center = next_center
        return True

    @staticmethod
    def _resolve_fusion_pet_window_drag_sensitivity(window_high: float | None) -> float:
        high = float(window_high) if window_high is not None and np.isfinite(float(window_high)) else 0.0
        return max(0.001, abs(high) * 0.01)

    def _handle_pet_window(self, view: ViewRecord, payload: ViewOperationRequest) -> bool:
        current_high = self._resolve_window_max(view.window_width, view.window_center)
        if payload.action_type is None and (payload.ww is not None or payload.wl is not None):
            if payload.ww is not None and payload.wl is not None:
                next_high = float(payload.wl) + float(payload.ww) / 2.0
            elif payload.ww is not None:
                next_high = float(payload.ww)
            else:
                next_high = float(payload.wl or 0.0) * 2.0
            changed = self._set_pet_window_range(view, min_value=0.0, max_value=next_high)
        elif payload.action_type == DRAG_ACTION_START:
            view.drag_origin_window_width = float(
                current_high if current_high is not None else FUSION_DEFAULT_SUV_WINDOW_MAX
            )
            view.drag_origin_window_center = 0.0
            return True
        elif payload.action_type == DRAG_ACTION_MOVE or self._is_authoritative_drag_end(payload):
            base_high = float(
                view.drag_origin_window_width
                if view.drag_origin_window_width is not None
                else current_high if current_high is not None else FUSION_DEFAULT_SUV_WINDOW_MAX
            )
            delta = float(payload.x or 0.0) - float(payload.y or 0.0)
            next_high = base_high + delta * self._resolve_fusion_pet_window_drag_sensitivity(base_high)
            changed = self._set_pet_window_range(view, min_value=0.0, max_value=max(1e-6, next_high))
            if payload.action_type == DRAG_ACTION_END:
                view.drag_origin_window_width = None
                view.drag_origin_window_center = None
        elif payload.action_type == DRAG_ACTION_END:
            view.drag_origin_window_width = None
            view.drag_origin_window_center = None
            return True
        else:
            return False

        if changed:
            view.is_initialized = True
        return changed

    def _handle_pet_config(self, view: ViewRecord, payload: ViewOperationRequest) -> bool:
        if not self._is_pet_view_type(view.view_type):
            return False
        changed = False
        next_preset = PET_STANDALONE_PSEUDOCOLOR_PRESET
        if view.pseudocolor_preset != next_preset:
            view.pseudocolor_preset = next_preset
            changed = True
        if payload.pet_unit is not None:
            next_unit = self._normalize_fusion_pet_unit(payload.pet_unit)
            if view.pet_unit != next_unit:
                series = compat.series_registry.get(view.series_id, workspace_id=view.workspace_id)
                pet_volume = self._get_series_volume(series)
                pet_display = self._build_fusion_pet_display_volume(series, pet_volume, next_unit)
                view.pet_unit = pet_display.unit
                view.pet_unit_label = pet_display.unit_label
                pet_ww, pet_wl = self._derive_default_pet_window_for_display_volume(pet_display)
                view.window_width = pet_ww
                view.window_center = pet_wl
                changed = True
        if payload.pet_window_min is not None or payload.pet_window_max is not None:
            current_low = self._resolve_window_min(view.window_width, view.window_center)
            current_high = self._resolve_window_max(view.window_width, view.window_center)
            next_low = (
                float(payload.pet_window_min)
                if payload.pet_window_min is not None
                else float(current_low if current_low is not None else 0.0)
            )
            next_high = (
                float(payload.pet_window_max)
                if payload.pet_window_max is not None
                else float(current_high if current_high is not None else FUSION_DEFAULT_SUV_WINDOW_MAX)
            )
            if self._set_pet_window_range(view, min_value=next_low, max_value=next_high):
                changed = True
        if changed:
            view.is_initialized = True
        return changed

    def _bump_fusion_revision(self, group: ViewGroupRecord | None) -> int | None:
        if group is None or str(group.group_type).lower() != "fusion":
            return None
        group.fusion_revision += 1
        return int(group.fusion_revision)

    def _get_fusion_revision(self, group: ViewGroupRecord | None) -> int | None:
        if group is None or str(group.group_type).lower() != "fusion":
            return None
        return int(group.fusion_revision)

    def _map_fusion_registration_canvas_delta_to_plane_mm(
        self,
        view: ViewRecord,
        *,
        delta_x: float,
        delta_y: float,
        origin_registration: FusionRegistrationState | None = None,
    ) -> tuple[float, float]:
        """Map a screen-space registration drag into the CT axial plane axes."""
        cached_mapping = self._resolve_fusion_registration_cached_canvas_mapping(view, origin_registration)
        if cached_mapping is not None:
            row_mm, col_mm = self._map_fusion_registration_canvas_delta_with_mapping(
                cached_mapping,
                delta_x=delta_x,
                delta_y=delta_y,
            )
            if np.isfinite(row_mm) and np.isfinite(col_mm):
                return row_mm, col_mm
        try:
            group, ct_series, _ = self._resolve_fusion_group_series(view)
            ct_volume = self._get_series_volume(ct_series)
            ct_geometry = self._get_series_volume_geometry(ct_series, ct_volume.shape)
            axial_index = group.fusion_axial_index if group is not None else int(ct_volume.shape[0]) // 2
            plane = build_ct_axial_plane(ct_geometry, tuple(int(value) for value in ct_volume.shape), axial_index)
            pixel_aspect_x, pixel_aspect_y = self._get_display_aspect_xy_from_spacing(
                (float(plane.pixel_spacing_col_mm), float(plane.pixel_spacing_row_mm))
            )
            image_transform = compat.viewport_transformer.build_image_to_canvas_transform(
                image_width=int(plane.output_shape[1]),
                image_height=int(plane.output_shape[0]),
                canvas_width=view.width or int(plane.output_shape[1]),
                canvas_height=view.height or int(plane.output_shape[0]),
                view=view,
                pixel_aspect_x=pixel_aspect_x,
                pixel_aspect_y=pixel_aspect_y,
            )
            inverse_linear, _ = image_transform.inverse_components()
            source_delta = inverse_linear @ np.asarray([float(delta_x), float(delta_y)], dtype=np.float64)
            col_mm = float(source_delta[0]) * float(plane.pixel_spacing_col_mm)
            row_mm = float(source_delta[1]) * float(plane.pixel_spacing_row_mm)
            if np.isfinite(col_mm) and np.isfinite(row_mm):
                return row_mm, col_mm
        except Exception:
            logger.debug("failed to map fusion registration canvas delta; falling back to zoom", exc_info=True)

        pixels_per_mm = max(float(view.zoom or 1.0), 1e-6)
        return float(delta_y) / pixels_per_mm, float(delta_x) / pixels_per_mm

    def _map_fusion_registration_canvas_point_to_plane_mm(
        self,
        view: ViewRecord,
        *,
        canvas_x: float | None,
        canvas_y: float | None,
        origin_registration: FusionRegistrationState | None = None,
    ) -> tuple[float, float]:
        """Map a rendered overlay canvas point to row/col mm from the CT axial center."""
        if canvas_x is None or canvas_y is None:
            return 0.0, 0.0
        cached_mapping = self._resolve_fusion_registration_cached_canvas_mapping(view, origin_registration)
        if cached_mapping is not None:
            row_mm, col_mm = self._map_fusion_registration_canvas_point_with_mapping(
                cached_mapping,
                canvas_x=float(canvas_x),
                canvas_y=float(canvas_y),
            )
            if np.isfinite(row_mm) and np.isfinite(col_mm):
                return row_mm, col_mm
        try:
            group, ct_series, _ = self._resolve_fusion_group_series(view)
            ct_volume = self._get_series_volume(ct_series)
            ct_geometry = self._get_series_volume_geometry(ct_series, ct_volume.shape)
            axial_index = group.fusion_axial_index if group is not None else int(ct_volume.shape[0]) // 2
            plane = build_ct_axial_plane(ct_geometry, tuple(int(value) for value in ct_volume.shape), axial_index)
            pixel_aspect_x, pixel_aspect_y = self._get_display_aspect_xy_from_spacing(
                (float(plane.pixel_spacing_col_mm), float(plane.pixel_spacing_row_mm))
            )
            image_transform = compat.viewport_transformer.build_image_to_canvas_transform(
                image_width=int(plane.output_shape[1]),
                image_height=int(plane.output_shape[0]),
                canvas_width=view.width or int(plane.output_shape[1]),
                canvas_height=view.height or int(plane.output_shape[0]),
                view=view,
                pixel_aspect_x=pixel_aspect_x,
                pixel_aspect_y=pixel_aspect_y,
            )
            inverse_linear, inverse_offset = image_transform.inverse_components()
            source_point = inverse_linear @ np.asarray([float(canvas_x), float(canvas_y)], dtype=np.float64) + inverse_offset
            col_mm = (float(source_point[0]) - float(plane.output_shape[1]) / 2.0) * float(plane.pixel_spacing_col_mm)
            row_mm = (float(source_point[1]) - float(plane.output_shape[0]) / 2.0) * float(plane.pixel_spacing_row_mm)
            if np.isfinite(row_mm) and np.isfinite(col_mm):
                return row_mm, col_mm
        except Exception:
            logger.debug("failed to map fusion registration canvas pivot; falling back to viewport center", exc_info=True)

        pixels_per_mm = max(float(view.zoom or 1.0), 1e-6)
        center_x = float(view.width or 0.0) / 2.0 + float(view.offset_x or 0.0)
        center_y = float(view.height or 0.0) / 2.0 + float(view.offset_y or 0.0)
        return (float(canvas_y) - center_y) / pixels_per_mm, (float(canvas_x) - center_x) / pixels_per_mm

    @staticmethod
    def _normalize_fusion_registration_rotation_delta(delta_degrees: float) -> float:
        return (float(delta_degrees) + 180.0) % 360.0 - 180.0

    def _resolve_fusion_registration_cached_canvas_mapping(
        self,
        view: ViewRecord,
        origin_registration: FusionRegistrationState | None,
    ) -> FusionRegistrationCanvasMapping | None:
        if origin_registration is None:
            return None
        try:
            group, ct_series, pet_series = self._resolve_fusion_group_series(view)
            cache_key = self._build_fusion_registration_pet_layer_cache_key(
                view,
                group,
                ct_series,
                pet_series,
                origin_registration,
            )
            cached = self._get_fusion_registration_pet_layer_cache(cache_key)
            if cached is not None and cached.canvas_mapping is not None:
                return cached.canvas_mapping
            locked_frame = self._get_locked_fusion_registration_overlay_frame(view, group)
            return locked_frame.canvas_mapping if locked_frame is not None else None
        except Exception:
            logger.debug("failed to resolve fusion registration cached canvas mapping", exc_info=True)
            return None

    def _resolve_fusion_registration_overlay_render_frame(
        self,
        view: ViewRecord,
        group: ViewGroupRecord,
        ct_series: SeriesRecord,
        pet_series: SeriesRecord,
        origin_registration: FusionRegistrationState | None,
    ) -> FusionRegistrationOverlayRenderFrame | None:
        locked_frame = self._get_locked_fusion_registration_overlay_frame(view, group)
        if locked_frame is not None:
            return locked_frame
        if origin_registration is None:
            return None
        try:
            cache_key = self._build_fusion_registration_pet_layer_cache_key(
                view,
                group,
                ct_series,
                pet_series,
                origin_registration,
            )
            cached = self._get_fusion_registration_pet_layer_cache(cache_key)
            if cached is None or cached.overlay_frame is None:
                return None
            self._lock_fusion_registration_overlay_frame(view, group, cached.overlay_frame)
            return cached.overlay_frame
        except Exception:
            logger.debug("failed to resolve fusion registration overlay render frame", exc_info=True)
            return None

    def _resolve_fusion_registration_pet_center_canvas(
        self,
        view: ViewRecord,
        group: ViewGroupRecord,
        origin_registration: FusionRegistrationState | None,
    ) -> tuple[float, float] | None:
        locked_frame = self._get_locked_fusion_registration_overlay_frame(view, group)
        if locked_frame is not None and locked_frame.pet_center_canvas is not None:
            return locked_frame.pet_center_canvas
        if origin_registration is None:
            return None
        try:
            _, ct_series, pet_series = self._resolve_fusion_group_series(view)
            cache_key = self._build_fusion_registration_pet_layer_cache_key(
                view,
                group,
                ct_series,
                pet_series,
                origin_registration,
            )
            cached = self._get_fusion_registration_pet_layer_cache(cache_key)
            if cached is None:
                return None
            self._lock_fusion_registration_overlay_frame(view, group, cached.overlay_frame)
            return cached.pet_center_canvas
        except Exception:
            logger.debug("failed to resolve fusion registration PET center", exc_info=True)
            return None

    def _resolve_fusion_registration_pointer_angle_rad(
        self,
        view: ViewRecord,
        payload: ViewOperationRequest,
    ) -> float | None:
        current_x = payload.current_x
        current_y = payload.current_y
        if (
            current_x is None
            or current_y is None
            or not view.width
            or not view.height
            or not np.isfinite(float(current_x))
            or not np.isfinite(float(current_y))
        ):
            return None

        pivot_x = payload.pivot_x
        pivot_y = payload.pivot_y
        if (
            pivot_x is None
            or pivot_y is None
            or not np.isfinite(float(pivot_x))
            or not np.isfinite(float(pivot_y))
        ):
            pivot_x = float(view.width) / 2.0
            pivot_y = float(view.height) / 2.0
        vector_x = float(current_x) - float(pivot_x)
        vector_y = float(current_y) - float(pivot_y)
        if float(np.hypot(vector_x, vector_y)) < 4.0:
            return None
        return float(np.arctan2(vector_y, vector_x))

    def _resolve_fusion_registration_pointer_rotation_delta_degrees(
        self,
        view: ViewRecord,
        payload: ViewOperationRequest,
        *,
        pivot_x: float | None = None,
        pivot_y: float | None = None,
    ) -> float | None:
        anchor_x = payload.anchor_x
        anchor_y = payload.anchor_y
        current_x = payload.current_x
        current_y = payload.current_y
        if all(
            value is not None and np.isfinite(float(value))
            for value in (anchor_x, anchor_y, current_x, current_y)
        ):
            pivot_x = payload.pivot_x if pivot_x is None else pivot_x
            pivot_y = payload.pivot_y if pivot_y is None else pivot_y
            if (
                pivot_x is None
                or pivot_y is None
                or not np.isfinite(float(pivot_x))
                or not np.isfinite(float(pivot_y))
            ):
                pivot_x = float(view.width or 0.0) / 2.0
                pivot_y = float(view.height or 0.0) / 2.0
            start = np.asarray([float(anchor_x) - float(pivot_x), float(anchor_y) - float(pivot_y)], dtype=np.float64)
            current = np.asarray([float(current_x) - float(pivot_x), float(current_y) - float(pivot_y)], dtype=np.float64)
            if float(np.linalg.norm(start)) >= 4.0 and float(np.linalg.norm(current)) >= 4.0:
                start_angle = float(np.degrees(np.arctan2(start[1], start[0])))
                current_angle = float(np.degrees(np.arctan2(current[1], current[0])))
                return self._normalize_fusion_registration_rotation_delta(current_angle - start_angle)

        return None

    def _resolve_fusion_registration_rotation_delta_degrees(
        self,
        view: ViewRecord,
        payload: ViewOperationRequest,
        *,
        pivot_x: float | None = None,
        pivot_y: float | None = None,
    ) -> float:
        if payload.rotation_delta_degrees is not None and np.isfinite(float(payload.rotation_delta_degrees)):
            return float(payload.rotation_delta_degrees)
        pointer_delta = self._resolve_fusion_registration_pointer_rotation_delta_degrees(
            view,
            payload,
            pivot_x=pivot_x,
            pivot_y=pivot_y,
        )
        if pointer_delta is not None:
            return pointer_delta
        return float(payload.x or 0.0) * 0.35

    def _apply_fusion_registration_rotation_drag(
        self,
        view: ViewRecord,
        payload: ViewOperationRequest,
        registration: FusionRegistrationState,
        *,
        origin_registration: FusionRegistrationState,
        origin_row: float,
        origin_col: float,
        origin_rotation: float,
    ) -> bool:
        group = view.view_group

        def resolve_rotation_pivot_canvas() -> tuple[float | None, float | None]:
            return payload.pivot_x, payload.pivot_y

        pivot_x, pivot_y = resolve_rotation_pivot_canvas()

        def apply_absolute_delta(delta_degrees: float) -> None:
            pivot_row, pivot_col = self._map_fusion_registration_canvas_point_to_plane_mm(
                view,
                canvas_x=pivot_x,
                canvas_y=pivot_y,
                origin_registration=origin_registration,
            )
            angle_rad = float(np.deg2rad(float(delta_degrees)))
            cos_angle = float(np.cos(angle_rad))
            sin_angle = float(np.sin(angle_rad))
            origin_vector_col = float(origin_col) - pivot_col
            origin_vector_row = float(origin_row) - pivot_row
            registration.translate_col_mm = (
                pivot_col
                + cos_angle * origin_vector_col
                - sin_angle * origin_vector_row
            )
            registration.translate_row_mm = (
                pivot_row
                + sin_angle * origin_vector_col
                + cos_angle * origin_vector_row
            )
            registration.rotation_degrees = float(origin_rotation) + float(delta_degrees)

        def apply_incremental_delta(delta_degrees: float) -> None:
            pivot_row, pivot_col = self._map_fusion_registration_canvas_point_to_plane_mm(
                view,
                canvas_x=pivot_x,
                canvas_y=pivot_y,
                origin_registration=origin_registration,
            )
            angle_rad = float(np.deg2rad(float(delta_degrees)))
            cos_angle = float(np.cos(angle_rad))
            sin_angle = float(np.sin(angle_rad))
            current_vector_col = float(registration.translate_col_mm) - pivot_col
            current_vector_row = float(registration.translate_row_mm) - pivot_row
            registration.translate_col_mm = (
                pivot_col
                + cos_angle * current_vector_col
                - sin_angle * current_vector_row
            )
            registration.translate_row_mm = (
                pivot_row
                + sin_angle * current_vector_col
                + cos_angle * current_vector_row
            )
            registration.rotation_degrees = float(registration.rotation_degrees) + float(delta_degrees)

        absolute_delta = self._resolve_fusion_registration_rotation_delta_degrees(
            view,
            payload,
            pivot_x=pivot_x,
            pivot_y=pivot_y,
        )
        if (
            payload.rotation_delta_degrees is not None
            or self._resolve_fusion_registration_pointer_rotation_delta_degrees(
                view,
                payload,
                pivot_x=pivot_x,
                pivot_y=pivot_y,
            ) is not None
        ):
            apply_absolute_delta(float(absolute_delta))
            if payload.action_type == DRAG_ACTION_END:
                view.drag_origin_arcball_x = None
                view.drag_origin_arcball_y = None
            return True

        absolute_pointer_delta = None
        pointer_angle_rad = self._resolve_fusion_registration_pointer_angle_rad(view, payload)
        previous_angle_rad = view.drag_origin_arcball_x
        if payload.action_type == DRAG_ACTION_END:
            view.drag_origin_arcball_x = None
            view.drag_origin_arcball_y = None
        elif pointer_angle_rad is not None:
            view.drag_origin_arcball_x = pointer_angle_rad
            view.drag_origin_arcball_y = None

        if absolute_pointer_delta is not None:
            apply_absolute_delta(float(absolute_pointer_delta))
            return True

        if previous_angle_rad is not None and pointer_angle_rad is not None:
            delta_angle_rad = self._normalize_screen_full_turn_delta(
                float(pointer_angle_rad) - float(previous_angle_rad)
            )
            if abs(delta_angle_rad) < 1e-8:
                return payload.action_type == DRAG_ACTION_END
            apply_incremental_delta(float(np.degrees(delta_angle_rad)))
            return True

        if pointer_angle_rad is not None and payload.action_type != DRAG_ACTION_END:
            view.drag_origin_arcball_x = pointer_angle_rad
            view.drag_origin_arcball_y = None
            return False

        apply_absolute_delta(absolute_delta)
        return True

    def _handle_fusion_scroll(self, view: ViewRecord, payload: ViewOperationRequest) -> bool:
        if payload.delta is None:
            return False
        group, ct_series, _ = self._resolve_fusion_group_series(view)
        ct_shape = self._get_series_volume(ct_series).shape
        group.fusion_axial_index = max(0, min(int(group.fusion_axial_index) + int(payload.delta), ct_shape[0] - 1))
        self._clear_fusion_registration_overlay_frame_locks(group)
        group.fusion_revision += 1
        for group_view in self._get_group_views(view):
            self._sync_fusion_view_state_from_group(group_view)
            group_view.is_initialized = True
        return True

    def _handle_fusion_drag_pan(self, view: ViewRecord, payload: ViewOperationRequest) -> None:
        group = view.view_group
        group_views = self._get_group_views(view)
        if payload.action_type == DRAG_ACTION_START:
            for group_view in group_views:
                group_view.drag_origin_offset_x = group_view.offset_x
                group_view.drag_origin_offset_y = group_view.offset_y
            return
        if payload.action_type == DRAG_ACTION_MOVE:
            for group_view in group_views:
                base_x = group_view.drag_origin_offset_x if group_view.drag_origin_offset_x is not None else group_view.offset_x
                base_y = group_view.drag_origin_offset_y if group_view.drag_origin_offset_y is not None else group_view.offset_y
                group_view.offset_x = float(base_x) + float(payload.x or 0.0)
                group_view.offset_y = float(base_y) + float(payload.y or 0.0)
                group_view.is_initialized = True
            if group is not None:
                self._clear_fusion_registration_overlay_frame_locks(group)
                group.fusion_revision += 1
            return
        if payload.action_type == DRAG_ACTION_END:
            for group_view in group_views:
                group_view.drag_origin_offset_x = None
                group_view.drag_origin_offset_y = None

    def _handle_fusion_drag_zoom(self, view: ViewRecord, payload: ViewOperationRequest) -> None:
        group = view.view_group
        group_views = self._get_group_views(view)
        if payload.action_type == DRAG_ACTION_START:
            for group_view in group_views:
                group_view.drag_origin_zoom = group_view.zoom
                group_view.drag_origin_offset_x = group_view.offset_x
                group_view.drag_origin_offset_y = group_view.offset_y
            return
        should_apply_delta = payload.action_type == DRAG_ACTION_MOVE or self._is_authoritative_drag_end(payload)
        if should_apply_delta:
            delta_y = float(payload.y or 0.0)
            for group_view in group_views:
                base_zoom = group_view.drag_origin_zoom if group_view.drag_origin_zoom is not None else group_view.zoom
                if payload.canvas_height is not None and float(payload.canvas_height) > 0:
                    normalized_delta_y = delta_y / float(payload.canvas_height)
                    zoom_factor = float(np.exp(-normalized_delta_y * ZOOM_DRAG_LOG_SENSITIVITY))
                else:
                    zoom_factor = max(ZOOM_DRAG_FACTOR_MIN, 1.0 - delta_y * ZOOM_DRAG_SENSITIVITY)
                next_zoom = compat.viewport_transformer.clamp_zoom(float(base_zoom) * zoom_factor)
                group_view.zoom = next_zoom
                self._apply_zoom_anchor_offset(group_view, payload, float(base_zoom), float(next_zoom))
                group_view.is_initialized = True
            if group is not None:
                self._clear_fusion_registration_overlay_frame_locks(group)
                group.fusion_revision += 1
            if payload.action_type == DRAG_ACTION_MOVE:
                return
        if payload.action_type == DRAG_ACTION_END:
            for group_view in group_views:
                group_view.drag_origin_zoom = None
                group_view.drag_origin_offset_x = None
                group_view.drag_origin_offset_y = None

    def _handle_fusion_window(self, view: ViewRecord, payload: ViewOperationRequest) -> bool:
        group = view.view_group
        if group is None:
            return False
        role = self._resolve_fusion_pane_role(view)
        if self._is_fusion_pet_display_role(role):
            current_high = self._resolve_window_max(
                group.fusion_pet_window.window_width,
                group.fusion_pet_window.window_center,
            )
            if payload.action_type is None and (payload.ww is not None or payload.wl is not None):
                if payload.ww is not None and payload.wl is not None:
                    next_high = float(payload.wl) + float(payload.ww) / 2.0
                elif payload.ww is not None:
                    next_high = float(payload.ww)
                else:
                    next_high = float(payload.wl or 0.0) * 2.0
                changed = self._set_fusion_pet_window_range(group, min_value=0.0, max_value=next_high)
            elif payload.action_type == DRAG_ACTION_START:
                group.drag_origin_window_width = float(
                    current_high if current_high is not None else FUSION_DEFAULT_SUV_WINDOW_MAX
                )
                group.drag_origin_window_center = 0.0
                return True
            elif payload.action_type == DRAG_ACTION_MOVE or self._is_authoritative_drag_end(payload):
                base_high = float(
                    group.drag_origin_window_width
                    if group.drag_origin_window_width is not None
                    else current_high if current_high is not None else FUSION_DEFAULT_SUV_WINDOW_MAX
                )
                delta = float(payload.x or 0.0) - float(payload.y or 0.0)
                next_high = base_high + delta * self._resolve_fusion_pet_window_drag_sensitivity(base_high)
                changed = self._set_fusion_pet_window_range(group, min_value=0.0, max_value=next_high)
                if payload.action_type == DRAG_ACTION_END:
                    group.drag_origin_window_width = None
                    group.drag_origin_window_center = None
            elif payload.action_type == DRAG_ACTION_END:
                group.drag_origin_window_width = None
                group.drag_origin_window_center = None
                return True
            else:
                return False

            if not changed:
                return False
            self._clear_fusion_registration_overlay_frame_locks(group)
            group.fusion_revision += 1
            for group_view in self._get_group_views(view):
                self._sync_fusion_view_state_from_group(group_view)
                group_view.is_initialized = True
            return True

        target_window = group.window

        if payload.action_type is None and (payload.ww is not None or payload.wl is not None):
            if payload.ww is not None:
                target_window.window_width = float(payload.ww)
            if payload.wl is not None:
                target_window.window_center = float(payload.wl)
        elif payload.action_type == DRAG_ACTION_START:
            group.drag_origin_window_width = target_window.window_width
            group.drag_origin_window_center = target_window.window_center
            return True
        elif payload.action_type == DRAG_ACTION_MOVE or self._is_authoritative_drag_end(payload):
            base_ww = float(group.drag_origin_window_width if group.drag_origin_window_width is not None else target_window.window_width or 0.0)
            base_wl = float(group.drag_origin_window_center if group.drag_origin_window_center is not None else target_window.window_center or 0.0)
            sensitivity = self._resolve_window_drag_sensitivity(base_ww)
            target_window.window_width = base_ww + float(payload.x or 0.0) * sensitivity
            target_window.window_center = base_wl - float(payload.y or 0.0) * sensitivity
            if payload.action_type == DRAG_ACTION_END:
                group.drag_origin_window_width = None
                group.drag_origin_window_center = None
        elif payload.action_type == DRAG_ACTION_END:
            group.drag_origin_window_width = None
            group.drag_origin_window_center = None
            return True
        else:
            return False

        group.fusion_revision += 1
        for group_view in self._get_group_views(view):
            self._sync_fusion_view_state_from_group(group_view)
            group_view.is_initialized = True
        return True

    def _handle_fusion_pseudocolor(self, view: ViewRecord, payload: ViewOperationRequest) -> bool:
        group = view.view_group
        if group is None or payload.pseudocolor_preset is None:
            return False
        next_preset = normalize_pseudocolor_preset(payload.pseudocolor_preset)
        if group.fusion_pet_pseudocolor_preset == next_preset:
            return False
        group.fusion_pet_pseudocolor_preset = next_preset
        self._clear_fusion_registration_overlay_frame_locks(group)
        group.fusion_revision += 1
        for group_view in self._get_group_views(view):
            self._sync_fusion_view_state_from_group(group_view)
            group_view.is_initialized = True
        return True

    def _handle_fusion_config(self, view: ViewRecord, payload: ViewOperationRequest) -> bool:
        group = view.view_group
        if group is None:
            return False
        should_finalize_drag = payload.action_type == DRAG_ACTION_END
        changed = False
        if payload.fusion_alpha is not None:
            next_alpha = max(0.0, min(float(payload.fusion_alpha), 1.0))
            if abs(group.fusion_alpha - next_alpha) > 1e-6:
                group.fusion_alpha = next_alpha
                changed = True
        if payload.pseudocolor_preset is not None:
            next_preset = normalize_pseudocolor_preset(payload.pseudocolor_preset)
            if group.fusion_pet_pseudocolor_preset != next_preset:
                group.fusion_pet_pseudocolor_preset = next_preset
                changed = True
        if payload.fusion_pet_unit is not None:
            next_unit = self._normalize_fusion_pet_unit(payload.fusion_pet_unit)
            if group.fusion_pet_unit != next_unit:
                group.fusion_pet_unit = next_unit
                try:
                    _, _, pet_series = self._resolve_fusion_group_series(view)
                    pet_volume = self._get_series_volume(pet_series)
                    pet_display = self._build_fusion_pet_display_volume(pet_series, pet_volume, next_unit)
                    group.fusion_pet_unit = pet_display.unit
                    pet_ww, pet_wl = self._derive_default_pet_window_for_display_volume(pet_display)
                    group.fusion_pet_window.window_width = pet_ww
                    group.fusion_pet_window.window_center = pet_wl
                except Exception:
                    logger.debug("failed to reset fusion PET window for unit=%s", next_unit, exc_info=True)
                changed = True
        if payload.fusion_pet_window_min is not None or payload.fusion_pet_window_max is not None:
            current_high = self._resolve_window_max(group.fusion_pet_window.window_width, group.fusion_pet_window.window_center)
            next_high = (
                float(payload.fusion_pet_window_max)
                if payload.fusion_pet_window_max is not None
                else float(current_high or FUSION_DEFAULT_SUV_WINDOW_MAX)
            )
            if not np.isfinite(next_high):
                next_high = FUSION_DEFAULT_SUV_WINDOW_MAX
            if self._set_fusion_pet_window_range(group, min_value=0.0, max_value=next_high):
                changed = True
        if changed:
            self._clear_fusion_registration_overlay_frame_locks(group)
            group.fusion_revision += 1
            for group_view in self._get_group_views(view):
                self._sync_fusion_view_state_from_group(group_view)
                group_view.is_initialized = True
        return changed or should_finalize_drag

    @staticmethod
    def _finite_or_default(value: float | int | None, default: float) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return float(default)
        return number if np.isfinite(number) else float(default)

    def _set_fusion_registration_preview_drag(
        self,
        view: ViewRecord,
        group: ViewGroupRecord,
        payload: ViewOperationRequest,
        *,
        sub_op_type: str,
        origin_registration: FusionRegistrationState,
        rotation_delta_degrees: float | None = None,
    ) -> None:
        self._fusion_registration_preview_drags[group.group_id] = FusionRegistrationPreviewDrag(
            group_id=str(group.group_id),
            origin_registration=self._copy_fusion_registration_state(origin_registration),
            sub_op_type=sub_op_type,
            delta_x=self._finite_or_default(payload.x, 0.0),
            delta_y=self._finite_or_default(payload.y, 0.0),
            pivot_x=self._finite_or_default(payload.pivot_x, float(view.width or 0) / 2.0),
            pivot_y=self._finite_or_default(payload.pivot_y, float(view.height or 0) / 2.0),
            rotation_delta_degrees=self._finite_or_default(
                rotation_delta_degrees if rotation_delta_degrees is not None else payload.rotation_delta_degrees,
                0.0,
            ),
        )

    def _prime_fusion_registration_preview_cache(
        self,
        view: ViewRecord,
        group: ViewGroupRecord,
    ) -> None:
        for group_view in self._get_group_views(view):
            if not group_view.width or not group_view.height:
                continue
            if self._resolve_fusion_pane_role(group_view) not in {
                FUSION_PANE_OVERLAY_AXIAL,
                FUSION_PANE_PET_AXIAL,
            }:
                continue
            try:
                self._render_fusion_view(group_view, image_format="png", fast_preview=False)
            except Exception:
                logger.warning(
                    "failed to prime fusion registration preview cache view_id=%s group_id=%s",
                    group_view.view_id,
                    group.group_id,
                    exc_info=True,
                )

    def _handle_fusion_registration(self, view: ViewRecord, payload: ViewOperationRequest) -> bool:
        group = view.view_group
        if group is None:
            return False
        sub_op = str(payload.sub_op_type or "translate").strip().lower()
        registration = group.fusion_registration
        if sub_op == "reset":
            self._fusion_registration_preview_drags.pop(group.group_id, None)
            self._clear_fusion_registration_overlay_frame_locks(group)
            group.fusion_registration = FusionRegistrationState()
        elif sub_op == "save":
            self._fusion_registration_preview_drags.pop(group.group_id, None)
            compat.view_group_registry.save_fusion_registration(group)
        elif sub_op == "load":
            self._fusion_registration_preview_drags.pop(group.group_id, None)
            self._clear_fusion_registration_overlay_frame_locks(group)
            return self._load_fusion_registration_sidecar(view, payload.fusion_registration_file)
        elif payload.action_type == DRAG_ACTION_START:
            group.rotation_drag = None
            origin_registration = self._copy_fusion_registration_state(registration)
            group.crosshair_drag_origin_center = (
                registration.translate_row_mm,
                registration.translate_col_mm,
                registration.rotation_degrees,
            )
            self._prime_fusion_registration_preview_cache(view, group)
            self._set_fusion_registration_preview_drag(
                view,
                group,
                payload,
                sub_op_type=sub_op,
                origin_registration=origin_registration,
            )
            view.drag_origin_arcball_x = (
                self._resolve_fusion_registration_pointer_angle_rad(view, payload)
                if sub_op == "rotate"
                else None
            )
            view.drag_origin_arcball_y = None
            return True
        elif payload.action_type in {DRAG_ACTION_MOVE, DRAG_ACTION_END}:
            origin = group.crosshair_drag_origin_center or (
                registration.translate_row_mm,
                registration.translate_col_mm,
                registration.rotation_degrees,
            )
            origin_row, origin_col, origin_rotation = (float(origin[0]), float(origin[1]), float(origin[2]))
            origin_registration = FusionRegistrationState(
                translate_row_mm=origin_row,
                translate_col_mm=origin_col,
                rotation_degrees=origin_rotation,
                saved=bool(registration.saved),
            )
            if sub_op == "rotate":
                changed = self._apply_fusion_registration_rotation_drag(
                    view,
                    payload,
                    registration,
                    origin_registration=origin_registration,
                    origin_row=origin_row,
                    origin_col=origin_col,
                    origin_rotation=origin_rotation,
                )
                if not changed:
                    if payload.action_type == DRAG_ACTION_END:
                        group.crosshair_drag_origin_center = None
                        view.drag_origin_arcball_x = None
                        view.drag_origin_arcball_y = None
                    return payload.action_type == DRAG_ACTION_END
            else:
                delta_row_mm, delta_col_mm = self._map_fusion_registration_canvas_delta_to_plane_mm(
                    view,
                    delta_x=float(payload.x or 0.0),
                    delta_y=float(payload.y or 0.0),
                    origin_registration=origin_registration,
                )
                registration.translate_col_mm = origin_col + delta_col_mm
                registration.translate_row_mm = origin_row + delta_row_mm
            registration.saved = False
            if payload.action_type in {DRAG_ACTION_MOVE, DRAG_ACTION_END}:
                effective_rotation_delta = (
                    float(registration.rotation_degrees) - float(origin_rotation)
                    if sub_op == "rotate"
                    else None
                )
                self._set_fusion_registration_preview_drag(
                    view,
                    group,
                    payload,
                    sub_op_type=sub_op,
                    origin_registration=origin_registration,
                    rotation_delta_degrees=effective_rotation_delta,
                )
            if payload.action_type == DRAG_ACTION_END:
                group.crosshair_drag_origin_center = None
                view.drag_origin_arcball_x = None
                view.drag_origin_arcball_y = None
        else:
            return False
        group.fusion_revision += 1
        for group_view in self._get_group_views(view):
            self._sync_fusion_view_state_from_group(group_view)
            group_view.is_initialized = True
        return True

    @staticmethod
    def _require_fusion_registration_mapping(value: object, name: str) -> dict[str, object]:
        if not isinstance(value, dict):
            raise HTTPException(status_code=400, detail=f"{name} must be an object")
        return value

    @staticmethod
    def _require_finite_fusion_registration_number(value: object, name: str) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"{name} must be a finite number") from exc
        if not np.isfinite(number):
            raise HTTPException(status_code=400, detail=f"{name} must be a finite number")
        return number

    @staticmethod
    def _fusion_registration_sidecar_matches_series(sidecar_series: dict[str, object], series: SeriesRecord) -> bool:
        sidecar_series_id = str(sidecar_series.get("seriesId") or "").strip()
        sidecar_uid = str(sidecar_series.get("seriesInstanceUid") or "").strip()
        current_uid = str(series.series_instance_uid or "").strip()
        return bool(
            (sidecar_series_id and sidecar_series_id == series.series_id)
            or (sidecar_uid and current_uid and sidecar_uid == current_uid)
        )

    def _load_fusion_registration_sidecar(
        self,
        view: ViewRecord,
        sidecar_payload: dict[str, Any] | None,
    ) -> bool:
        payload = self._require_fusion_registration_mapping(sidecar_payload, "fusionRegistrationFile")
        if payload.get("format") != "DicomVisionFusionRegistration":
            raise HTTPException(status_code=400, detail="Unsupported registration file format")

        group, ct_series, pet_series = self._resolve_fusion_group_series(view)
        ct_payload = self._require_fusion_registration_mapping(payload.get("ct"), "ct")
        pet_payload = self._require_fusion_registration_mapping(payload.get("pet"), "pet")
        if not self._fusion_registration_sidecar_matches_series(ct_payload, ct_series):
            raise HTTPException(status_code=400, detail="Registration file CT series does not match the current fusion view")
        if not self._fusion_registration_sidecar_matches_series(pet_payload, pet_series):
            raise HTTPException(status_code=400, detail="Registration file PET series does not match the current fusion view")

        registration_payload = self._require_fusion_registration_mapping(payload.get("registration"), "registration")
        group.fusion_registration.translate_row_mm = self._require_finite_fusion_registration_number(
            registration_payload.get("translateRowMm"),
            "registration.translateRowMm",
        )
        group.fusion_registration.translate_col_mm = self._require_finite_fusion_registration_number(
            registration_payload.get("translateColMm"),
            "registration.translateColMm",
        )
        group.fusion_registration.rotation_degrees = self._require_finite_fusion_registration_number(
            registration_payload.get("rotationDegrees"),
            "registration.rotationDegrees",
        )
        self._clear_fusion_registration_overlay_frame_locks(group)

        pet_unit = pet_payload.get("unit")
        if pet_unit is not None:
            group.fusion_pet_unit = self._normalize_fusion_pet_unit(str(pet_unit))

        window_payload = pet_payload.get("window")
        if isinstance(window_payload, dict) and (
            window_payload.get("min") is not None or window_payload.get("max") is not None
        ):
            window_min = self._require_finite_fusion_registration_number(
                window_payload.get("min", 0.0),
                "pet.window.min",
            )
            window_max = self._require_finite_fusion_registration_number(
                window_payload.get("max"),
                "pet.window.max",
            )
            self._set_fusion_pet_window_range(group, min_value=window_min, max_value=window_max)

        compat.view_group_registry.save_fusion_registration(group)
        group.fusion_revision += 1
        for group_view in self._get_group_views(view):
            self._sync_fusion_view_state_from_group(group_view)
            group_view.is_initialized = True
        return True

    def _get_fusion_reference_plane(self, view: ViewRecord) -> PlanePose:
        group, ct_series, _ = self._resolve_fusion_group_series(view)
        ct_volume = self._get_series_volume(ct_series)
        ct_geometry = self._get_series_volume_geometry(ct_series, ct_volume.shape)

        return build_ct_axial_plane(ct_geometry, ct_volume.shape, group.fusion_axial_index)

    def _handle_mpr_crosshair(self, view: ViewRecord, payload: ViewOperationRequest) -> bool:
        if payload.x is None or payload.y is None:
            return False
        if not self._is_mpr_view_type(view.view_type):
            return False
        ensure_view_size(view)

        series = compat.series_registry.get(view.series_id)
        volume = self._get_series_volume(series)
        target_viewport = self._resolve_mpr_viewport(view)
        pose_context = self._build_mpr_pose_context(view, volume.shape, series=series)
        active_plane = pose_context.poses[target_viewport]
        plane_shape = active_plane.output_shape
        canvas_width = max(float(view.width or 0), 1.0)
        canvas_height = max(float(view.height or 0), 1.0)
        pixel_aspect_x, pixel_aspect_y = self._get_mpr_display_aspect_xy_from_pose(active_plane)
        image_transform = compat.viewport_transformer.build_image_to_canvas_transform(
            image_width=int(plane_shape[1]),
            image_height=int(plane_shape[0]),
            canvas_width=int(canvas_width),
            canvas_height=int(canvas_height),
            view=view,
            pixel_aspect_x=pixel_aspect_x,
            pixel_aspect_y=pixel_aspect_y,
        )

        def payload_to_plane_image_point() -> tuple[float, float]:
            canvas_x = min(max(float(payload.x or 0.0), 0.0), 1.0) * canvas_width
            canvas_y = min(max(float(payload.y or 0.0), 0.0), 1.0) * canvas_height
            return self._canvas_to_image_coordinates(image_transform, canvas_x, canvas_y)

        if payload.action_type == DRAG_ACTION_START:
            view.mpr_crosshair_drag_active = True
            if view.view_group is not None:
                origin_center_ijk = world_to_ijk_point(pose_context.geometry, pose_context.cursor.center_world)
                view.view_group.crosshair_drag_origin_center = tuple(float(value) for value in origin_center_ijk)
                if payload.x is not None and payload.y is not None:
                    view.view_group.crosshair_drag_origin_image = payload_to_plane_image_point()
                else:
                    view.view_group.crosshair_drag_origin_image = None
            return False

        is_drag_end = payload.action_type == DRAG_ACTION_END
        was_dragging = view.mpr_crosshair_drag_active
        if (payload.action_type != DRAG_ACTION_MOVE and not is_drag_end) or not was_dragging:
            return False

        image_x, image_y = payload_to_plane_image_point()
        depth, height, width = volume.shape
        if view.view_group is not None:
            previous_center = tuple(float(value) for value in world_to_ijk_point(pose_context.geometry, pose_context.cursor.center_world))
            next_center_world = self._resolve_mpr_center_from_image_point(
                view.view_group,
                pose_context.poses[target_viewport],
                pose_context.geometry,
                image_x,
                image_y,
            )
            next_center = world_to_ijk_point(pose_context.geometry, next_center_world)
        else:
            previous_center = (float(view.mpr_axial_index), float(view.mpr_coronal_index), float(view.mpr_sagittal_index))
            next_center = np.array(previous_center, dtype=np.float64)
            if target_viewport == MPR_VIEWPORT_CORONAL:
                next_center[2] = float(max(0.0, min(image_x - 0.5, width - 1)))
                next_center[0] = float(max(0.0, min(depth - image_y - 0.5, depth - 1)))
            elif target_viewport == MPR_VIEWPORT_SAGITTAL:
                next_center[1] = float(max(0.0, min(image_x - 0.5, height - 1)))
                next_center[0] = float(max(0.0, min(depth - image_y - 0.5, depth - 1)))
            else:
                next_center[2] = float(max(0.0, min(image_x - 0.5, width - 1)))
                next_center[1] = float(max(0.0, min(image_y - 0.5, height - 1)))

        center_changed = not np.allclose(next_center, np.asarray(previous_center, dtype=np.float64), atol=1e-6)

        if center_changed:
            if view.view_group is not None:
                next_cursor = replace(pose_context.cursor, center_world=np.asarray(next_center_world, dtype=np.float64))
                self._sync_group_from_mpr_cursor(view.view_group, next_cursor, pose_context.geometry, volume.shape)
                view.view_group.mpr_use_display_basis_for_cursor_offsets = True
            else:
                view.mpr_axial_index = int(np.round(next_center[0]))
                view.mpr_coronal_index = int(np.round(next_center[1]))
                view.mpr_sagittal_index = int(np.round(next_center[2]))
            view.current_index = view.mpr_axial_index
            view.is_initialized = True

        if is_drag_end:
            view.mpr_crosshair_drag_active = False
            if view.view_group is not None:
                view.view_group.crosshair_drag_origin_center = None
                view.view_group.crosshair_drag_origin_image = None
            return was_dragging or center_changed

        return center_changed

    def _resolve_mpr_center_from_image_point(
        self,
        group: ViewGroupRecord,
        plane_pose: PlanePose,
        geometry: VolumeGeometry,
        image_x: float,
        image_y: float,
    ) -> np.ndarray:
        origin_center = np.asarray(
            group.crosshair_drag_origin_center or world_to_ijk_point(geometry, plane_pose.cursor_center_world),
            dtype=np.float64,
        )
        origin_center_world = ijk_to_world_point(geometry, origin_center)
        origin_image_x, origin_image_y = group.crosshair_drag_origin_image or self._project_world_point_to_plane_image(
            plane_pose,
            origin_center_world,
        )
        row_offset_mm = (float(image_y) - float(origin_image_y)) * float(plane_pose.pixel_spacing_row_mm)
        col_offset_mm = (float(image_x) - float(origin_image_x)) * float(plane_pose.pixel_spacing_col_mm)
        next_center_world = (
            origin_center_world
            + np.asarray(plane_pose.row_world, dtype=np.float64) * row_offset_mm
            + np.asarray(plane_pose.col_world, dtype=np.float64) * col_offset_mm
        )
        next_center_ijk = world_to_ijk_point(geometry, next_center_world)
        clamped_center_ijk = np.array(
            [
                max(0.0, min(float(next_center_ijk[0]), geometry.shape_ijk[0] - 1)),
                max(0.0, min(float(next_center_ijk[1]), geometry.shape_ijk[1] - 1)),
                max(0.0, min(float(next_center_ijk[2]), geometry.shape_ijk[2] - 1)),
            ],
            dtype=np.float64,
        )
        return ijk_to_world_point(geometry, clamped_center_ijk)

    def _handle_mpr_oblique(self, view: ViewRecord, payload: ViewOperationRequest) -> bool:
        if not self._is_mpr_view_type(view.view_type) or view.view_group is None:
            return False
        if payload.line not in {"horizontal", "vertical"}:
            return False
        if payload.x is None or payload.y is None:
            if payload.action_type == DRAG_ACTION_END:
                was_dragging = view.view_group.rotation_drag is not None
                view.view_group.rotation_drag = None
                return was_dragging
            return False

        group = view.view_group
        series = compat.series_registry.get(view.series_id)
        volume_shape = self._get_series_volume(series).shape
        pose_context = self._build_mpr_pose_context(view, volume_shape, series=series)
        self._ensure_mpr_reference_center(group, volume_shape)
        active_viewport = self._resolve_mpr_viewport(view)
        group.active_viewport = active_viewport
        active_plane = pose_context.poses[active_viewport]
        plane_shape = active_plane.output_shape
        pixel_aspect_x, pixel_aspect_y = self._get_mpr_display_aspect_xy_from_pose(active_plane)
        image_transform = compat.viewport_transformer.build_image_to_canvas_transform(
            image_width=int(plane_shape[1]),
            image_height=int(plane_shape[0]),
            canvas_width=int(view.width or 0),
            canvas_height=int(view.height or 0),
            view=view,
            pixel_aspect_x=pixel_aspect_x,
            pixel_aspect_y=pixel_aspect_y,
        )
        pointer_angle_rad = self._resolve_mpr_rotation_pointer_angle(
            view,
            active_plane,
            image_transform,
            float(payload.x),
            float(payload.y),
        )
        if pointer_angle_rad is None:
            if payload.action_type == DRAG_ACTION_END:
                was_dragging = group.rotation_drag is not None
                group.rotation_drag = None
                return was_dragging
            return False

        if payload.action_type == DRAG_ACTION_START:
            if self._get_mpr_crosshair_mode(group) == MPR_CROSSHAIR_MODE_DOUBLE_OBLIQUE:
                self._ensure_mpr_independent_plane_normals(group, pose_context.poses)
            self._ensure_mpr_crosshair_angle_cache(group, pose_context.poses)
            start_horizontal_angle, start_vertical_angle = self._get_mpr_visible_crosshair_line_angles(
                group,
                pose_context.poses,
                active_viewport,
            )
            group.rotation_drag = MprRotationDragRecord(
                viewport=active_viewport,
                line=payload.line,
                start_cursor=self._serialize_mpr_cursor_record(pose_context.cursor),
                start_pointer_angle_rad=pointer_angle_rad,
                start_line_angle_rad=start_horizontal_angle if payload.line == "horizontal" else start_vertical_angle,
                start_independent_plane_normals=deepcopy(group.mpr_independent_plane_normals),
            )
            return False

        if payload.action_type == DRAG_ACTION_END:
            was_dragging = group.rotation_drag is not None
            if was_dragging and group.rotation_drag is not None:
                self._apply_mpr_rotation_pointer_drag(
                    group,
                    group.rotation_drag,
                    pointer_angle_rad,
                    pose_context.geometry,
                    volume_shape,
                )
            group.rotation_drag = None
            return was_dragging

        if payload.action_type != DRAG_ACTION_MOVE or group.rotation_drag is None:
            return False

        self._apply_mpr_rotation_pointer_drag(
            group,
            group.rotation_drag,
            pointer_angle_rad,
            pose_context.geometry,
            volume_shape,
        )
        view.is_initialized = True
        return True

    def _resolve_mpr_rotation_pointer_angle(
        self,
        view: ViewRecord,
        active_plane: PlanePose,
        image_transform,
        normalized_x: float,
        normalized_y: float,
    ) -> float | None:
        canvas_width = float(view.width or 0)
        canvas_height = float(view.height or 0)
        if canvas_width <= 0.0 or canvas_height <= 0.0:
            return None
        canvas_x = min(max(float(normalized_x) * canvas_width, 0.0), max(canvas_width - 1e-6, 0.0))
        canvas_y = min(max(float(normalized_y) * canvas_height, 0.0), max(canvas_height - 1e-6, 0.0))
        center_image_x, center_image_y = self._project_world_point_to_plane_image(
            active_plane,
            active_plane.cursor_center_world,
        )
        center_canvas = image_transform.matrix @ np.array([center_image_x, center_image_y, 1.0], dtype=np.float64)
        delta_x = canvas_x - float(center_canvas[0])
        delta_y = canvas_y - float(center_canvas[1])
        if float(np.hypot(delta_x, delta_y)) <= 1e-6:
            return None
        return float(np.arctan2(delta_y, delta_x))

    def _apply_mpr_rotation_pointer_drag(
        self,
        group: ViewGroupRecord,
        drag: MprRotationDragRecord,
        pointer_angle_rad: float,
        geometry: VolumeGeometry,
        volume_shape: tuple[int, int, int],
    ) -> None:
        if self._get_mpr_crosshair_mode(group) == MPR_CROSSHAIR_MODE_DOUBLE_OBLIQUE:
            self._apply_mpr_double_oblique_rotation_pointer_drag(
                group,
                drag,
                pointer_angle_rad,
                geometry,
                volume_shape,
            )
            return

        start_cursor = self._deserialize_mpr_cursor_record(drag.start_cursor)
        start_poses = self._build_mpr_plane_poses(start_cursor, geometry, volume_shape)
        start_active_plane = start_poses[drag.viewport]
        active_normal = np.asarray(start_active_plane.normal_world, dtype=np.float64)
        active_row = np.asarray(start_active_plane.row_world, dtype=np.float64)
        active_col = np.asarray(start_active_plane.col_world, dtype=np.float64)
        target_line_angle_rad = float(drag.start_line_angle_rad) + self._normalize_screen_full_turn_delta(
            float(pointer_angle_rad) - float(drag.start_pointer_angle_rad)
        )
        self._set_mpr_visible_crosshair_line_angles(group, drag.viewport, drag.line, target_line_angle_rad)
        target_line_world = mpr_geometry.direction_from_screen_angle(
            active_row,
            active_col,
            target_line_angle_rad,
        )
        perpendicular_line_world = mpr_geometry.direction_from_screen_angle(
            active_row,
            active_col,
            target_line_angle_rad
            + (float(np.pi / 2.0) if drag.line == "horizontal" else -float(np.pi / 2.0)),
        )

        line_directions = {
            drag.line: target_line_world,
            self._resolve_perpendicular_crosshair_line(drag.line): perpendicular_line_world,
        }
        normal_updates: dict[str, np.ndarray] = {}
        for line, line_world in line_directions.items():
            target_viewport = self._resolve_mpr_oblique_target_viewport(drag.viewport, line)
            start_target_plane = start_poses[target_viewport]
            next_target_normal = mpr_geometry.normalize_oblique_vector(
                np.cross(line_world, active_normal),
                fallback=tuple(start_target_plane.normal_world),
            )
            if float(np.dot(next_target_normal, np.asarray(start_target_plane.normal_world, dtype=np.float64))) < 0.0:
                next_target_normal = -next_target_normal
            normal_updates[target_viewport] = next_target_normal

        next_cursor = self._replace_mpr_cursor_plane_normals(start_cursor, normal_updates)
        self._sync_group_from_mpr_cursor(group, next_cursor, geometry, volume_shape)

    def _apply_mpr_double_oblique_rotation_pointer_drag(
        self,
        group: ViewGroupRecord,
        drag: MprRotationDragRecord,
        pointer_angle_rad: float,
        geometry: VolumeGeometry,
        volume_shape: tuple[int, int, int],
    ) -> None:
        start_cursor = self._deserialize_mpr_cursor_record(drag.start_cursor)
        start_poses = self._build_mpr_plane_poses(
            start_cursor,
            geometry,
            volume_shape,
            normal_overrides=drag.start_independent_plane_normals,
        )
        start_active_plane = start_poses[drag.viewport]
        active_normal = np.asarray(start_active_plane.normal_world, dtype=np.float64)
        active_row = np.asarray(start_active_plane.row_world, dtype=np.float64)
        active_col = np.asarray(start_active_plane.col_world, dtype=np.float64)
        target_line_angle_rad = float(drag.start_line_angle_rad) + self._normalize_screen_full_turn_delta(
            float(pointer_angle_rad) - float(drag.start_pointer_angle_rad)
        )
        self._set_mpr_independent_visible_crosshair_line_angle(group, drag.viewport, drag.line, target_line_angle_rad)
        target_line_world = mpr_geometry.direction_from_screen_angle(
            active_row,
            active_col,
            target_line_angle_rad,
        )
        target_viewport = self._resolve_mpr_oblique_target_viewport(drag.viewport, drag.line)
        start_target_plane = start_poses[target_viewport]
        next_target_normal = mpr_geometry.normalize_oblique_vector(
            np.cross(target_line_world, active_normal),
            fallback=tuple(start_target_plane.normal_world),
        )
        if float(np.dot(next_target_normal, np.asarray(start_target_plane.normal_world, dtype=np.float64))) < 0.0:
            next_target_normal = -next_target_normal

        next_normals = self._normal_records_from_poses(start_poses)
        next_normals[target_viewport] = tuple(float(value) for value in next_target_normal)
        group.mpr_independent_plane_normals = next_normals

    @staticmethod
    def _replace_mpr_cursor_plane_normals(
        cursor: MprCursorState,
        normal_updates: dict[str, np.ndarray],
    ) -> MprCursorState:
        orientation = np.asarray(cursor.orientation_world, dtype=np.float64).copy()
        for viewport_key, normal_world in normal_updates.items():
            convention = DEFAULT_MPR_CONVENTION.get(viewport_key, DEFAULT_MPR_CONVENTION[MPR_VIEWPORT_AXIAL])
            normalized_normal = mpr_geometry.normalize_oblique_vector(
                normal_world,
                fallback=tuple(orientation[:, convention.normal_axis_index]),
            )
            orientation[:, convention.normal_axis_index] = normalized_normal / float(convention.normal_sign)
        return replace(cursor, orientation_world=orientation)

    @staticmethod
    def _resolve_perpendicular_crosshair_line(line: str) -> str:
        return "vertical" if line == "horizontal" else "horizontal"

    @staticmethod
    def _normalize_screen_half_turn_angle(angle_rad: float) -> float:
        return mpr_geometry.normalize_screen_half_turn_angle(angle_rad)

    def _ensure_mpr_crosshair_angle_cache(
        self,
        group: ViewGroupRecord,
        poses: dict[str, PlanePose],
    ) -> None:
        for viewport_key in (MPR_VIEWPORT_AXIAL, MPR_VIEWPORT_CORONAL, MPR_VIEWPORT_SAGITTAL):
            if viewport_key in group.mpr_crosshair_angles:
                continue
            group.mpr_crosshair_angles[viewport_key] = self._get_mpr_crosshair_line_angles_from_poses(
                poses,
                viewport_key,
            )

    def _get_mpr_visible_crosshair_line_angles(
        self,
        group: ViewGroupRecord | None,
        poses: dict[str, PlanePose],
        viewport_key: str,
    ) -> tuple[float, float]:
        cached_angles = group.mpr_crosshair_angles.get(viewport_key) if group is not None else None
        if cached_angles is not None:
            return (
                self._normalize_screen_half_turn_angle(float(cached_angles[0])),
                self._normalize_screen_half_turn_angle(float(cached_angles[1])),
            )
        return self._get_mpr_crosshair_line_angles_from_poses(poses, viewport_key)

    def _set_mpr_visible_crosshair_line_angles(
        self,
        group: ViewGroupRecord,
        viewport_key: str,
        line: str,
        line_angle_rad: float,
    ) -> None:
        if line == "horizontal":
            horizontal_angle = self._normalize_screen_half_turn_angle(line_angle_rad)
            vertical_angle = self._normalize_screen_half_turn_angle(line_angle_rad + float(np.pi / 2.0))
        else:
            vertical_angle = self._normalize_screen_half_turn_angle(line_angle_rad)
            horizontal_angle = self._normalize_screen_half_turn_angle(line_angle_rad - float(np.pi / 2.0))
        group.mpr_crosshair_angles[viewport_key] = (horizontal_angle, vertical_angle)

    def _set_mpr_independent_visible_crosshair_line_angle(
        self,
        group: ViewGroupRecord,
        viewport_key: str,
        line: str,
        line_angle_rad: float,
    ) -> None:
        cached_angles = group.mpr_crosshair_angles.get(viewport_key) or (0.0, float(np.pi / 2.0))
        if line == "horizontal":
            group.mpr_crosshair_angles[viewport_key] = (
                self._normalize_screen_half_turn_angle(line_angle_rad),
                self._normalize_screen_half_turn_angle(float(cached_angles[1])),
            )
            return

        group.mpr_crosshair_angles[viewport_key] = (
            self._normalize_screen_half_turn_angle(float(cached_angles[0])),
            self._normalize_screen_half_turn_angle(line_angle_rad),
        )

    @staticmethod
    def _normalize_screen_full_turn_delta(angle_rad: float) -> float:
        full_turn = float(np.pi * 2.0)
        delta = (float(angle_rad) + float(np.pi)) % full_turn - float(np.pi)
        if delta <= -float(np.pi):
            delta += full_turn
        return delta

    def _get_mpr_display_basis(
        self,
        viewport_key: str,
        normal_dir: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return mpr_geometry.get_mpr_display_basis(viewport_key, normal_dir)

    @staticmethod
    def _resolve_mpr_oblique_target_viewport(active_viewport: str, line: str) -> str:
        if active_viewport == MPR_VIEWPORT_CORONAL:
            return MPR_VIEWPORT_AXIAL if line == "horizontal" else MPR_VIEWPORT_SAGITTAL
        if active_viewport == MPR_VIEWPORT_SAGITTAL:
            return MPR_VIEWPORT_AXIAL if line == "horizontal" else MPR_VIEWPORT_CORONAL
        return MPR_VIEWPORT_CORONAL if line == "horizontal" else MPR_VIEWPORT_SAGITTAL

    @staticmethod
    def _default_mpr_oblique_plane(viewport_key: str) -> MprObliquePlaneState:
        return mpr_geometry.default_mpr_oblique_plane(viewport_key)

    @staticmethod
    def _build_scale_bar_info(
        render_view: ViewRecord,
        image_transform,
        spacing_xy: tuple[float, float] | None,
    ) -> ScaleBarInfo | None:
        if spacing_xy is None or not render_view.width or render_view.width <= 0:
            return None

        spacing_x = max(abs(float(spacing_xy[0])), 1e-6)
        spacing_y = max(abs(float(spacing_xy[1])), 1e-6)
        inverse = np.linalg.inv(image_transform.matrix)
        image_dx = float(inverse[0, 0])
        image_dy = float(inverse[1, 0])
        mm_per_canvas_pixel = float(np.hypot(image_dx * spacing_x, image_dy * spacing_y))
        if not np.isfinite(mm_per_canvas_pixel) or mm_per_canvas_pixel <= 0.0:
            return None

        selected_length_mm = 100.0
        selected_length_px = selected_length_mm / mm_per_canvas_pixel
        if not np.isfinite(selected_length_px) or selected_length_px <= 0.0:
            return None

        return ScaleBarInfo(
            lengthNorm=float(selected_length_px) / float(render_view.width),
            label="10 cm",
        )
