from __future__ import annotations

"""PET/CT fusion rendering and registration preview."""

from app.services.viewer.shared import *  # noqa: F403


class ViewerFusionMixin:
    @staticmethod
    def _build_fusion_projection_info(
        *,
        pane_role: str,
        source_projection: FusionSourceProjection | None,
        image_transform: Any,
        image_width: int,
        image_height: int,
    ) -> FusionProjectionInfo | None:
        if source_projection is None or image_width <= 0 or image_height <= 0:
            return None
        try:
            image_to_source = np.linalg.inv(np.asarray(image_transform.matrix, dtype=np.float64))
        except Exception:
            return None

        source_to_world_origin = np.asarray(source_projection.source_to_world_origin, dtype=np.float64)
        source_to_world_x = np.asarray(source_projection.source_to_world_x, dtype=np.float64)
        source_to_world_y = np.asarray(source_projection.source_to_world_y, dtype=np.float64)

        def source_to_world(source_x: float, source_y: float) -> np.ndarray:
            return source_to_world_origin + source_to_world_x * float(source_x) + source_to_world_y * float(source_y)

        def image_to_world(image_x: float, image_y: float) -> np.ndarray:
            source = image_to_source @ np.asarray([float(image_x), float(image_y), 1.0], dtype=np.float64)
            return source_to_world(float(source[0]), float(source[1]))

        normalized_origin = image_to_world(0.0, 0.0)
        normalized_x_world = image_to_world(float(image_width), 0.0) - normalized_origin
        normalized_y_world = image_to_world(0.0, float(image_height)) - normalized_origin

        source_from_world = np.asarray(
            [
                source_projection.world_to_source_x,
                source_projection.world_to_source_y,
                np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            ],
            dtype=np.float64,
        )
        image_from_world = np.asarray(image_transform.matrix, dtype=np.float64) @ source_from_world
        world_to_normalized_x = image_from_world[0] / float(image_width)
        world_to_normalized_y = image_from_world[1] / float(image_height)
        reference_world = np.asarray(source_projection.reference_world, dtype=np.float64)
        reference_homogeneous = np.asarray([*reference_world, 1.0], dtype=np.float64)
        reference_x = float(world_to_normalized_x @ reference_homogeneous)
        reference_y = float(world_to_normalized_y @ reference_homogeneous)

        def vector3(value: np.ndarray) -> tuple[float, float, float]:
            return (float(value[0]), float(value[1]), float(value[2]))

        def vector4(value: np.ndarray) -> tuple[float, float, float, float]:
            return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))

        return FusionProjectionInfo(
            paneRole=pane_role,
            referenceWorld=vector3(reference_world),
            referenceX=reference_x,
            referenceY=reference_y,
            normalizedToWorldOrigin=vector3(normalized_origin),
            normalizedToWorldX=vector3(normalized_x_world),
            normalizedToWorldY=vector3(normalized_y_world),
            worldToNormalizedX=vector4(world_to_normalized_x),
            worldToNormalizedY=vector4(world_to_normalized_y),
        )

    @staticmethod
    def _copy_fusion_registration_state(registration: FusionRegistrationState) -> FusionRegistrationState:
        return FusionRegistrationState(
            translate_row_mm=float(registration.translate_row_mm),
            translate_col_mm=float(registration.translate_col_mm),
            rotation_degrees=float(registration.rotation_degrees),
            saved=bool(registration.saved),
        )

    @staticmethod
    def _fusion_registration_visual_key(registration: FusionRegistrationState) -> tuple[float, float, float]:
        return (
            float(registration.translate_row_mm),
            float(registration.translate_col_mm),
            float(registration.rotation_degrees),
        )

    def _build_fusion_registration_pet_layer_cache_key(
        self,
        view: ViewRecord,
        group: ViewGroupRecord,
        ct_series: SeriesRecord,
        pet_series: SeriesRecord,
        registration: FusionRegistrationState,
    ) -> tuple[object, ...]:
        return (
            str(group.workspace_id),
            str(group.group_id),
            str(view.view_id),
            self._resolve_fusion_pane_role(view),
            str(ct_series.series_id),
            str(ct_series.volume_cache_key or ct_series.series_instance_uid or ""),
            str(pet_series.series_id),
            str(pet_series.volume_cache_key or pet_series.series_instance_uid or ""),
            int(group.fusion_axial_index),
            int(view.width or 0),
            int(view.height or 0),
            float(view.zoom),
            float(view.offset_x),
            float(view.offset_y),
            int(view.rotation_degrees),
            bool(view.hor_flip),
            bool(view.ver_flip),
            str(group.fusion_pet_unit),
            str(group.fusion_pet_pseudocolor_preset),
            None if group.fusion_pet_window.window_width is None else float(group.fusion_pet_window.window_width),
            None if group.fusion_pet_window.window_center is None else float(group.fusion_pet_window.window_center),
            float(group.fusion_alpha),
            self._fusion_registration_visual_key(registration),
        )

    @staticmethod
    def _build_fusion_registration_overlay_frame_lock_key(
        view: ViewRecord,
        group: ViewGroupRecord,
    ) -> tuple[str, str]:
        return str(group.group_id), str(view.view_id)

    def _clear_fusion_registration_overlay_frame_locks(
        self,
        group: ViewGroupRecord | None = None,
        *,
        view: ViewRecord | None = None,
    ) -> None:
        if group is None:
            self._fusion_registration_overlay_frame_locks.clear()
            return
        if view is not None:
            self._fusion_registration_overlay_frame_locks.pop(
                self._build_fusion_registration_overlay_frame_lock_key(view, group),
                None,
            )
            return
        group_id = str(group.group_id)
        for lock_key in [
            key for key in self._fusion_registration_overlay_frame_locks
            if key[0] == group_id
        ]:
            self._fusion_registration_overlay_frame_locks.pop(lock_key, None)

    def _lock_fusion_registration_overlay_frame(
        self,
        view: ViewRecord,
        group: ViewGroupRecord,
        frame: FusionRegistrationOverlayRenderFrame | None,
    ) -> None:
        if frame is None:
            return
        self._fusion_registration_overlay_frame_locks[
            self._build_fusion_registration_overlay_frame_lock_key(view, group)
        ] = frame

    def _get_locked_fusion_registration_overlay_frame(
        self,
        view: ViewRecord,
        group: ViewGroupRecord,
    ) -> FusionRegistrationOverlayRenderFrame | None:
        return self._fusion_registration_overlay_frame_locks.get(
            self._build_fusion_registration_overlay_frame_lock_key(view, group)
        )

    def _get_fusion_registration_pet_layer_cache(
        self,
        cache_key: tuple[object, ...],
    ) -> FusionRegistrationPetLayerCacheEntry | None:
        cached = self._fusion_registration_pet_layer_cache.get(cache_key)
        if cached is None:
            return None
        self._fusion_registration_pet_layer_cache.move_to_end(cache_key)
        return cached

    @staticmethod
    def _resolve_fusion_registration_image_center_canvas(image: Image.Image) -> tuple[float, float]:
        width, height = image.size
        return (max(float(width) / 2.0, 0.0), max(float(height) / 2.0, 0.0))

    @staticmethod
    def _project_fusion_pet_geometry_center_to_canvas(
        *,
        pet_geometry: VolumeGeometry,
        pet_shape: tuple[int, int, int],
        pet_plane: PlanePose | None,
        image_transform: AffineTransform,
    ) -> tuple[float, float] | None:
        if pet_plane is None:
            return None
        try:
            center_ijk = np.asarray(
                [
                    (float(pet_shape[0]) - 1.0) / 2.0,
                    (float(pet_shape[1]) - 1.0) / 2.0,
                    (float(pet_shape[2]) - 1.0) / 2.0,
                    1.0,
                ],
                dtype=np.float64,
            )
            center_world = np.asarray(pet_geometry.ijk_to_world, dtype=np.float64) @ center_ijk
            delta_world = center_world[:3] - np.asarray(pet_plane.center_world, dtype=np.float64)
            source_x = (
                float(np.dot(delta_world, np.asarray(pet_plane.col_world, dtype=np.float64)))
                / max(float(pet_plane.pixel_spacing_col_mm), 1e-6)
                + (float(pet_plane.output_shape[1]) - 1.0) / 2.0
            )
            source_y = (
                float(np.dot(delta_world, np.asarray(pet_plane.row_world, dtype=np.float64)))
                / max(float(pet_plane.pixel_spacing_row_mm), 1e-6)
                + (float(pet_plane.output_shape[0]) - 1.0) / 2.0
            )
            canvas_point = np.asarray(image_transform.matrix, dtype=np.float64) @ np.asarray(
                [source_x, source_y, 1.0],
                dtype=np.float64,
            )
            center_x = float(canvas_point[0])
            center_y = float(canvas_point[1])
            if np.isfinite(center_x) and np.isfinite(center_y):
                return center_x, center_y
        except Exception:
            logger.debug("failed to project fusion PET geometry center", exc_info=True)
        return None

    def _store_fusion_registration_pet_layer_cache(
        self,
        cache_key: tuple[object, ...],
        *,
        image: Image.Image,
        slice_index: int,
        slice_total: int,
        pet_unit_label: str,
        canvas_mapping: FusionRegistrationCanvasMapping | None = None,
        overlay_plane: PlanePose | None = None,
        pet_center_canvas: tuple[float, float] | None = None,
    ) -> FusionRegistrationPetLayerCacheEntry:
        cached_image = image.convert("RGBA").copy()
        resolved_pet_center_canvas = (
            pet_center_canvas
            if pet_center_canvas is not None
            else self._resolve_fusion_registration_image_center_canvas(cached_image)
        )
        overlay_frame = (
            FusionRegistrationOverlayRenderFrame(
                plane=overlay_plane,
                cache_key=cache_key,
                canvas_mapping=canvas_mapping,
                pet_center_canvas=resolved_pet_center_canvas,
            )
            if overlay_plane is not None
            else None
        )
        cached_entry = FusionRegistrationPetLayerCacheEntry(
            image=cached_image,
            slice_index=int(slice_index),
            slice_total=max(1, int(slice_total)),
            pet_unit_label=str(pet_unit_label),
            canvas_mapping=canvas_mapping,
            overlay_frame=overlay_frame,
            pet_center_canvas=resolved_pet_center_canvas,
        )
        self._fusion_registration_pet_layer_cache[cache_key] = cached_entry
        self._fusion_registration_pet_layer_cache.move_to_end(cache_key)
        while len(self._fusion_registration_pet_layer_cache) > FUSION_REGISTRATION_PET_LAYER_CACHE_MAX_ITEMS:
            self._fusion_registration_pet_layer_cache.popitem(last=False)
        return cached_entry

    @staticmethod
    def _build_fusion_registration_canvas_mapping(
        *,
        source_projection: FusionSourceProjection | None,
        image_transform: Any,
        row_world: np.ndarray | None,
        col_world: np.ndarray | None,
    ) -> FusionRegistrationCanvasMapping | None:
        if source_projection is None or row_world is None or col_world is None:
            return None
        try:
            image_to_source = np.linalg.inv(np.asarray(image_transform.matrix, dtype=np.float64))
            source_to_world_origin = np.asarray(source_projection.source_to_world_origin, dtype=np.float64)
            source_to_world_x = np.asarray(source_projection.source_to_world_x, dtype=np.float64)
            source_to_world_y = np.asarray(source_projection.source_to_world_y, dtype=np.float64)
            reference_world = np.asarray(source_projection.reference_world, dtype=np.float64)
            row_direction = np.asarray(row_world, dtype=np.float64)
            col_direction = np.asarray(col_world, dtype=np.float64)

            def canvas_to_col_row(canvas_x: float, canvas_y: float) -> tuple[float, float]:
                source = image_to_source @ np.asarray([float(canvas_x), float(canvas_y), 1.0], dtype=np.float64)
                world = (
                    source_to_world_origin
                    + source_to_world_x * float(source[0])
                    + source_to_world_y * float(source[1])
                )
                delta_world = world - reference_world
                col_mm = float(np.dot(delta_world, col_direction))
                row_mm = float(np.dot(delta_world, row_direction))
                return col_mm, row_mm

            origin_col, origin_row = canvas_to_col_row(0.0, 0.0)
            x_col, x_row = canvas_to_col_row(1.0, 0.0)
            y_col, y_row = canvas_to_col_row(0.0, 1.0)
            col_coefficients = (
                float(x_col - origin_col),
                float(y_col - origin_col),
                float(origin_col),
            )
            row_coefficients = (
                float(x_row - origin_row),
                float(y_row - origin_row),
                float(origin_row),
            )
            if all(np.isfinite(value) for value in (*col_coefficients, *row_coefficients)):
                return FusionRegistrationCanvasMapping(
                    col_mm_from_canvas=col_coefficients,
                    row_mm_from_canvas=row_coefficients,
                )
        except Exception:
            logger.debug("failed to build fusion registration canvas mapping", exc_info=True)
        return None

    @staticmethod
    def _map_fusion_registration_canvas_point_with_mapping(
        mapping: FusionRegistrationCanvasMapping,
        *,
        canvas_x: float,
        canvas_y: float,
    ) -> tuple[float, float]:
        col = mapping.col_mm_from_canvas
        row = mapping.row_mm_from_canvas
        col_mm = float(col[0]) * float(canvas_x) + float(col[1]) * float(canvas_y) + float(col[2])
        row_mm = float(row[0]) * float(canvas_x) + float(row[1]) * float(canvas_y) + float(row[2])
        return row_mm, col_mm

    @staticmethod
    def _map_fusion_registration_canvas_delta_with_mapping(
        mapping: FusionRegistrationCanvasMapping,
        *,
        delta_x: float,
        delta_y: float,
    ) -> tuple[float, float]:
        col = mapping.col_mm_from_canvas
        row = mapping.row_mm_from_canvas
        col_mm = float(col[0]) * float(delta_x) + float(col[1]) * float(delta_y)
        row_mm = float(row[0]) * float(delta_x) + float(row[1]) * float(delta_y)
        return row_mm, col_mm

    @staticmethod
    def _fusion_pet_standalone_fill_color(image: Image.Image) -> int | tuple[int, int, int] | tuple[int, int, int, int]:
        if image.mode == "RGBA":
            return (255, 255, 255, 255)
        if image.mode == "RGB":
            return (255, 255, 255)
        return 255

    @staticmethod
    def _translate_fusion_registration_preview_image(
        image: Image.Image,
        dx: int,
        dy: int,
        *,
        fillcolor: object | None = None,
    ) -> Image.Image:
        width, height = image.size
        if fillcolor is None:
            fillcolor = (0, 0, 0, 0) if image.mode == "RGBA" else 0
        result = Image.new(image.mode, (width, height), fillcolor)
        copy_width = width - abs(int(dx))
        copy_height = height - abs(int(dy))
        if copy_width <= 0 or copy_height <= 0:
            return result
        source_left = max(0, -int(dx))
        source_top = max(0, -int(dy))
        target_left = max(0, int(dx))
        target_top = max(0, int(dy))
        crop = image.crop((source_left, source_top, source_left + copy_width, source_top + copy_height))
        result.paste(crop, (target_left, target_top))
        return result

    @staticmethod
    def _build_fusion_registration_preview_transform(drag: FusionRegistrationPreviewDrag) -> AffineTransform:
        if drag.sub_op_type == "rotate":
            radians = np.deg2rad(float(drag.rotation_delta_degrees))
            cos_theta = float(np.cos(radians))
            sin_theta = float(np.sin(radians))
            pivot_x = float(drag.pivot_x)
            pivot_y = float(drag.pivot_y)
            translate_to_origin = np.asarray(
                [
                    [1.0, 0.0, -pivot_x],
                    [0.0, 1.0, -pivot_y],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            rotate = np.asarray(
                [
                    [cos_theta, -sin_theta, 0.0],
                    [sin_theta, cos_theta, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            translate_back = np.asarray(
                [
                    [1.0, 0.0, pivot_x],
                    [0.0, 1.0, pivot_y],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            return AffineTransform(matrix=translate_back @ rotate @ translate_to_origin)

        return AffineTransform(
            matrix=np.asarray(
                [
                    [1.0, 0.0, float(drag.delta_x)],
                    [0.0, 1.0, float(drag.delta_y)],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
        )

    @staticmethod
    def _with_fusion_registration_preview_rotation_center(
        drag: FusionRegistrationPreviewDrag,
        pet_center_canvas: tuple[float, float] | None,
    ) -> FusionRegistrationPreviewDrag:
        del pet_center_canvas
        return drag

    def _apply_fusion_registration_preview_transform(
        self,
        image: Image.Image,
        drag: FusionRegistrationPreviewDrag,
        *,
        fillcolor: object | None = None,
    ) -> Image.Image:
        width, height = image.size
        if width <= 0 or height <= 0:
            return image.copy()

        if drag.sub_op_type != "rotate":
            dx = float(drag.delta_x)
            dy = float(drag.delta_y)
            rounded_dx = int(round(dx))
            rounded_dy = int(round(dy))
            if abs(dx - rounded_dx) <= 1e-3 and abs(dy - rounded_dy) <= 1e-3:
                if rounded_dx == 0 and rounded_dy == 0:
                    return image.copy()
                return self._translate_fusion_registration_preview_image(
                    image,
                    rounded_dx,
                    rounded_dy,
                    fillcolor=fillcolor,
                )

        if drag.sub_op_type == "rotate" and abs(float(drag.rotation_delta_degrees)) <= 1e-6:
            return image.copy()

        return compat.viewport_transformer.apply_affine(
            image,
            int(width),
            int(height),
            self._build_fusion_registration_preview_transform(drag),
            resample=Image.Resampling.BILINEAR,
            fillcolor=fillcolor,
        )

    def _build_fusion_registration_layer_preview_result(
        self,
        view: ViewRecord,
        group: ViewGroupRecord,
        ct_series: SeriesRecord,
        pet_series: SeriesRecord,
        *,
        pet_image: Image.Image,
        image_format: ImageFormat,
        slice_index: int,
        slice_total: int,
        pet_unit_label: str,
        render_started_at: float,
        cache_hit: bool,
        transform_ms: float | None = None,
    ) -> RenderedImageResult:
        canvas_width, canvas_height = pet_image.size
        registration_info = FusionRegistrationInfo(
            translateRowMm=float(group.fusion_registration.translate_row_mm),
            translateColMm=float(group.fusion_registration.translate_col_mm),
            rotationDegrees=float(group.fusion_registration.rotation_degrees),
            saved=bool(group.fusion_registration.saved),
        )
        fusion_composite = FusionCompositeInfo(
            revision=int(group.fusion_revision),
            alpha=float(group.fusion_alpha),
            registration=registration_info,
            width=int(canvas_width),
            height=int(canvas_height),
            layers=[FusionCompositeLayerInfo(key="pet", role="pet", imageFormat="png")],
            primary_image_unchanged=True,
        )
        pet_encode_started_at = perf_counter()
        pet_bytes = self._encode_image(pet_image, "png", fast_preview=False)
        pet_encode_ms = (perf_counter() - pet_encode_started_at) * 1000.0
        extra_image_bytes = {
            "pet": pet_bytes
        }
        logger.info(
            (
                "fusion registration preview layer view_id=%s role=%s cache_hit=%s "
                "render=%sx%s transform_ms=%s pet_encode_ms=%.1f total_ms=%.1f pet_bytes=%s"
            ),
            view.view_id,
            FUSION_PANE_OVERLAY_AXIAL,
            cache_hit,
            canvas_width,
            canvas_height,
            None if transform_ms is None else round(float(transform_ms), 1),
            pet_encode_ms,
            (perf_counter() - render_started_at) * 1000.0,
            len(pet_bytes),
        )
        return RenderedImageResult(
            meta=ViewImageResponse(
                slice_info=SliceInfo(current=int(slice_index) + 1, total=max(1, int(slice_total))),
                window_info=WindowInfo(ww=view.window_width, wl=view.window_center),
                imageFormat=image_format,
                viewId=view.view_id,
                transform=self._build_view_transform_payload(view),
                color=ViewColorInfo(pseudocolorPreset=group.fusion_pet_pseudocolor_preset),
                fusionInfo=FusionInfo(
                    paneRole=FUSION_PANE_OVERLAY_AXIAL,
                    ctSeriesId=ct_series.series_id,
                    petSeriesId=pet_series.series_id,
                    petPseudocolorPreset=group.fusion_pet_pseudocolor_preset,
                    petUnit=group.fusion_pet_unit,
                    petUnitLabel=pet_unit_label,
                    petWindowMin=self._resolve_window_min(
                        group.fusion_pet_window.window_width,
                        group.fusion_pet_window.window_center,
                    ),
                    petWindowMax=self._resolve_window_max(
                        group.fusion_pet_window.window_width,
                        group.fusion_pet_window.window_center,
                    ),
                    alpha=float(group.fusion_alpha),
                    revision=int(group.fusion_revision),
                    registration=registration_info,
                ),
                fusionComposite=fusion_composite,
            ),
            image_bytes=self._fusion_registration_transparent_primary_png,
            extra_image_bytes=extra_image_bytes,
        )

    def _build_fusion_registration_primary_preview_result(
        self,
        view: ViewRecord,
        group: ViewGroupRecord,
        ct_series: SeriesRecord,
        pet_series: SeriesRecord,
        *,
        role: str,
        image: Image.Image,
        image_format: ImageFormat,
        slice_index: int,
        slice_total: int,
        pet_unit_label: str,
        render_started_at: float,
        cache_hit: bool,
        transform_ms: float | None = None,
    ) -> RenderedImageResult:
        registration_info = FusionRegistrationInfo(
            translateRowMm=float(group.fusion_registration.translate_row_mm),
            translateColMm=float(group.fusion_registration.translate_col_mm),
            rotationDegrees=float(group.fusion_registration.rotation_degrees),
            saved=bool(group.fusion_registration.saved),
        )
        encode_started_at = perf_counter()
        image_bytes = self._encode_image(image, image_format, fast_preview=False)
        encode_ms = (perf_counter() - encode_started_at) * 1000.0
        logger.info(
            (
                "fusion registration preview primary view_id=%s role=%s cache_hit=%s "
                "render=%sx%s transform_ms=%s encode_ms=%.1f total_ms=%.1f bytes=%s"
            ),
            view.view_id,
            role,
            cache_hit,
            image.width,
            image.height,
            None if transform_ms is None else round(float(transform_ms), 1),
            encode_ms,
            (perf_counter() - render_started_at) * 1000.0,
            len(image_bytes),
        )
        return RenderedImageResult(
            meta=ViewImageResponse(
                slice_info=SliceInfo(current=int(slice_index) + 1, total=max(1, int(slice_total))),
                window_info=WindowInfo(ww=view.window_width, wl=view.window_center),
                imageFormat=image_format,
                viewId=view.view_id,
                transform=self._build_view_transform_payload(view),
                color=ViewColorInfo(
                    pseudocolorPreset=(
                        FUSION_PET_STANDALONE_PSEUDOCOLOR_PRESET
                        if role == FUSION_PANE_PET_AXIAL
                        else group.fusion_pet_pseudocolor_preset
                    )
                ),
                fusionInfo=FusionInfo(
                    paneRole=role,
                    ctSeriesId=ct_series.series_id,
                    petSeriesId=pet_series.series_id,
                    petPseudocolorPreset=group.fusion_pet_pseudocolor_preset,
                    petUnit=group.fusion_pet_unit,
                    petUnitLabel=pet_unit_label,
                    petWindowMin=self._resolve_window_min(
                        group.fusion_pet_window.window_width,
                        group.fusion_pet_window.window_center,
                    ),
                    petWindowMax=self._resolve_window_max(
                        group.fusion_pet_window.window_width,
                        group.fusion_pet_window.window_center,
                    ),
                    alpha=float(group.fusion_alpha),
                    revision=int(group.fusion_revision),
                    registration=registration_info,
                ),
            ),
            image_bytes=image_bytes,
        )

    def _try_render_cached_fusion_registration_layer_preview(
        self,
        view: ViewRecord,
        group: ViewGroupRecord,
        ct_series: SeriesRecord,
        pet_series: SeriesRecord,
        *,
        image_format: ImageFormat,
        render_started_at: float,
    ) -> RenderedImageResult | None:
        drag = self._fusion_registration_preview_drags.get(group.group_id)
        if drag is None:
            return None
        role = self._resolve_fusion_pane_role(view)
        if role not in {FUSION_PANE_OVERLAY_AXIAL, FUSION_PANE_PET_AXIAL}:
            return None
        cache_key = self._build_fusion_registration_pet_layer_cache_key(
            view,
            group,
            ct_series,
            pet_series,
            drag.origin_registration,
        )
        cached = self._get_fusion_registration_pet_layer_cache(cache_key)
        if cached is None:
            logger.info(
                "fusion registration preview cache miss view_id=%s group_id=%s role=%s",
                view.view_id,
                group.group_id,
                role,
            )
            return None
        self._lock_fusion_registration_overlay_frame(view, group, cached.overlay_frame)
        transform_started_at = perf_counter()
        preview_drag = self._with_fusion_registration_preview_rotation_center(
            drag,
            cached.pet_center_canvas,
        )
        preview_fillcolor = (
            self._fusion_pet_standalone_fill_color(cached.image)
            if role == FUSION_PANE_PET_AXIAL
            else None
        )
        transformed_pet = self._apply_fusion_registration_preview_transform(
            cached.image,
            preview_drag,
            fillcolor=preview_fillcolor,
        )
        transform_ms = (perf_counter() - transform_started_at) * 1000.0
        if role == FUSION_PANE_PET_AXIAL:
            return self._build_fusion_registration_primary_preview_result(
                view,
                group,
                ct_series,
                pet_series,
                role=role,
                image=transformed_pet,
                image_format=image_format,
                slice_index=cached.slice_index,
                slice_total=cached.slice_total,
                pet_unit_label=cached.pet_unit_label,
                render_started_at=render_started_at,
                cache_hit=True,
                transform_ms=transform_ms,
            )
        return self._build_fusion_registration_layer_preview_result(
            view,
            group,
            ct_series,
            pet_series,
            pet_image=transformed_pet,
            image_format=image_format,
            slice_index=cached.slice_index,
            slice_total=cached.slice_total,
            pet_unit_label=cached.pet_unit_label,
            render_started_at=render_started_at,
            cache_hit=True,
            transform_ms=transform_ms,
        )

    def _render_fusion_view(
        self,
        view: ViewRecord,
        image_format: ImageFormat = "webp",
        *,
        fast_preview: bool = False,
        fast_preview_full_resolution: bool = False,
        metadata_mode: str = "full",
        progress_callback: ViewRenderProgressCallback | None = None,
    ) -> RenderedImageResult:
        render_started_at = perf_counter()
        ensure_view_size(view)
        if not view.is_initialized:
            self._initialize_fusion_viewport(view)

        group, ct_series, pet_series = self._resolve_fusion_group_series(view)
        role = self._resolve_fusion_pane_role(view)
        registration_preview = (
            fast_preview
            and metadata_mode == "fusion-registration-layer-preview"
            and role in {FUSION_PANE_OVERLAY_AXIAL, FUSION_PANE_PET_AXIAL}
        )
        primary_image_unchanged = registration_preview and role == FUSION_PANE_OVERLAY_AXIAL
        self._sync_fusion_view_state_from_group(view)
        if registration_preview:
            cached_preview = self._try_render_cached_fusion_registration_layer_preview(
                view,
                group,
                ct_series,
                pet_series,
                image_format=image_format,
                render_started_at=render_started_at,
            )
            if cached_preview is not None:
                return cached_preview

        preview_volume_ms: float | None = None
        preview_fusion_ms: float | None = None
        preview_pet_canvas_ms: float | None = None
        preview_transform_ms: float | None = None
        preview_pet_encode_ms: float | None = None
        preview_pet_bytes: int | None = None
        preview_volume_started_at = perf_counter() if primary_image_unchanged else None
        ct_volume = self._get_series_volume(ct_series, progress_callback=progress_callback)
        pet_volume = self._get_series_volume(pet_series, progress_callback=progress_callback)
        pet_display = self._build_fusion_pet_display_volume(pet_series, pet_volume, group.fusion_pet_unit)
        ct_transform = self._get_series_patient_transform(ct_series)
        pet_transform = self._get_series_patient_transform(pet_series)
        ct_geometry = self._get_series_volume_geometry(ct_series, ct_volume.shape)
        pet_geometry = self._get_series_volume_geometry(pet_series, pet_volume.shape)
        if preview_volume_started_at is not None:
            preview_volume_ms = (perf_counter() - preview_volume_started_at) * 1000.0
        registration_drag = self._fusion_registration_preview_drags.get(group.group_id)
        preview_drag = registration_drag if registration_preview else None
        render_registration = preview_drag.origin_registration if preview_drag is not None else group.fusion_registration
        locked_overlay_frame = (
            self._resolve_fusion_registration_overlay_render_frame(
                view,
                group,
                ct_series,
                pet_series,
                registration_drag.origin_registration,
            )
            if role == FUSION_PANE_OVERLAY_AXIAL and registration_drag is not None
            else None
        )
        overlay_plane_override = (
            locked_overlay_frame.plane
            if primary_image_unchanged and locked_overlay_frame is not None
            else None
        )
        if (
            role == FUSION_PANE_OVERLAY_AXIAL
            and registration_drag is not None
            and locked_overlay_frame is None
            and primary_image_unchanged
        ):
            logger.warning(
                "fusion registration locked overlay frame missing view_id=%s group_id=%s; using current overlay plane",
                view.view_id,
                group.group_id,
            )
        self._emit_render_progress(progress_callback, "render", progress_percent=82)

        preview_fusion_started_at = perf_counter() if primary_image_unchanged else None
        fusion_result = render_fusion_pixels(
            pane_role=role,
            ct_volume=ct_volume,
            ct_geometry=ct_geometry,
            pet_volume=pet_display.volume,
            pet_geometry=pet_geometry,
            axial_index=group.fusion_axial_index,
            ct_window_width=group.window.window_width,
            ct_window_center=group.window.window_center,
            pet_window_width=group.fusion_pet_window.window_width,
            pet_window_center=group.fusion_pet_window.window_center,
            pet_pseudocolor_preset=group.fusion_pet_pseudocolor_preset,
            registration=render_registration,
            alpha=group.fusion_alpha,
            ct_has_patient_geometry=(
                ct_transform is not None
                and tuple(int(value) for value in ct_transform.shape)
                == tuple(int(value) for value in ct_volume.shape)
            ),
            pet_has_patient_geometry=(
                pet_transform is not None
                and tuple(int(value) for value in pet_transform.shape)
                == tuple(int(value) for value in pet_volume.shape)
            ),
            interpolation_order=0 if fast_preview and not fast_preview_full_resolution else 1,
            overlay_pet_layer_only=primary_image_unchanged,
            overlay_plane_override=overlay_plane_override,
        )
        if preview_fusion_started_at is not None:
            preview_fusion_ms = (perf_counter() - preview_fusion_started_at) * 1000.0
        source_image = image_from_pixels(fusion_result.pixels)
        pixel_aspect_x, pixel_aspect_y = self._get_display_aspect_xy_from_spacing(fusion_result.spacing_xy)
        render_plan = self._build_render_plan_for_shape(
            view,
            source_image.height,
            source_image.width,
            pixel_aspect_x=pixel_aspect_x,
            pixel_aspect_y=pixel_aspect_y,
        )
        image_transform = compat.viewport_transformer.build_image_to_canvas_transform(
            image_width=source_image.width,
            image_height=source_image.height,
            canvas_width=render_plan.render_view.width or 0,
            canvas_height=render_plan.render_view.height or 0,
            view=render_plan.render_view,
            pixel_aspect_x=pixel_aspect_x,
            pixel_aspect_y=pixel_aspect_y,
        )
        interpolation_order = 0 if fast_preview else 1
        canvas_width = render_plan.render_view.width or 0
        canvas_height = render_plan.render_view.height or 0
        fusion_composite: FusionCompositeInfo | None = None
        extra_image_bytes: dict[str, bytes] = {}
        pet_standalone_primary = role in {FUSION_PANE_PET_AXIAL, FUSION_PANE_PET_CORONAL_MIP}
        if (
            role == FUSION_PANE_OVERLAY_AXIAL
            and fusion_result.pet_layer_pixels is not None
            and (primary_image_unchanged or fusion_result.ct_layer_pixels is not None)
        ):
            preview_pet_canvas_started_at = perf_counter() if primary_image_unchanged else None
            transformed_pet = compat.viewport_transformer.apply_affine_array(
                fusion_result.pet_layer_pixels,
                canvas_width,
                canvas_height,
                image_transform,
                order=interpolation_order,
                cval=0.0,
            )
            if preview_pet_canvas_started_at is not None:
                preview_pet_canvas_ms = (perf_counter() - preview_pet_canvas_started_at) * 1000.0
            transformed_pet_image = image_from_pixels(transformed_pet)
            cache_key = self._build_fusion_registration_pet_layer_cache_key(
                view,
                group,
                ct_series,
                pet_series,
                render_registration,
            )
            canvas_mapping = self._build_fusion_registration_canvas_mapping(
                source_projection=fusion_result.source_projection,
                image_transform=image_transform,
                row_world=fusion_result.row_world,
                col_world=fusion_result.col_world,
            )
            pet_center_canvas = self._project_fusion_pet_geometry_center_to_canvas(
                pet_geometry=pet_geometry,
                pet_shape=tuple(int(value) for value in pet_display.volume.shape),
                pet_plane=fusion_result.pet_plane_pose,
                image_transform=image_transform,
            )
            cached_entry = self._store_fusion_registration_pet_layer_cache(
                cache_key,
                image=transformed_pet_image,
                slice_index=fusion_result.slice_index,
                slice_total=fusion_result.slice_total,
                pet_unit_label=pet_display.unit_label,
                canvas_mapping=canvas_mapping,
                overlay_plane=fusion_result.plane_pose,
                pet_center_canvas=pet_center_canvas,
            )
            self._lock_fusion_registration_overlay_frame(view, group, cached_entry.overlay_frame)
            if primary_image_unchanged and preview_drag is not None:
                preview_transform_started_at = perf_counter()
                preview_drag_for_transform = self._with_fusion_registration_preview_rotation_center(
                    preview_drag,
                    cached_entry.pet_center_canvas,
                )
                transformed_pet_image = self._apply_fusion_registration_preview_transform(
                    transformed_pet_image,
                    preview_drag_for_transform,
                )
                preview_transform_ms = (perf_counter() - preview_transform_started_at) * 1000.0
            elif not primary_image_unchanged:
                self._fusion_registration_preview_drags.pop(group.group_id, None)
            preview_pet_encode_started_at = perf_counter() if primary_image_unchanged else None
            extra_image_bytes["pet"] = self._encode_image(transformed_pet_image, "png", fast_preview=False)
            if preview_pet_encode_started_at is not None:
                preview_pet_encode_ms = (perf_counter() - preview_pet_encode_started_at) * 1000.0
                preview_pet_bytes = len(extra_image_bytes["pet"])
            if primary_image_unchanged:
                image = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
            else:
                transformed_ct = compat.viewport_transformer.apply_affine_array(
                    fusion_result.ct_layer_pixels,
                    canvas_width,
                    canvas_height,
                    image_transform,
                    order=interpolation_order,
                    cval=0.0,
                )
                image = image_from_pixels(transformed_ct)
        else:
            transformed = compat.viewport_transformer.apply_affine_array(
                np.asarray(source_image),
                canvas_width,
                canvas_height,
                image_transform,
                order=interpolation_order,
                cval=FUSION_PET_STANDALONE_BACKGROUND_CVAL if pet_standalone_primary else 0.0,
            )
            image = image_from_pixels(transformed)
            if role == FUSION_PANE_PET_AXIAL:
                cache_key = self._build_fusion_registration_pet_layer_cache_key(
                    view,
                    group,
                    ct_series,
                    pet_series,
                    render_registration,
                )
                pet_center_canvas = self._project_fusion_pet_geometry_center_to_canvas(
                    pet_geometry=pet_geometry,
                    pet_shape=tuple(int(value) for value in pet_display.volume.shape),
                    pet_plane=fusion_result.pet_plane_pose,
                    image_transform=image_transform,
                )
                cached_entry = self._store_fusion_registration_pet_layer_cache(
                    cache_key,
                    image=image,
                    slice_index=fusion_result.slice_index,
                    slice_total=fusion_result.slice_total,
                    pet_unit_label=pet_display.unit_label,
                    canvas_mapping=None,
                    overlay_plane=fusion_result.plane_pose,
                    pet_center_canvas=pet_center_canvas,
                )
                self._lock_fusion_registration_overlay_frame(view, group, cached_entry.overlay_frame)
                if registration_preview and preview_drag is not None:
                    preview_transform_started_at = perf_counter()
                    preview_drag_for_transform = self._with_fusion_registration_preview_rotation_center(
                        preview_drag,
                        cached_entry.pet_center_canvas,
                    )
                    image = self._apply_fusion_registration_preview_transform(
                        image,
                        preview_drag_for_transform,
                        fillcolor=self._fusion_pet_standalone_fill_color(image),
                    )
                    preview_transform_ms = (perf_counter() - preview_transform_started_at) * 1000.0
        fusion_projection = self._build_fusion_projection_info(
            pane_role=role,
            source_projection=fusion_result.source_projection,
            image_transform=image_transform,
            image_width=image.width,
            image_height=image.height,
        )
        scale_bar = self._build_scale_bar_info(render_plan.render_view, image_transform, fusion_result.spacing_xy)
        orientation_overlay = self._build_direction_orientation_overlay(
            render_plan.render_view,
            fusion_result.row_world,
            fusion_result.col_world,
        )
        corner_series = pet_series if role in {FUSION_PANE_PET_AXIAL, FUSION_PANE_PET_CORONAL_MIP} else ct_series
        viewport_label = self._build_fusion_corner_viewport_label(role)
        corner_instance, corner_cached = self._get_indexed_instance_and_cache(corner_series, fusion_result.slice_index)
        corner_info = (
            self._build_slice_corner_info_overlay(
                view,
                corner_series,
                corner_cached.dataset,
                current_index=fusion_result.slice_index,
                total_slices=fusion_result.slice_total,
                viewport_label=viewport_label,
                show_physical_location=role != FUSION_PANE_PET_CORONAL_MIP,
                show_image_index=role != FUSION_PANE_PET_CORONAL_MIP,
            )
            if corner_instance is not None and corner_cached is not None
            else None
        )
        if corner_info is not None and role in {FUSION_PANE_PET_AXIAL, FUSION_PANE_PET_CORONAL_MIP}:
            corner_info = self._with_pet_window_corner_info(
                corner_info,
                pet_display,
                group.fusion_pet_window.window_width,
                group.fusion_pet_window.window_center,
            )
        include_fusion_annotation_payloads = not (
            fast_preview
            and metadata_mode in {"mpr-pixel-preview", "stack-pixel-preview", "fusion-registration-layer-preview"}
        )
        visible_annotations = self._build_visible_annotations(view) if include_fusion_annotation_payloads else ()
        registration_info = FusionRegistrationInfo(
            translateRowMm=float(group.fusion_registration.translate_row_mm),
            translateColMm=float(group.fusion_registration.translate_col_mm),
            rotationDegrees=float(group.fusion_registration.rotation_degrees),
            saved=bool(group.fusion_registration.saved),
        )
        if extra_image_bytes:
            fusion_composite = FusionCompositeInfo(
                revision=int(group.fusion_revision),
                alpha=float(group.fusion_alpha),
                registration=registration_info,
                width=int(canvas_width if primary_image_unchanged else image.width),
                height=int(canvas_height if primary_image_unchanged else image.height),
                layers=[
                    *([] if primary_image_unchanged else [FusionCompositeLayerInfo(key="primary", role="ct", imageFormat=image_format)]),
                    FusionCompositeLayerInfo(key="pet", role="pet", imageFormat="png"),
                ],
                primary_image_unchanged=primary_image_unchanged,
            )
        self._emit_render_progress(progress_callback, "encode", progress_percent=96)
        image_bytes = (
            self._fusion_registration_transparent_primary_png
            if primary_image_unchanged
            else self._encode_image(image, image_format, fast_preview=fast_preview)
        )
        logger.debug(
            "fusion render timing view_id=%s role=%s fast_preview=%s image_format=%s source_shape=%s render=%sx%s total_ms=%.1f",
            view.view_id,
            role,
            fast_preview,
            image_format,
            tuple(int(value) for value in fusion_result.pixels.shape[:2]),
            render_plan.render_view.width,
            render_plan.render_view.height,
            (perf_counter() - render_started_at) * 1000.0,
        )
        if primary_image_unchanged:
            logger.info(
                (
                    "fusion registration preview fallback view_id=%s role=%s cache_hit=False "
                    "render=%sx%s volume_ms=%s fusion_ms=%s pet_canvas_ms=%s "
                    "preview_transform_ms=%s pet_encode_ms=%s total_ms=%.1f pet_bytes=%s"
                ),
                view.view_id,
                role,
                canvas_width,
                canvas_height,
                None if preview_volume_ms is None else round(preview_volume_ms, 1),
                None if preview_fusion_ms is None else round(preview_fusion_ms, 1),
                None if preview_pet_canvas_ms is None else round(preview_pet_canvas_ms, 1),
                None if preview_transform_ms is None else round(preview_transform_ms, 1),
                None if preview_pet_encode_ms is None else round(preview_pet_encode_ms, 1),
                (perf_counter() - render_started_at) * 1000.0,
                preview_pet_bytes,
            )
        return RenderedImageResult(
            meta=ViewImageResponse(
                slice_info=SliceInfo(current=fusion_result.slice_index + 1, total=max(1, fusion_result.slice_total)),
                window_info=WindowInfo(ww=view.window_width, wl=view.window_center),
                imageFormat=image_format,
                viewId=view.view_id,
                scaleBar=scale_bar,
                cornerInfo=self._serialize_corner_info_overlay(corner_info) if corner_info is not None else None,
                orientation=self._serialize_orientation_overlay(orientation_overlay),
                transform=self._build_view_transform_payload(view),
                color=ViewColorInfo(pseudocolorPreset=fusion_result.pseudocolor_preset),
                annotations=[] if not include_fusion_annotation_payloads else self._serialize_annotations(
                    visible_annotations,
                    image_transform=image_transform,
                    canvas_width=render_plan.render_view.width or 0,
                    canvas_height=render_plan.render_view.height or 0,
                ),
                fusionProjection=fusion_projection,
                fusionInfo=FusionInfo(
                    paneRole=role,
                    ctSeriesId=ct_series.series_id,
                    petSeriesId=pet_series.series_id,
                    petPseudocolorPreset=group.fusion_pet_pseudocolor_preset,
                    petUnit=pet_display.unit,
                    petUnitLabel=pet_display.unit_label,
                    petWindowMin=self._resolve_window_min(
                        group.fusion_pet_window.window_width,
                        group.fusion_pet_window.window_center,
                    ),
                    petWindowMax=self._resolve_window_max(
                        group.fusion_pet_window.window_width,
                        group.fusion_pet_window.window_center,
                    ),
                    alpha=float(group.fusion_alpha),
                    revision=int(group.fusion_revision),
                    registration=registration_info,
                ),
                fusionComposite=fusion_composite,
            ),
            image_bytes=image_bytes,
            extra_image_bytes=extra_image_bytes,
        )
