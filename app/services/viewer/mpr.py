from __future__ import annotations

"""MPR rendering, segmentation, VOI, and geometry."""

from app.services.viewer.shared import *  # noqa: F403


class ViewerMprMixin:
    def _render_mpr_view(
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

        series = compat.series_registry.get(view.series_id)
        self._emit_render_progress(progress_callback, "volume", progress_percent=6)
        volume_started_at = perf_counter()
        volume = self._get_series_volume(series, progress_callback=progress_callback)
        volume_ms = (perf_counter() - volume_started_at) * 1000.0
        if not view.is_initialized:
            self._emit_render_progress(progress_callback, "initialize", progress_percent=72)
            self._initialize_mpr_viewport(view)
            view.is_initialized = True

        target_viewport = self._resolve_mpr_viewport(view)
        self._emit_render_progress(progress_callback, "render", progress_percent=82)
        preview_plane_shape = (
            self._get_mpr_fast_preview_plane_shape(
                volume.shape,
                target_viewport,
                viewport_size=(view.height or 0, view.width or 0),
            )
            if fast_preview and not fast_preview_full_resolution
            else None
        )
        reslice_started_at = perf_counter()
        plane_pixels, current, total = self._extract_mpr_plane(
            view,
            volume,
            target_viewport,
            output_shape=preview_plane_shape,
            interpolation_order=0 if fast_preview and not fast_preview_full_resolution else 1,
        )
        reslice_ms = (perf_counter() - reslice_started_at) * 1000.0
        metadata_started_at = perf_counter()
        payload_pose_context = self._build_mpr_pose_context(view, volume.shape, series=series)
        target_plane_pose = payload_pose_context.poses[target_viewport]
        segmentation_plane_pose = self._pose_for_sampled_mpr_plane(target_plane_pose, plane_pixels.shape[:2])
        plane_state = self._plane_state_from_pose(target_plane_pose) if view.view_group is not None else None
        pixel_aspect_x, pixel_aspect_y = self._get_mpr_display_aspect_xy_from_pose(target_plane_pose)
        full_plane_height, full_plane_width = target_plane_pose.output_shape
        source_plane_height, source_plane_width = plane_pixels.shape[:2]
        render_pixel_aspect_x = pixel_aspect_x * float(full_plane_width) / float(max(1, source_plane_width))
        render_pixel_aspect_y = pixel_aspect_y * float(full_plane_height) / float(max(1, source_plane_height))
        render_plan = self._build_render_plan_for_shape(
            view,
            *plane_pixels.shape[:2],
            pixel_aspect_x=render_pixel_aspect_x,
            pixel_aspect_y=render_pixel_aspect_y,
        )
        render_image_transform = compat.viewport_transformer.build_image_to_canvas_transform(
            image_width=plane_pixels.shape[1],
            image_height=plane_pixels.shape[0],
            canvas_width=render_plan.render_view.width or 0,
            canvas_height=render_plan.render_view.height or 0,
            view=render_plan.render_view,
            pixel_aspect_x=render_pixel_aspect_x,
            pixel_aspect_y=render_pixel_aspect_y,
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
        scale_bar = self._build_scale_bar_info(
            render_plan.render_view,
            metadata_image_transform,
            self._get_mpr_spacing_xy_from_pose(target_plane_pose),
        )
        plane_min = float(np.min(plane_pixels))
        plane_max = float(np.max(plane_pixels))
        mpr_crosshair_overlay = self._build_mpr_crosshair_overlay(
            render_plan.render_view,
            volume.shape,
            target_plane_pose.output_shape,
            metadata_image_transform,
        )
        include_static_preview_metadata = not (
            fast_preview and metadata_mode in {"mpr-pan-zoom-preview", "mpr-zoom-preview", "mpr-crosshair-preview"}
        )
        reference_instance, reference_cached = (
            (None, None)
            if fast_preview
            else self._get_reference_instance_and_cache(series)
        )
        slice_corner_info = (
            None
            if not include_static_preview_metadata
            else self._build_slice_corner_info_overlay(
                view,
                series,
                reference_cached.dataset if reference_cached is not None else None,
                current_index=current,
                total_slices=total,
                viewport_label=self._build_mpr_viewport_label(target_viewport, plane_state),
                plane_state=plane_state,
                plane_pose=target_plane_pose,
                cursor=payload_pose_context.cursor,
            )
        )
        include_mpr_measurement_payloads = not fast_preview or metadata_mode in {"mpr-pan-zoom-preview", "mpr-zoom-preview"}
        visible_measurements = self._build_visible_measurements(view) if include_mpr_measurement_payloads else []
        visible_annotations = self._build_visible_annotations(view) if include_mpr_measurement_payloads else []
        context = RenderContext(
            view=render_plan.render_view,
            source_pixels=plane_pixels,
            pixel_min=plane_min,
            pixel_max=plane_max,
            image_transform=render_image_transform,
            instance=reference_instance,
            cached=reference_cached,
            mpr_viewport=target_viewport,
            measurements=visible_measurements,
            mpr_crosshair=None,
            corner_info=None,
            orientation=None,
        )
        metadata_ms = (perf_counter() - metadata_started_at) * 1000.0
        image_started_at = perf_counter()
        if fast_preview:
            image = self._render_fast_mpr_preview(
                context,
                order=1 if fast_preview_full_resolution else 0,
            )
        else:
            image = compat.layered_renderer.render(context)
        include_mpr_segmentation_overlay = not fast_preview or metadata_mode == "mpr-segmentation-preview"
        mpr_segmentation_overlay = (
            self._build_mpr_segmentation_overlay_payload(
                plane_pixels,
                view.mpr_segmentation,
                target_viewport,
                segmentation_plane_pose,
                include_samples=not fast_preview or metadata_mode == "mpr-segmentation-preview",
                sample_limit=(
                    MPR_SEGMENTATION_OVERLAY_PREVIEW_SAMPLE_LIMIT
                    if fast_preview and metadata_mode == "mpr-segmentation-preview"
                    else MPR_SEGMENTATION_OVERLAY_SAMPLE_LIMIT
                ),
            )
            if include_mpr_segmentation_overlay
            else None
        )
        has_local_segmentation_samples = bool(
            mpr_segmentation_overlay
            and any(region.samples is not None for region in mpr_segmentation_overlay.regions)
        )
        if include_mpr_segmentation_overlay and not has_local_segmentation_samples:
            image = self._apply_mpr_segmentation_overlay(
                image,
                view.mpr_segmentation,
                plane_pixels,
                target_viewport,
                segmentation_plane_pose,
                render_image_transform,
                render_plan.render_view.width or 0,
                render_plan.render_view.height or 0,
            )
        image_ms = (perf_counter() - image_started_at) * 1000.0

        self._emit_render_progress(progress_callback, "encode", progress_percent=96)
        encode_started_at = perf_counter()
        image_bytes = self._encode_image(
            image,
            image_format,
            fast_preview=fast_preview and not fast_preview_full_resolution,
        )
        encode_ms = (perf_counter() - encode_started_at) * 1000.0
        logger.debug(
            "mpr render timing view_id=%s viewport=%s fast_preview=%s source_shape=%s full_shape=%s volume_ms=%.1f reslice_ms=%.1f metadata_ms=%.1f image_ms=%.1f encode_ms=%.1f total_ms=%.1f",
            view.view_id,
            target_viewport,
            fast_preview,
            plane_pixels.shape,
            target_plane_pose.output_shape,
            volume_ms,
            reslice_ms,
            metadata_ms,
            image_ms,
            encode_ms,
            (perf_counter() - render_started_at) * 1000.0,
        )

        return RenderedImageResult(
            meta=ViewImageResponse(
                slice_info=SliceInfo(current=current, total=total),
                window_info=WindowInfo(ww=view.window_width, wl=view.window_center),
                imageFormat=image_format,
                viewId=view.view_id,
                color=ViewColorInfo(pseudocolorPreset=view.pseudocolor_preset),
                mprFrame=self._build_mpr_frame_payload(payload_pose_context.cursor, payload_pose_context.geometry),
                mprCursor=self._build_mpr_cursor_payload(payload_pose_context.cursor),
                mprRevision=self._get_mpr_revision(view.view_group),
                mprPlane=self._build_mpr_plane_payload(
                    view,
                    target_viewport,
                    plane_pose=target_plane_pose,
                    geometry=payload_pose_context.geometry,
                    image_transform=metadata_image_transform,
                ),
                mprMipConfig=self._serialize_mpr_mip_config(view.mpr_mip),
                mprSegmentationConfig=self._serialize_mpr_segmentation_config(view.mpr_segmentation),
                mprSegmentationOverlay=mpr_segmentation_overlay,
                mprCrosshairMode=self._get_mpr_crosshair_mode(view.view_group),
                mpr_crosshair=self._build_mpr_crosshair_info(mpr_crosshair_overlay),
                scaleBar=scale_bar,
                cornerInfo=self._serialize_corner_info_overlay(slice_corner_info) if slice_corner_info is not None else None,
                measurements=[] if not include_mpr_measurement_payloads else self._serialize_measurements(
                    visible_measurements,
                    image_transform=metadata_image_transform,
                    canvas_width=render_plan.render_view.width or 0,
                    canvas_height=render_plan.render_view.height or 0,
                ),
                annotations=[] if not include_mpr_measurement_payloads else self._serialize_annotations(
                    tuple(visible_annotations),
                    image_transform=metadata_image_transform,
                    canvas_width=render_plan.render_view.width or 0,
                    canvas_height=render_plan.render_view.height or 0,
                ),
                transform=self._build_view_transform_payload(view),
                orientation=None if not include_static_preview_metadata else self._serialize_orientation_overlay(
                    self._build_mpr_orientation_overlay(
                        render_plan.render_view,
                        target_viewport,
                        plane_state,
                        plane_pose=target_plane_pose,
                    )
                ),
            ),
            image_bytes=image_bytes,
        )

    def _render_fast_mpr_preview(self, context: RenderContext, *, order: int = 0) -> Image.Image:
        return self._render_cached_fast_base_image(context, order=order)

    def _render_fast_preview(self, context: RenderContext) -> Image.Image:
        image = self._render_cached_fast_base_image(context)
        if not compat.layered_renderer._has_overlay_content(context):
            return image
        return compat.layered_renderer.composite_overlays(image.convert("RGBA"), context)

    def _render_cached_fast_base_image(self, context: RenderContext, *, order: int = 1) -> Image.Image:
        base_pixels = self._get_cached_fast_base_pixels(context)
        if context.view.pseudocolor_preset != DEFAULT_PSEUDOCOLOR_PRESET:
            transformed_color = compat.viewport_transformer.apply_affine_array(
                apply_pseudocolor(base_pixels, context.view.pseudocolor_preset),
                context.view.width or 0,
                context.view.height or 0,
                context.image_transform,
                order=order,
                cval=context.background_cval,
            )
            return Image.fromarray(transformed_color)
        transformed = compat.viewport_transformer.apply_affine_array(
            base_pixels,
            context.view.width or 0,
            context.view.height or 0,
            context.image_transform,
            order=order,
            cval=context.background_cval,
        )
        return Image.fromarray(transformed)

    def _get_cached_fast_base_pixels(self, context: RenderContext) -> np.ndarray:
        cache_key = self._build_fast_base_pixels_cache_key(context)
        cached = self._fast_base_pixels_cache.get(cache_key)
        if cached is not None:
            self._fast_base_pixels_cache.move_to_end(cache_key)
            return cached

        base_pixels = self._window_array(
            context.source_pixels,
            context.view.window_width,
            context.view.window_center,
            pixel_min=context.pixel_min,
            pixel_max=context.pixel_max,
        )
        self._fast_base_pixels_cache[cache_key] = base_pixels
        self._fast_base_pixels_cache.move_to_end(cache_key)
        while len(self._fast_base_pixels_cache) > FAST_BASE_PIXELS_CACHE_MAX_ITEMS:
            self._fast_base_pixels_cache.popitem(last=False)
        return base_pixels

    @staticmethod
    def _build_fast_base_pixels_cache_key(context: RenderContext) -> tuple[object, ...]:
        return (
            id(context.source_pixels),
            tuple(context.source_pixels.shape),
            str(context.source_pixels.dtype),
            float(context.pixel_min),
            float(context.pixel_max),
            None if context.view.window_width is None else float(context.view.window_width),
            None if context.view.window_center is None else float(context.view.window_center),
        )

    @staticmethod
    def _render_fast_base_image(
        source_pixels: np.ndarray,
        pixel_min: float,
        pixel_max: float,
        render_view: ViewRecord,
        image_transform,
        *,
        order: int = 1,
    ) -> Image.Image:
        base_pixels = compat.ViewerService._window_array(
            source_pixels,
            render_view.window_width,
            render_view.window_center,
            pixel_min=pixel_min,
            pixel_max=pixel_max,
        )
        transformed = compat.viewport_transformer.apply_affine_array(
            base_pixels,
            render_view.width or 0,
            render_view.height or 0,
            image_transform,
            order=order,
            cval=0.0,
        )
        if render_view.pseudocolor_preset != DEFAULT_PSEUDOCOLOR_PRESET:
            transformed = apply_pseudocolor(transformed, render_view.pseudocolor_preset)
            return Image.fromarray(transformed)
        return Image.fromarray(transformed)

    def _build_render_plan_for_shape(
        self,
        view: ViewRecord,
        image_height: int,
        image_width: int,
        *,
        pixel_aspect_x: float = 1.0,
        pixel_aspect_y: float = 1.0,
    ) -> RenderPlan:
        render_ratio = self._resolve_render_ratio_for_shape(
            view,
            image_height,
            image_width,
            pixel_aspect_x=pixel_aspect_x,
            pixel_aspect_y=pixel_aspect_y,
        )
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
    def _resolve_render_ratio_for_shape(
        view: ViewRecord,
        image_height: int,
        image_width: int,
        *,
        pixel_aspect_x: float = 1.0,
        pixel_aspect_y: float = 1.0,
    ) -> float:
        if not view.width or not view.height:
            return 1.0

        physical_width = image_width * max(abs(float(pixel_aspect_x)), 1e-6)
        physical_height = image_height * max(abs(float(pixel_aspect_y)), 1e-6)
        if view.width <= physical_width or view.height <= physical_height:
            return 1.0

        contain_zoom = compat.viewport_transformer.calculate_contain_zoom(
            image_width=image_width,
            image_height=image_height,
            canvas_width=view.width,
            canvas_height=view.height,
            pixel_aspect_x=pixel_aspect_x,
            pixel_aspect_y=pixel_aspect_y,
        )
        if view.zoom > contain_zoom:
            return 1.0

        width_ratio = physical_width / view.width
        height_ratio = physical_height / view.height
        return max(width_ratio, height_ratio)

    @staticmethod
    def _get_mpr_plane_shape(volume_shape: tuple[int, int, int], viewport_key: str) -> tuple[int, int]:
        depth, height, width = volume_shape
        if viewport_key == MPR_VIEWPORT_CORONAL:
            return depth, width
        if viewport_key == MPR_VIEWPORT_SAGITTAL:
            return depth, height
        return height, width

    @staticmethod
    def _get_mpr_fast_preview_plane_shape(
        volume_shape: tuple[int, int, int],
        viewport_key: str,
        viewport_size: tuple[int, int] | None = None,
    ) -> tuple[int, int]:
        full_height, full_width = compat.ViewerService._get_mpr_plane_shape(volume_shape, viewport_key)
        viewport_height = int(viewport_size[0]) if viewport_size is not None else 0
        viewport_width = int(viewport_size[1]) if viewport_size is not None else 0

        def preview_dimension(value: int, viewport_value: int) -> int:
            if value <= MPR_FAST_PREVIEW_MIN_SIDE:
                return max(1, int(value))
            volume_scaled = max(MPR_FAST_PREVIEW_MIN_SIDE, int(round(float(value) * MPR_FAST_PREVIEW_SCALE)))
            if viewport_value > 0:
                viewport_scaled = max(
                    MPR_FAST_PREVIEW_MIN_SIDE,
                    int(round(float(viewport_value) * MPR_FAST_PREVIEW_SCALE)),
                )
                volume_scaled = min(volume_scaled, viewport_scaled)
            return min(
                int(value),
                volume_scaled,
            )

        return preview_dimension(full_height, viewport_height), preview_dimension(full_width, viewport_width)

    @staticmethod
    def _create_default_mpr_mip_state() -> MprMipState:
        return MprMipState()

    @staticmethod
    def _create_default_mpr_segmentation_state() -> MprSegmentationState:
        return MprSegmentationState()

    @staticmethod
    def _normalize_mpr_crosshair_mode(value: object) -> str:
        mode = str(value or "").strip().lower()
        return mode if mode in MPR_CROSSHAIR_MODES else MPR_CROSSHAIR_MODE_ORTHOGONAL

    @staticmethod
    def _get_mpr_crosshair_mode(group: ViewGroupRecord | None) -> str:
        return compat.ViewerService._normalize_mpr_crosshair_mode(
            group.mpr_crosshair_mode if group is not None else MPR_CROSSHAIR_MODE_ORTHOGONAL
        )

    @staticmethod
    def _get_mpr_revision(group: ViewGroupRecord | None) -> int | None:
        return int(group.mpr_revision) if group is not None else None

    @staticmethod
    def _bump_mpr_revision(group: ViewGroupRecord | None) -> int | None:
        if group is None:
            return None
        group.mpr_revision = max(0, int(group.mpr_revision)) + 1
        return group.mpr_revision

    @staticmethod
    def _normalize_plane_normal_record(value: object) -> tuple[float, float, float] | None:
        try:
            vector = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if vector.shape != (3,):
            return None
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 1e-6:
            return None
        return tuple(float(component) for component in vector / norm)

    def _get_independent_plane_normal_overrides(
        self,
        group: ViewGroupRecord | None,
    ) -> dict[str, tuple[float, float, float]]:
        if self._get_mpr_crosshair_mode(group) != MPR_CROSSHAIR_MODE_DOUBLE_OBLIQUE or group is None:
            return {}
        return {
            viewport_key: normal
            for viewport_key in (MPR_VIEWPORT_AXIAL, MPR_VIEWPORT_CORONAL, MPR_VIEWPORT_SAGITTAL)
            if (normal := self._normalize_plane_normal_record(group.mpr_independent_plane_normals.get(viewport_key))) is not None
        }

    def _derive_mpr_plane_pose(
        self,
        cursor: MprCursorState,
        viewport_key: str,
        geometry: VolumeGeometry,
        shape_policy: OutputShapePolicy,
        normal_overrides: dict[str, tuple[float, float, float]] | None = None,
        use_display_basis_for_cursor_offsets: bool = False,
    ) -> PlanePose:
        return derive_plane_pose(
            cursor,
            viewport_key,
            geometry,
            shape_policy,
            normal_world_override=(normal_overrides or {}).get(viewport_key),
            use_display_basis_for_cursor_offsets=use_display_basis_for_cursor_offsets,
        )

    def _build_mpr_plane_poses(
        self,
        cursor: MprCursorState,
        geometry: VolumeGeometry,
        volume_shape: tuple[int, int, int],
        *,
        normal_overrides: dict[str, tuple[float, float, float]] | None = None,
        use_display_basis_for_cursor_offsets: bool = False,
    ) -> dict[str, PlanePose]:
        shape_policy = OutputShapePolicy(
            viewport_shapes={
                viewport_key: self._get_mpr_plane_shape(volume_shape, viewport_key)
                for viewport_key in (MPR_VIEWPORT_AXIAL, MPR_VIEWPORT_CORONAL, MPR_VIEWPORT_SAGITTAL)
            }
        )
        return {
            viewport_key: self._derive_mpr_plane_pose(
                cursor,
                viewport_key,
                geometry,
                shape_policy,
                normal_overrides,
                use_display_basis_for_cursor_offsets=use_display_basis_for_cursor_offsets,
            )
            for viewport_key in (MPR_VIEWPORT_AXIAL, MPR_VIEWPORT_CORONAL, MPR_VIEWPORT_SAGITTAL)
        }

    @staticmethod
    def _normal_records_from_poses(poses: dict[str, PlanePose]) -> dict[str, tuple[float, float, float]]:
        return {
            viewport_key: tuple(float(value) for value in mpr_geometry.normalize_oblique_vector(
                pose.normal_world,
                fallback=(1.0, 0.0, 0.0),
            ))
            for viewport_key, pose in poses.items()
        }

    @staticmethod
    def _serialize_mpr_mip_config(state: MprMipState) -> MprMipConfig:
        return MprMipConfig(
            enabled=bool(state.enabled),
            algorithm=str(state.algorithm or "maximum"),
            viewports={
                viewport_key: MprMipViewportConfig(thickness=max(0, min(100, int(viewport_state.thickness))))
                for viewport_key, viewport_state in state.viewports.items()
            },
        )

    @staticmethod
    def _serialize_mpr_segmentation_config(state: MprSegmentationState) -> MprSegmentationConfig:
        def serialize_stats(stats: MprThresholdRegionStatsState | None) -> MprThresholdRegionStats | None:
            if stats is None:
                return None
            return MprThresholdRegionStats(
                huMean=stats.hu_mean,
                huMin=stats.hu_min,
                huMax=stats.hu_max,
                huStdDev=stats.hu_std_dev,
                volumeCm3=float(stats.volume_cm3),
                sampleCount=int(stats.sample_count),
                effectiveThresholdHu=stats.effective_threshold_hu,
            )

        def serialize_region(region: MprThresholdRegionState) -> MprThresholdRegion:
            return MprThresholdRegion(
                id=str(region.id),
                enabled=bool(region.enabled),
                label=str(region.label or ""),
                thresholdHu=float(region.threshold_hu),
                thresholdMode=str(region.threshold_mode or "hu"),
                thresholdPercentile=float(region.threshold_percentile),
                color=str(region.color or "#ff4df8"),
                box=MprThresholdRegionBox(
                    centerWorld=compat.ViewerService._vector_payload(region.box.center_world),
                    rowWorld=compat.ViewerService._vector_payload(region.box.row_world),
                    colWorld=compat.ViewerService._vector_payload(region.box.col_world),
                    normalWorld=compat.ViewerService._vector_payload(region.box.normal_world),
                    widthMm=float(region.box.width_mm),
                    heightMm=float(region.box.height_mm),
                    depthMm=float(region.box.depth_mm),
                    sourceViewport=str(region.box.source_viewport or MPR_VIEWPORT_AXIAL),
                ),
                stats=serialize_stats(region.stats),
            )

        def serialize_voi_stats(stats: MprVoiSphereStatsState | None) -> MprVoiSphereStats | None:
            if stats is None:
                return None
            return MprVoiSphereStats(
                huMean=stats.hu_mean,
                huMin=stats.hu_min,
                huMax=stats.hu_max,
                huStdDev=stats.hu_std_dev,
                volumeCm3=float(stats.volume_cm3),
                sampleCount=int(stats.sample_count),
            )

        def serialize_voi_sphere(sphere: MprVoiSphereState) -> MprVoiSphere:
            return MprVoiSphere(
                id=str(sphere.id or ""),
                label=str(sphere.label or ""),
                enabled=bool(sphere.enabled),
                centerWorld=compat.ViewerService._vector_payload(sphere.center_world),
                radiusMm=float(sphere.radius_mm),
                color=str(sphere.color or "#22d3ee"),
                stats=serialize_voi_stats(sphere.stats),
            )

        legacy_voi_box = state.voi_box
        voi_spheres = compat.ViewerService._get_mpr_voi_spheres(state)
        selected_voi_id = state.selected_voi_id if any(sphere.id == state.selected_voi_id for sphere in voi_spheres) else None
        selected_voi_sphere = next((sphere for sphere in voi_spheres if sphere.id == selected_voi_id), None)
        legacy_voi_sphere = selected_voi_sphere or (voi_spheres[0] if voi_spheres else None)
        return MprSegmentationConfig(
            enabled=bool(state.enabled),
            clientRevision=max(0, int(state.client_revision)),
            selectedRegionId=state.selected_region_id,
            selectedVoi=bool(selected_voi_id),
            selectedVoiId=selected_voi_id,
            thresholdRegions=[serialize_region(region) for region in state.threshold_regions],
            voiSpheres=[serialize_voi_sphere(sphere) for sphere in voi_spheres],
            voiSphere=None if legacy_voi_sphere is None else serialize_voi_sphere(legacy_voi_sphere),
            lowerHu=float(state.lower_hu),
            upperHu=float(state.upper_hu),
            opacity=float(state.opacity),
            color=str(state.color or "#ff4df8"),
            voiBox=None if legacy_voi_box is None else MprSegmentationVoiBox(
                xMin=float(legacy_voi_box.x_min),
                xMax=float(legacy_voi_box.x_max),
                yMin=float(legacy_voi_box.y_min),
                yMax=float(legacy_voi_box.y_max),
                zMin=float(legacy_voi_box.z_min),
                zMax=float(legacy_voi_box.z_max),
            ),
        )

    def _handle_mpr_segmentation_config(
        self,
        view: ViewRecord,
        payload: ViewOperationRequest,
        *,
        series: SeriesRecord | None = None,
        refresh_stats: bool = True,
    ) -> bool:
        if not self._is_mpr_view_type(view.view_type) or view.view_group is None:
            return False
        if payload.mpr_segmentation_config is None:
            return False
        next_state = self._normalize_mpr_segmentation_state(payload.mpr_segmentation_config)
        if refresh_stats:
            self._refresh_mpr_segmentation_stats_for_view(view, next_state, series=series)
        view.view_group.mpr_segmentation = next_state
        return True

    @classmethod
    def _normalize_mpr_segmentation_state(cls, config: MprSegmentationConfig) -> MprSegmentationState:
        lower_hu = cls._clamp_float(config.lower_hu, -1024.0, 3071.0, 300.0)
        upper_hu = cls._clamp_float(config.upper_hu, -1024.0, 3071.0, 3071.0)
        if lower_hu > upper_hu:
            lower_hu, upper_hu = upper_hu, lower_hu
        threshold_regions = [
            normalized
            for region in config.threshold_regions
            if (normalized := cls._normalize_mpr_threshold_region(region)) is not None
        ]
        selected_region_id = str(config.selected_region_id).strip() if config.selected_region_id else None
        if selected_region_id and not any(region.id == selected_region_id for region in threshold_regions):
            selected_region_id = threshold_regions[0].id if threshold_regions else None
        voi_spheres = cls._normalize_mpr_voi_spheres(config)
        selected_voi_id = str(config.selected_voi_id).strip() if config.selected_voi_id else None
        if selected_voi_id and not any(sphere.id == selected_voi_id for sphere in voi_spheres):
            selected_voi_id = None
        if selected_voi_id is None and config.selected_voi and voi_spheres:
            legacy_selected_id = str(getattr(config.voi_sphere, "id", "") or "").strip() if config.voi_sphere is not None else ""
            selected_voi_id = legacy_selected_id if any(sphere.id == legacy_selected_id for sphere in voi_spheres) else voi_spheres[0].id
        selected_voi = bool(selected_voi_id)
        selected_voi_sphere = next((sphere for sphere in voi_spheres if sphere.id == selected_voi_id), None)
        if selected_voi_id:
            selected_region_id = None
        legacy_enabled = (
            not threshold_regions
            and (
                config.lower_hu is not None
                or config.upper_hu is not None
                or config.voi_box is not None
            )
        )
        return MprSegmentationState(
            enabled=bool(config.enabled),
            client_revision=max(0, int(cls._clamp_float(config.client_revision, 0.0, float(2**31 - 1), 0.0))),
            selected_region_id=selected_region_id,
            selected_voi=selected_voi,
            selected_voi_id=selected_voi_id,
            threshold_regions=threshold_regions,
            voi_spheres=voi_spheres,
            voi_sphere=selected_voi_sphere or (voi_spheres[0] if voi_spheres else None),
            lower_hu=lower_hu,
            upper_hu=upper_hu,
            opacity=cls._clamp_float(config.opacity, 0.0, 1.0, 0.45),
            color=cls._normalize_mpr_segmentation_color(config.color),
            voi_box=cls._normalize_mpr_segmentation_voi_box(config.voi_box),
            legacy_enabled=legacy_enabled,
        )

    @classmethod
    def _normalize_mpr_threshold_region(
        cls,
        region: MprThresholdRegion | MprThresholdRegionState | None,
    ) -> MprThresholdRegionState | None:
        if region is None:
            return None
        region_id = str(getattr(region, "id", "") or "").strip()
        if not region_id:
            return None
        box = cls._normalize_mpr_threshold_region_box(getattr(region, "box", None))
        if box is None:
            return None
        return MprThresholdRegionState(
            id=region_id,
            enabled=bool(getattr(region, "enabled", True)),
            label=str(getattr(region, "label", "") or ""),
            threshold_hu=cls._clamp_float(getattr(region, "threshold_hu", 300.0), -1024.0, 3071.0, 300.0),
            threshold_mode=cls._normalize_mpr_threshold_mode(getattr(region, "threshold_mode", "hu")),
            threshold_percentile=cls._clamp_float(getattr(region, "threshold_percentile", 80.0), 0.0, 100.0, 80.0),
            color=cls._normalize_mpr_segmentation_color(getattr(region, "color", "#ff4df8"), fallback="#ff4df8"),
            box=box,
            stats=cls._normalize_mpr_threshold_region_stats(getattr(region, "stats", None)),
        )

    @classmethod
    def _normalize_mpr_threshold_region_box(
        cls,
        box: MprThresholdRegionBox | MprThresholdRegionBoxState | None,
    ) -> MprThresholdRegionBoxState | None:
        if box is None:
            return None
        return MprThresholdRegionBoxState(
            center_world=cls._normalize_mpr_vec3(getattr(box, "center_world", None), (0.0, 0.0, 0.0)),
            row_world=cls._normalize_world_unit_vector(getattr(box, "row_world", None), (0.0, 1.0, 0.0)),
            col_world=cls._normalize_world_unit_vector(getattr(box, "col_world", None), (0.0, 0.0, 1.0)),
            normal_world=cls._normalize_world_unit_vector(getattr(box, "normal_world", None), (1.0, 0.0, 0.0)),
            width_mm=cls._clamp_float(getattr(box, "width_mm", 1.0), 1e-3, 10000.0, 1.0),
            height_mm=cls._clamp_float(getattr(box, "height_mm", 1.0), 1e-3, 10000.0, 1.0),
            depth_mm=cls._clamp_float(getattr(box, "depth_mm", 1.0), 1e-3, 10000.0, 1.0),
            source_viewport=cls._normalize_mpr_viewport_key(getattr(box, "source_viewport", MPR_VIEWPORT_AXIAL)),
        )

    @classmethod
    def _normalize_mpr_threshold_region_stats(
        cls,
        stats: MprThresholdRegionStats | MprThresholdRegionStatsState | None,
    ) -> MprThresholdRegionStatsState | None:
        if stats is None:
            return None
        sample_count = int(cls._clamp_float(getattr(stats, "sample_count", 0), 0.0, float(2**31 - 1), 0.0))
        return MprThresholdRegionStatsState(
            hu_mean=cls._optional_finite_float(getattr(stats, "hu_mean", None)),
            hu_min=cls._optional_finite_float(getattr(stats, "hu_min", None)),
            hu_max=cls._optional_finite_float(getattr(stats, "hu_max", None)),
            hu_std_dev=cls._optional_finite_float(getattr(stats, "hu_std_dev", None)),
            volume_cm3=cls._clamp_float(getattr(stats, "volume_cm3", 0.0), 0.0, float("inf"), 0.0),
            sample_count=sample_count,
            effective_threshold_hu=cls._optional_finite_float(getattr(stats, "effective_threshold_hu", None)),
        )

    @classmethod
    def _normalize_mpr_voi_spheres(cls, config: MprSegmentationConfig) -> list[MprVoiSphereState]:
        raw_spheres: list[MprVoiSphere | MprVoiSphereState] = list(config.voi_spheres or [])
        if not raw_spheres and config.voi_sphere is not None:
            raw_spheres = [config.voi_sphere]
        normalized_spheres: list[MprVoiSphereState] = []
        used_ids: set[str] = set()
        for index, sphere in enumerate(raw_spheres, start=1):
            normalized = cls._normalize_mpr_voi_sphere(sphere, default_index=index)
            if normalized is None:
                continue
            base_id = normalized.id or f"voi-{index}"
            sphere_id = base_id
            suffix = 2
            while sphere_id in used_ids:
                sphere_id = f"{base_id}-{suffix}"
                suffix += 1
            normalized.id = sphere_id
            if not normalized.label:
                normalized.label = str(len(normalized_spheres) + 1)
            used_ids.add(sphere_id)
            normalized_spheres.append(normalized)
        return normalized_spheres

    @classmethod
    def _normalize_mpr_voi_sphere(
        cls,
        sphere: MprVoiSphere | MprVoiSphereState | None,
        *,
        default_index: int = 1,
    ) -> MprVoiSphereState | None:
        if sphere is None:
            return None
        sphere_id = str(getattr(sphere, "id", "") or "").strip() or f"voi-{default_index}"
        label = str(getattr(sphere, "label", "") or "").strip() or str(default_index)
        return MprVoiSphereState(
            id=sphere_id,
            label=label,
            enabled=bool(getattr(sphere, "enabled", True)),
            center_world=cls._normalize_mpr_vec3(getattr(sphere, "center_world", None), (0.0, 0.0, 0.0)),
            radius_mm=cls._clamp_float(getattr(sphere, "radius_mm", 10.0), 1e-3, 10000.0, 10.0),
            color=cls._normalize_mpr_segmentation_color(getattr(sphere, "color", "#22d3ee"), fallback="#22d3ee"),
            stats=cls._normalize_mpr_voi_sphere_stats(getattr(sphere, "stats", None)),
        )

    @classmethod
    def _normalize_mpr_voi_sphere_stats(
        cls,
        stats: MprVoiSphereStats | MprVoiSphereStatsState | None,
    ) -> MprVoiSphereStatsState | None:
        if stats is None:
            return None
        sample_count = int(cls._clamp_float(getattr(stats, "sample_count", 0), 0.0, float(2**31 - 1), 0.0))
        return MprVoiSphereStatsState(
            hu_mean=cls._optional_finite_float(getattr(stats, "hu_mean", None)),
            hu_min=cls._optional_finite_float(getattr(stats, "hu_min", None)),
            hu_max=cls._optional_finite_float(getattr(stats, "hu_max", None)),
            hu_std_dev=cls._optional_finite_float(getattr(stats, "hu_std_dev", None)),
            volume_cm3=cls._clamp_float(getattr(stats, "volume_cm3", 0.0), 0.0, float("inf"), 0.0),
            sample_count=sample_count,
        )

    @staticmethod
    def _normalize_mpr_threshold_mode(value: object) -> str:
        return "percentile" if str(value or "hu").strip().lower() == "percentile" else "hu"

    @staticmethod
    def _normalize_mpr_vec3(value: object, fallback: tuple[float, float, float]) -> tuple[float, float, float]:
        try:
            vector = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError):
            return fallback
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            return fallback
        return tuple(float(component) for component in vector)

    @classmethod
    def _normalize_world_unit_vector(cls, value: object, fallback: tuple[float, float, float]) -> tuple[float, float, float]:
        vector = np.asarray(cls._normalize_mpr_vec3(value, fallback), dtype=np.float64)
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 1e-6:
            return fallback
        return tuple(float(component) for component in (vector / norm))

    @staticmethod
    def _normalize_mpr_viewport_key(value: object) -> str:
        text = str(value or "").strip()
        if text in {MPR_VIEWPORT_AXIAL, MPR_VIEWPORT_CORONAL, MPR_VIEWPORT_SAGITTAL}:
            return text
        return MPR_VIEWPORT_AXIAL

    @classmethod
    def _normalize_mpr_segmentation_voi_box(
        cls,
        voi_box: MprSegmentationVoiBox | MprSegmentationVoiBoxState | None,
    ) -> MprSegmentationVoiBoxState | None:
        if voi_box is None:
            return None

        def axis_range(min_name: str, max_name: str) -> tuple[float, float]:
            lower = cls._clamp_float(getattr(voi_box, min_name, 0.0), 0.0, 1.0, 0.0)
            upper = cls._clamp_float(getattr(voi_box, max_name, 1.0), 0.0, 1.0, 1.0)
            if lower > upper:
                lower, upper = upper, lower
            return lower, upper

        x_min, x_max = axis_range("x_min", "x_max")
        y_min, y_max = axis_range("y_min", "y_max")
        z_min, z_max = axis_range("z_min", "z_max")
        return MprSegmentationVoiBoxState(
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            z_min=z_min,
            z_max=z_max,
        )

    @staticmethod
    def _normalize_mpr_segmentation_color(color: object, fallback: str = "#ff4df8") -> str:
        text = str(color or "").strip()
        if len(text) == 7 and text.startswith("#") and all(ch in "0123456789abcdefABCDEF" for ch in text[1:]):
            return text.lower()
        return fallback

    @staticmethod
    def _optional_finite_float(value: object) -> float | None:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        return numeric if np.isfinite(numeric) else None

    @staticmethod
    def _pose_for_sampled_mpr_plane(plane_pose: PlanePose, sampled_shape: tuple[int, int]) -> PlanePose:
        sampled_height = max(1, int(sampled_shape[0]))
        sampled_width = max(1, int(sampled_shape[1]))
        full_height = max(1, int(plane_pose.output_shape[0]))
        full_width = max(1, int(plane_pose.output_shape[1]))
        if (sampled_height, sampled_width) == (full_height, full_width):
            return plane_pose
        return replace(
            plane_pose,
            output_shape=(sampled_height, sampled_width),
            pixel_spacing_row_mm=float(plane_pose.pixel_spacing_row_mm) * float(full_height) / float(sampled_height),
            pixel_spacing_col_mm=float(plane_pose.pixel_spacing_col_mm) * float(full_width) / float(sampled_width),
        )

    @staticmethod
    def _get_mpr_voi_spheres(state: MprSegmentationState) -> list[MprVoiSphereState]:
        if state.voi_spheres:
            return state.voi_spheres
        return [state.voi_sphere] if state.voi_sphere is not None else []

    def _refresh_mpr_segmentation_stats_for_view(
        self,
        view: ViewRecord,
        state: MprSegmentationState,
        *,
        series: SeriesRecord | None = None,
    ) -> None:
        if not state.threshold_regions and not self._get_mpr_voi_spheres(state):
            return
        try:
            series_record = series if series is not None else compat.series_registry.get(view.series_id)
            if not getattr(series_record, "instances", None):
                return
            volume = self._get_series_volume(series_record)
            geometry = self._get_series_volume_geometry(series_record, volume.shape)
            self._refresh_mpr_segmentation_stats(state, volume, geometry)
        except Exception:
            logger.debug("failed to refresh MPR segmentation stats view_id=%s", view.view_id, exc_info=True)

    @classmethod
    def _refresh_mpr_segmentation_stats(
        cls,
        state: MprSegmentationState,
        volume: np.ndarray,
        geometry: VolumeGeometry,
    ) -> None:
        cls._refresh_mpr_segmentation_region_stats(state, volume, geometry)
        for sphere in cls._get_mpr_voi_spheres(state):
            sphere.stats = (
                cls._empty_mpr_voi_sphere_stats()
                if not sphere.enabled
                else cls._compute_mpr_voi_sphere_stats(volume, geometry, sphere)
            )

    @classmethod
    def _refresh_mpr_segmentation_region_stats(
        cls,
        state: MprSegmentationState,
        volume: np.ndarray,
        geometry: VolumeGeometry,
    ) -> None:
        if not state.threshold_regions:
            return
        for region in state.threshold_regions:
            region.stats = (
                cls._empty_mpr_threshold_region_stats()
                if not region.enabled
                else cls._compute_mpr_threshold_region_stats(volume, geometry, region)
            )

    @classmethod
    def _empty_mpr_threshold_region_stats(cls, effective_threshold_hu: float | None = None) -> MprThresholdRegionStatsState:
        return MprThresholdRegionStatsState(
            hu_mean=None,
            hu_min=None,
            hu_max=None,
            hu_std_dev=None,
            volume_cm3=0.0,
            sample_count=0,
            effective_threshold_hu=effective_threshold_hu,
        )

    @classmethod
    def _empty_mpr_voi_sphere_stats(cls) -> MprVoiSphereStatsState:
        return MprVoiSphereStatsState(
            hu_mean=None,
            hu_min=None,
            hu_max=None,
            hu_std_dev=None,
            volume_cm3=0.0,
            sample_count=0,
        )

    @staticmethod
    def _get_geometry_voxel_volume_mm3(geometry: VolumeGeometry) -> float:
        affine = np.asarray(geometry.ijk_to_world, dtype=np.float64)
        voxel_volume_mm3 = float(abs(np.linalg.det(affine[:3, :3])))
        if not np.isfinite(voxel_volume_mm3) or voxel_volume_mm3 <= 0.0:
            voxel_volume_mm3 = float(np.prod(np.asarray(geometry.spacing_hint_mm, dtype=np.float64)))
        if not np.isfinite(voxel_volume_mm3) or voxel_volume_mm3 <= 0.0:
            return 1.0
        return voxel_volume_mm3

    @classmethod
    def _get_mpr_threshold_region_effective_threshold_hu(cls, region: MprThresholdRegionState) -> float:
        if cls._normalize_mpr_threshold_mode(region.threshold_mode) == "percentile":
            stats_threshold = None if region.stats is None else region.stats.effective_threshold_hu
            if stats_threshold is not None and np.isfinite(stats_threshold):
                return float(stats_threshold)
        return cls._clamp_float(region.threshold_hu, -1024.0, 3071.0, 300.0)

    @classmethod
    def _compute_mpr_threshold_region_stats(
        cls,
        volume: np.ndarray,
        geometry: VolumeGeometry,
        region: MprThresholdRegionState,
    ) -> MprThresholdRegionStatsState:
        threshold_mode = cls._normalize_mpr_threshold_mode(region.threshold_mode)
        threshold_hu = cls._clamp_float(region.threshold_hu, -1024.0, 3071.0, 300.0)
        empty_stats = cls._empty_mpr_threshold_region_stats(threshold_hu)
        voxels = np.asarray(volume)
        if voxels.ndim != 3 or any(int(size) <= 0 for size in voxels.shape[:3]):
            return empty_stats

        box = region.box
        center = np.asarray(box.center_world, dtype=np.float64)
        row = np.asarray(box.row_world, dtype=np.float64)
        col = np.asarray(box.col_world, dtype=np.float64)
        normal = np.asarray(box.normal_world, dtype=np.float64)
        half_row = row * (float(box.height_mm) / 2.0)
        half_col = col * (float(box.width_mm) / 2.0)
        half_normal = normal * (float(box.depth_mm) / 2.0)
        corners_world = np.asarray(
            [
                center + row_sign * half_row + col_sign * half_col + normal_sign * half_normal
                for row_sign in (-1.0, 1.0)
                for col_sign in (-1.0, 1.0)
                for normal_sign in (-1.0, 1.0)
            ],
            dtype=np.float64,
        )
        try:
            corners_ijk = np.asarray([world_to_ijk_point(geometry, corner) for corner in corners_world], dtype=np.float64)
        except (TypeError, ValueError):
            return empty_stats
        if corners_ijk.shape != (8, 3) or not np.all(np.isfinite(corners_ijk)):
            return empty_stats

        shape = np.asarray(voxels.shape[:3], dtype=np.int64)
        min_index = np.maximum(0, np.floor(np.min(corners_ijk, axis=0) - 1.0).astype(np.int64))
        max_index = np.minimum(shape - 1, np.ceil(np.max(corners_ijk, axis=0) + 1.0).astype(np.int64))
        if bool(np.any(min_index > max_index)):
            return empty_stats

        affine = np.asarray(geometry.ijk_to_world, dtype=np.float64)
        voxel_volume_mm3 = cls._get_geometry_voxel_volume_mm3(geometry)

        sample_count = 0
        value_sum = 0.0
        value_sum_sq = 0.0
        hu_min: float | None = None
        hu_max: float | None = None
        inside_value_blocks: list[np.ndarray] = []
        block_depth = 16
        i_start = int(min_index[0])
        i_stop = int(max_index[0])
        j_start = int(min_index[1])
        j_stop = int(max_index[1])
        k_start = int(min_index[2])
        k_stop = int(max_index[2])

        for block_i_start in range(i_start, i_stop + 1, block_depth):
            block_i_stop = min(i_stop, block_i_start + block_depth - 1)
            block = np.asarray(
                voxels[block_i_start : block_i_stop + 1, j_start : j_stop + 1, k_start : k_stop + 1],
                dtype=np.float64,
            )
            if block.size == 0:
                continue
            indices = np.indices(block.shape, dtype=np.float64)
            ii = indices[0] + float(block_i_start)
            jj = indices[1] + float(j_start)
            kk = indices[2] + float(k_start)
            world_x = affine[0, 0] * ii + affine[0, 1] * jj + affine[0, 2] * kk + affine[0, 3]
            world_y = affine[1, 0] * ii + affine[1, 1] * jj + affine[1, 2] * kk + affine[1, 3]
            world_z = affine[2, 0] * ii + affine[2, 1] * jj + affine[2, 2] * kk + affine[2, 3]
            delta_x = world_x - center[0]
            delta_y = world_y - center[1]
            delta_z = world_z - center[2]
            row_distance = delta_x * row[0] + delta_y * row[1] + delta_z * row[2]
            col_distance = delta_x * col[0] + delta_y * col[1] + delta_z * col[2]
            normal_distance = delta_x * normal[0] + delta_y * normal[1] + delta_z * normal[2]
            inside_box = (
                (np.abs(row_distance) <= float(box.height_mm) / 2.0 + 1e-6)
                & (np.abs(col_distance) <= float(box.width_mm) / 2.0 + 1e-6)
                & (np.abs(normal_distance) <= float(box.depth_mm) / 2.0 + 1e-6)
            )
            finite_inside = inside_box & np.isfinite(block)
            if not bool(np.any(finite_inside)):
                continue
            inside_values = block[finite_inside]
            if threshold_mode == "percentile":
                inside_value_blocks.append(np.asarray(inside_values, dtype=np.float64))
                continue
            values = inside_values[inside_values > threshold_hu]
            if values.size <= 0:
                continue
            count = int(values.size)
            sample_count += count
            value_sum += float(np.sum(values, dtype=np.float64))
            value_sum_sq += float(np.sum(values * values, dtype=np.float64))
            block_min = float(np.min(values))
            block_max = float(np.max(values))
            hu_min = block_min if hu_min is None else min(hu_min, block_min)
            hu_max = block_max if hu_max is None else max(hu_max, block_max)

        effective_threshold_hu = threshold_hu
        if threshold_mode == "percentile":
            if not inside_value_blocks:
                return empty_stats
            inside_values = np.concatenate(inside_value_blocks)
            if inside_values.size <= 0:
                return empty_stats
            effective_threshold_hu = float(
                np.percentile(
                    inside_values,
                    cls._clamp_float(region.threshold_percentile, 0.0, 100.0, 80.0),
                )
            )
            values = inside_values[inside_values > effective_threshold_hu]
            sample_count = int(values.size)
            if sample_count > 0:
                value_sum = float(np.sum(values, dtype=np.float64))
                value_sum_sq = float(np.sum(values * values, dtype=np.float64))
                hu_min = float(np.min(values))
                hu_max = float(np.max(values))

        if sample_count <= 0:
            return cls._empty_mpr_threshold_region_stats(effective_threshold_hu)
        hu_mean = value_sum / float(sample_count)
        variance = max(0.0, value_sum_sq / float(sample_count) - hu_mean * hu_mean)
        return MprThresholdRegionStatsState(
            hu_mean=hu_mean,
            hu_min=hu_min,
            hu_max=hu_max,
            hu_std_dev=float(np.sqrt(variance)),
            volume_cm3=float(sample_count) * voxel_volume_mm3 / 1000.0,
            sample_count=sample_count,
            effective_threshold_hu=effective_threshold_hu,
        )

    @classmethod
    def _compute_mpr_voi_sphere_stats(
        cls,
        volume: np.ndarray,
        geometry: VolumeGeometry,
        sphere: MprVoiSphereState,
    ) -> MprVoiSphereStatsState:
        empty_stats = cls._empty_mpr_voi_sphere_stats()
        voxels = np.asarray(volume)
        if voxels.ndim != 3 or any(int(size) <= 0 for size in voxels.shape[:3]):
            return empty_stats

        center = np.asarray(sphere.center_world, dtype=np.float64)
        radius_mm = max(1e-6, float(sphere.radius_mm))
        corners_world = np.asarray(
            [
                center + np.asarray((x_sign * radius_mm, y_sign * radius_mm, z_sign * radius_mm), dtype=np.float64)
                for x_sign in (-1.0, 1.0)
                for y_sign in (-1.0, 1.0)
                for z_sign in (-1.0, 1.0)
            ],
            dtype=np.float64,
        )
        try:
            corners_ijk = np.asarray([world_to_ijk_point(geometry, corner) for corner in corners_world], dtype=np.float64)
        except (TypeError, ValueError):
            return empty_stats
        if corners_ijk.shape != (8, 3) or not np.all(np.isfinite(corners_ijk)):
            return empty_stats

        shape = np.asarray(voxels.shape[:3], dtype=np.int64)
        min_index = np.maximum(0, np.floor(np.min(corners_ijk, axis=0) - 1.0).astype(np.int64))
        max_index = np.minimum(shape - 1, np.ceil(np.max(corners_ijk, axis=0) + 1.0).astype(np.int64))
        if bool(np.any(min_index > max_index)):
            return empty_stats

        affine = np.asarray(geometry.ijk_to_world, dtype=np.float64)
        voxel_volume_mm3 = cls._get_geometry_voxel_volume_mm3(geometry)
        sample_count = 0
        value_sum = 0.0
        value_sum_sq = 0.0
        hu_min: float | None = None
        hu_max: float | None = None
        radius_sq = radius_mm * radius_mm
        block_depth = 16
        i_start = int(min_index[0])
        i_stop = int(max_index[0])
        j_start = int(min_index[1])
        j_stop = int(max_index[1])
        k_start = int(min_index[2])
        k_stop = int(max_index[2])

        for block_i_start in range(i_start, i_stop + 1, block_depth):
            block_i_stop = min(i_stop, block_i_start + block_depth - 1)
            block = np.asarray(
                voxels[block_i_start : block_i_stop + 1, j_start : j_stop + 1, k_start : k_stop + 1],
                dtype=np.float64,
            )
            if block.size == 0:
                continue
            indices = np.indices(block.shape, dtype=np.float64)
            ii = indices[0] + float(block_i_start)
            jj = indices[1] + float(j_start)
            kk = indices[2] + float(k_start)
            world_x = affine[0, 0] * ii + affine[0, 1] * jj + affine[0, 2] * kk + affine[0, 3]
            world_y = affine[1, 0] * ii + affine[1, 1] * jj + affine[1, 2] * kk + affine[1, 3]
            world_z = affine[2, 0] * ii + affine[2, 1] * jj + affine[2, 2] * kk + affine[2, 3]
            distance_sq = (
                (world_x - center[0]) * (world_x - center[0])
                + (world_y - center[1]) * (world_y - center[1])
                + (world_z - center[2]) * (world_z - center[2])
            )
            finite_inside = (distance_sq <= radius_sq + 1e-6) & np.isfinite(block)
            if not bool(np.any(finite_inside)):
                continue
            values = block[finite_inside]
            count = int(values.size)
            sample_count += count
            value_sum += float(np.sum(values, dtype=np.float64))
            value_sum_sq += float(np.sum(values * values, dtype=np.float64))
            block_min = float(np.min(values))
            block_max = float(np.max(values))
            hu_min = block_min if hu_min is None else min(hu_min, block_min)
            hu_max = block_max if hu_max is None else max(hu_max, block_max)

        if sample_count <= 0:
            return empty_stats
        hu_mean = value_sum / float(sample_count)
        variance = max(0.0, value_sum_sq / float(sample_count) - hu_mean * hu_mean)
        return MprVoiSphereStatsState(
            hu_mean=hu_mean,
            hu_min=hu_min,
            hu_max=hu_max,
            hu_std_dev=float(np.sqrt(variance)),
            volume_cm3=float(sample_count) * voxel_volume_mm3 / 1000.0,
            sample_count=sample_count,
        )

    @staticmethod
    def _project_mpr_voi_sphere_to_plane(
        sphere: MprVoiSphereState,
        plane_pose: PlanePose,
    ) -> dict[str, float | bool | tuple[float, float]]:
        center = np.asarray(sphere.center_world, dtype=np.float64)
        delta = center - np.asarray(plane_pose.center_world, dtype=np.float64)
        row_mm = float(np.dot(delta, np.asarray(plane_pose.row_world, dtype=np.float64)))
        col_mm = float(np.dot(delta, np.asarray(plane_pose.col_world, dtype=np.float64)))
        normal_mm = float(np.dot(delta, np.asarray(plane_pose.normal_world, dtype=np.float64)))
        radius_mm = max(1e-6, float(sphere.radius_mm))
        intersects = abs(normal_mm) <= radius_mm
        display_radius_mm = (
            float(np.sqrt(max(0.0, radius_mm * radius_mm - normal_mm * normal_mm)))
            if intersects
            else radius_mm
        )
        return {
            "centerMm": (row_mm, col_mm),
            "distanceToPlaneMm": normal_mm,
            "radiusMm": display_radius_mm,
            "intersects": bool(intersects),
        }

    @staticmethod
    def _clamp_float(value: object, minimum: float, maximum: float, fallback: float) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return fallback
        if not np.isfinite(numeric):
            return fallback
        return max(minimum, min(maximum, numeric))

    @classmethod
    def _apply_mpr_segmentation_overlay(
        cls,
        image: Image.Image,
        state: MprSegmentationState,
        source_pixels: np.ndarray,
        viewport_key: str,
        plane_pose: PlanePose | None,
        image_transform,
        canvas_width: int,
        canvas_height: int,
    ) -> Image.Image:
        if canvas_width <= 0 or canvas_height <= 0:
            return image
        masks = cls._build_mpr_segmentation_region_plane_masks(source_pixels, state, viewport_key, plane_pose)
        if not masks:
            return image

        pixels = np.asarray(image.convert("RGBA"), dtype=np.float32).copy()
        any_overlay = False
        for region_mask in masks:
            if region_mask.mask is None or not bool(np.any(region_mask.mask)):
                continue
            transformed_mask = compat.viewport_transformer.apply_affine_array(
                region_mask.mask.astype(np.uint8) * 255,
                int(canvas_width),
                int(canvas_height),
                image_transform,
                order=0,
                cval=0.0,
            )
            overlay_mask = cls._apply_segmentation_dot_pattern(transformed_mask > 0)
            if not bool(np.any(overlay_mask)):
                continue
            any_overlay = True
            color = np.asarray(cls._parse_hex_rgb(region_mask.color), dtype=np.float32)
            alpha = 0.88
            pixels[overlay_mask, :3] = pixels[overlay_mask, :3] * (1.0 - alpha) + color * alpha
            pixels[overlay_mask, 3] = 255.0
        if not any_overlay:
            return image
        return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))

    @classmethod
    def _build_mpr_segmentation_plane_mask(
        cls,
        source_pixels: np.ndarray,
        state: MprSegmentationState,
        viewport_key: str,
        plane_pose: PlanePose | None = None,
    ) -> np.ndarray | None:
        masks = cls._build_mpr_segmentation_region_plane_masks(source_pixels, state, viewport_key, plane_pose)
        if not masks:
            return None
        combined = np.zeros(np.asarray(source_pixels).shape[:2], dtype=bool)
        for region_mask in masks:
            combined |= region_mask.mask
        return combined

    @classmethod
    def _build_mpr_segmentation_overlay_payload(
        cls,
        source_pixels: np.ndarray,
        state: MprSegmentationState,
        viewport_key: str,
        plane_pose: PlanePose | None = None,
        *,
        include_samples: bool = True,
        sample_limit: int = MPR_SEGMENTATION_OVERLAY_SAMPLE_LIMIT,
    ) -> MprSegmentationOverlay | None:
        if not state.enabled or not state.threshold_regions:
            return None
        pixels = np.asarray(source_pixels)
        if pixels.ndim >= 3:
            pixels = pixels[..., 0]
        plane_grid = (
            cls._build_mpr_threshold_plane_grid(plane_pose, pixels.shape[:2])
            if plane_pose is not None and pixels.ndim >= 2
            else None
        )
        masks = cls._build_mpr_segmentation_region_plane_masks(source_pixels, state, viewport_key, plane_pose)
        masks_by_region_id = {mask.region_id: mask.mask for mask in masks}
        regions: list[MprSegmentationOverlayRegion] = []
        for region in state.threshold_regions:
            mask = masks_by_region_id.get(str(region.id))
            rect = cls._build_mpr_segmentation_mask_rect(mask) if mask is not None else None
            samples: MprSegmentationOverlaySamples | None = None
            sample_revision = 0
            if region.enabled and plane_pose is not None and plane_grid is not None:
                geometry_mask = cls._build_mpr_threshold_region_plane_mask(
                    region,
                    plane_pose,
                    pixels.shape[:2],
                    plane_grid,
                )
                sample_revision = cls._build_mpr_segmentation_sample_revision(region, plane_pose, pixels.shape[:2])
                if include_samples:
                    samples = cls._build_mpr_segmentation_overlay_samples(
                        pixels,
                        geometry_mask,
                        sample_limit=sample_limit,
                    )
            regions.append(
                MprSegmentationOverlayRegion(
                    regionId=str(region.id),
                    visible=rect is not None,
                    rect=rect,
                    sampleRevision=sample_revision,
                    samples=samples,
                )
            )
        return MprSegmentationOverlay(regions=regions)

    @staticmethod
    def _build_mpr_segmentation_sample_revision(
        region: MprThresholdRegionState,
        plane_pose: PlanePose,
        shape: tuple[int, int],
    ) -> int:
        box = region.box

        def vector_payload(values: tuple[float, float, float] | np.ndarray) -> list[float]:
            return [round(float(value), 6) for value in values]

        payload = {
            "box": {
                "center": vector_payload(box.center_world),
                "row": vector_payload(box.row_world),
                "col": vector_payload(box.col_world),
                "normal": vector_payload(box.normal_world),
                "width": round(float(box.width_mm), 6),
                "height": round(float(box.height_mm), 6),
                "depth": round(float(box.depth_mm), 6),
                "sourceViewport": str(box.source_viewport or ""),
            },
            "plane": {
                "center": vector_payload(plane_pose.center_world),
                "row": vector_payload(plane_pose.row_world),
                "col": vector_payload(plane_pose.col_world),
                "normal": vector_payload(plane_pose.normal_world),
                "rowSpacing": round(float(plane_pose.pixel_spacing_row_mm), 6),
                "colSpacing": round(float(plane_pose.pixel_spacing_col_mm), 6),
            },
            "shape": [int(shape[0]), int(shape[1])],
        }
        encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        return int.from_bytes(hashlib.blake2b(encoded, digest_size=4).digest(), "big")

    @staticmethod
    def _build_mpr_segmentation_overlay_samples(
        pixels: np.ndarray,
        geometry_mask: np.ndarray,
        *,
        sample_limit: int = MPR_SEGMENTATION_OVERLAY_SAMPLE_LIMIT,
    ) -> MprSegmentationOverlaySamples | None:
        pixel_array = np.asarray(pixels)
        mask_array = np.asarray(geometry_mask, dtype=bool)
        if pixel_array.ndim != 2 or mask_array.ndim != 2 or pixel_array.shape[:2] != mask_array.shape[:2]:
            return None
        finite_mask = mask_array & np.isfinite(pixel_array)
        if not bool(np.any(finite_mask)):
            return None

        rows, cols = np.nonzero(finite_mask)
        total_count = int(rows.size)
        if total_count <= 0:
            return None

        resolved_sample_limit = max(1, int(sample_limit))
        if total_count > resolved_sample_limit:
            row_hash = rows.astype(np.uint64) * np.uint64(0x9E3779B185EBCA87)
            col_hash = cols.astype(np.uint64) * np.uint64(0xC2B2AE3D27D4EB4F)
            hashes = row_hash ^ col_hash ^ ((row_hash >> np.uint64(17)) + (col_hash << np.uint64(7)))
            selected = np.argpartition(hashes, resolved_sample_limit - 1)[:resolved_sample_limit]
            selected = selected[np.argsort(hashes[selected])]
            rows = rows[selected]
            cols = cols[selected]

        values = pixel_array[rows, cols].astype(np.float32, copy=False)
        points = np.empty(int(values.size) * 3, dtype=np.float32)
        points[0::3] = cols.astype(np.float32, copy=False) + np.float32(0.5)
        points[1::3] = rows.astype(np.float32, copy=False) + np.float32(0.5)
        points[2::3] = values
        return MprSegmentationOverlaySamples(
            points=points.tolist(),
            totalCount=total_count,
            sampledCount=int(values.size),
        )

    @staticmethod
    def _build_mpr_segmentation_mask_rect(mask: np.ndarray | None) -> MprSegmentationOverlayRect | None:
        if mask is None:
            return None
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.ndim != 2 or not bool(np.any(mask_array)):
            return None
        height, width = mask_array.shape[:2]
        if height <= 0 or width <= 0:
            return None
        rows, cols = np.where(mask_array)
        if rows.size <= 0 or cols.size <= 0:
            return None
        return MprSegmentationOverlayRect(
            xMin=max(0.0, min(1.0, float(np.min(cols)) / float(width))),
            yMin=max(0.0, min(1.0, float(np.min(rows)) / float(height))),
            xMax=max(0.0, min(1.0, float(np.max(cols) + 1) / float(width))),
            yMax=max(0.0, min(1.0, float(np.max(rows) + 1) / float(height))),
        )

    @classmethod
    def _build_mpr_segmentation_region_plane_masks(
        cls,
        source_pixels: np.ndarray,
        state: MprSegmentationState,
        viewport_key: str,
        plane_pose: PlanePose | None = None,
    ) -> list[MprThresholdPlaneMask]:
        if not state.enabled:
            return []
        pixels = np.asarray(source_pixels)
        if pixels.ndim < 2:
            return []
        if pixels.ndim == 3:
            pixels = pixels[..., 0]
        if state.threshold_regions and plane_pose is not None:
            masks: list[MprThresholdPlaneMask] = []
            plane_grid = cls._build_mpr_threshold_plane_grid(plane_pose, pixels.shape[:2])
            threshold_masks: dict[float, np.ndarray] = {}
            for region in state.threshold_regions:
                if not region.enabled:
                    continue
                region_mask = cls._build_mpr_threshold_region_plane_mask(region, plane_pose, pixels.shape[:2], plane_grid)
                if not bool(np.any(region_mask)):
                    continue
                threshold_hu = cls._get_mpr_threshold_region_effective_threshold_hu(region)
                threshold_mask = threshold_masks.get(threshold_hu)
                if threshold_mask is None:
                    threshold_mask = pixels > threshold_hu
                    threshold_masks[threshold_hu] = threshold_mask
                mask = threshold_mask & region_mask
                if bool(np.any(mask)):
                    masks.append(MprThresholdPlaneMask(region_id=str(region.id), mask=mask, color=region.color))
            return masks

        if not state.legacy_enabled:
            return []
        legacy_mask = cls._build_legacy_mpr_segmentation_plane_mask(pixels, state, viewport_key)
        return [] if legacy_mask is None else [MprThresholdPlaneMask(region_id="legacy", mask=legacy_mask, color=state.color)]

    @staticmethod
    def _build_mpr_threshold_plane_grid(
        plane_pose: PlanePose,
        shape: tuple[int, int],
    ) -> MprThresholdPlaneGrid:
        height, width = int(shape[0]), int(shape[1])
        row_offsets_mm = (np.arange(height, dtype=np.float64) - (float(height) - 1.0) / 2.0) * float(plane_pose.pixel_spacing_row_mm)
        col_offsets_mm = (np.arange(width, dtype=np.float64) - (float(width) - 1.0) / 2.0) * float(plane_pose.pixel_spacing_col_mm)
        col_grid_mm, row_grid_mm = np.meshgrid(col_offsets_mm, row_offsets_mm)
        return MprThresholdPlaneGrid(
            row_grid_mm=row_grid_mm,
            col_grid_mm=col_grid_mm,
            center_world=np.asarray(plane_pose.center_world, dtype=np.float64),
            row_world=np.asarray(plane_pose.row_world, dtype=np.float64),
            col_world=np.asarray(plane_pose.col_world, dtype=np.float64),
        )

    @classmethod
    def _build_legacy_mpr_segmentation_plane_mask(
        cls,
        pixels: np.ndarray,
        state: MprSegmentationState,
        viewport_key: str,
    ) -> np.ndarray | None:
        if state.opacity <= 0.0:
            return None
        lower_hu = cls._clamp_float(state.lower_hu, -1024.0, 3071.0, 300.0)
        upper_hu = cls._clamp_float(state.upper_hu, -1024.0, 3071.0, 3071.0)
        if lower_hu > upper_hu:
            lower_hu, upper_hu = upper_hu, lower_hu
        mask = (pixels >= lower_hu) & (pixels <= upper_hu)
        return cls._apply_voi_box_to_mpr_plane_mask(mask, state.voi_box, viewport_key)

    @classmethod
    def _build_mpr_threshold_region_plane_mask(
        cls,
        region: MprThresholdRegionState,
        plane_pose: PlanePose,
        shape: tuple[int, int],
        plane_grid: MprThresholdPlaneGrid | None = None,
    ) -> np.ndarray:
        height, width = int(shape[0]), int(shape[1])
        if height <= 0 or width <= 0:
            return np.zeros((max(0, height), max(0, width)), dtype=bool)
        grid = plane_grid or cls._build_mpr_threshold_plane_grid(plane_pose, (height, width))
        box = region.box
        delta_center = grid.center_world - np.asarray(box.center_world, dtype=np.float64)
        box_row = np.asarray(box.row_world, dtype=np.float64)
        box_col = np.asarray(box.col_world, dtype=np.float64)
        box_normal = np.asarray(box.normal_world, dtype=np.float64)

        row_distance = (
            float(np.dot(delta_center, box_row))
            + grid.row_grid_mm * float(np.dot(grid.row_world, box_row))
            + grid.col_grid_mm * float(np.dot(grid.col_world, box_row))
        )
        col_distance = (
            float(np.dot(delta_center, box_col))
            + grid.row_grid_mm * float(np.dot(grid.row_world, box_col))
            + grid.col_grid_mm * float(np.dot(grid.col_world, box_col))
        )
        normal_distance = (
            float(np.dot(delta_center, box_normal))
            + grid.row_grid_mm * float(np.dot(grid.row_world, box_normal))
            + grid.col_grid_mm * float(np.dot(grid.col_world, box_normal))
        )
        epsilon = 1e-6
        return (
            (np.abs(col_distance) <= float(box.width_mm) / 2.0 + epsilon)
            & (np.abs(row_distance) <= float(box.height_mm) / 2.0 + epsilon)
            & (np.abs(normal_distance) <= float(box.depth_mm) / 2.0 + epsilon)
        )

    @staticmethod
    def _apply_segmentation_dot_pattern(mask: np.ndarray) -> np.ndarray:
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.ndim != 2 or not bool(np.any(mask_array)):
            return np.zeros(mask_array.shape[:2], dtype=bool)
        sample_count = int(np.count_nonzero(mask_array))
        if sample_count <= 16:
            return mask_array
        height, width = mask_array.shape[:2]
        row_index, col_index = np.indices((height, width), dtype=np.uint32)
        # Hash in canvas space so zoom/flip transforms do not amplify source-space diagonal striping.
        hashed = (
            (row_index * np.uint32(0x45D9F3B))
            ^ (col_index * np.uint32(0x119DE1F3))
            ^ ((row_index + col_index) * np.uint32(0x27D4EB2D))
        )
        hashed ^= hashed >> np.uint32(15)
        hashed *= np.uint32(0x2C1B3C6D)
        hashed ^= hashed >> np.uint32(12)
        pattern = (hashed % np.uint32(100)) < np.uint32(52)
        dotted = mask_array & pattern
        if bool(np.any(dotted)):
            return dotted
        return mask_array

    @classmethod
    def _apply_voi_box_to_mpr_plane_mask(
        cls,
        mask: np.ndarray,
        voi_box: MprSegmentationVoiBoxState | None,
        viewport_key: str,
    ) -> np.ndarray:
        if voi_box is None:
            return mask.astype(bool, copy=False)

        height, width = mask.shape[:2]
        if viewport_key == MPR_VIEWPORT_CORONAL:
            horizontal_min, horizontal_max = voi_box.x_min, voi_box.x_max
            vertical_min, vertical_max = voi_box.z_min, voi_box.z_max
        elif viewport_key == MPR_VIEWPORT_SAGITTAL:
            horizontal_min, horizontal_max = voi_box.y_min, voi_box.y_max
            vertical_min, vertical_max = voi_box.z_min, voi_box.z_max
        else:
            horizontal_min, horizontal_max = voi_box.x_min, voi_box.x_max
            vertical_min, vertical_max = voi_box.y_min, voi_box.y_max

        col_start, col_end = cls._project_normalized_range_to_indices(horizontal_min, horizontal_max, width)
        row_start, row_end = cls._project_normalized_range_to_indices(vertical_min, vertical_max, height)
        if col_start >= col_end or row_start >= row_end:
            return np.zeros(mask.shape[:2], dtype=bool)

        voi_mask = np.zeros(mask.shape[:2], dtype=bool)
        voi_mask[row_start:row_end, col_start:col_end] = True
        return mask.astype(bool, copy=False) & voi_mask

    @classmethod
    def _project_normalized_range_to_indices(cls, minimum: float, maximum: float, size: int) -> tuple[int, int]:
        if size <= 0:
            return 0, 0
        lower = cls._clamp_float(minimum, 0.0, 1.0, 0.0)
        upper = cls._clamp_float(maximum, 0.0, 1.0, 1.0)
        if lower > upper:
            lower, upper = upper, lower
        start = int(np.floor(lower * size))
        end = int(np.ceil(upper * size))
        return max(0, min(size, start)), max(0, min(size, end))

    @staticmethod
    def _parse_hex_rgb(color: str) -> tuple[int, int, int]:
        normalized = compat.ViewerService._normalize_mpr_segmentation_color(color, fallback="#ff4df8")
        return (
            int(normalized[1:3], 16),
            int(normalized[3:5], 16),
            int(normalized[5:7], 16),
        )

    def _handle_mpr_mip_config(self, view: ViewRecord, payload: ViewOperationRequest) -> bool:
        if not self._is_mpr_view_type(view.view_type) or payload.mpr_mip_config is None:
            return False

        incoming = payload.mpr_mip_config
        current_state = view.mpr_mip
        next_viewports = dict(current_state.viewports)
        for viewport_key in (MPR_VIEWPORT_AXIAL, MPR_VIEWPORT_CORONAL, MPR_VIEWPORT_SAGITTAL):
            next_config = incoming.viewports.get(viewport_key)
            if next_config is None:
                next_viewports[viewport_key] = current_state.viewports.get(viewport_key, MprMipViewportState())
                continue
            next_viewports[viewport_key] = MprMipViewportState(thickness=max(0, min(100, int(next_config.thickness))))

        next_state = MprMipState(
            enabled=bool(incoming.enabled),
            algorithm=str(incoming.algorithm or "maximum"),
            viewports=next_viewports,
        )
        if view.view_group is not None:
            view.view_group.mpr_mip = next_state
        return True

    def _handle_mpr_crosshair_mode(self, view: ViewRecord, payload: ViewOperationRequest) -> bool:
        if not self._is_mpr_view_type(view.view_type) or view.view_group is None:
            return False
        if payload.mpr_crosshair_mode is None:
            return False
        next_mode = self._normalize_mpr_crosshair_mode(payload.mpr_crosshair_mode)
        group = view.view_group
        current_mode = self._get_mpr_crosshair_mode(group)
        if next_mode == current_mode:
            return False

        series = compat.series_registry.get(view.series_id)
        volume_shape = self._get_series_volume(series).shape
        pose_context = self._build_mpr_pose_context(view, volume_shape, series=series)
        group.active_viewport = self._resolve_mpr_viewport(view)
        group.rotation_drag = None

        if next_mode == MPR_CROSSHAIR_MODE_DOUBLE_OBLIQUE:
            group.mpr_crosshair_mode = MPR_CROSSHAIR_MODE_DOUBLE_OBLIQUE
            self._ensure_mpr_independent_plane_normals(group, pose_context.poses)
            group.mpr_crosshair_angles.clear()
            self._ensure_mpr_crosshair_angle_cache(group, pose_context.poses)
            view.is_initialized = True
            return True

        self._reorthogonalize_mpr_group_from_pose_context(group, pose_context, volume_shape)
        group.mpr_crosshair_mode = MPR_CROSSHAIR_MODE_ORTHOGONAL
        group.mpr_independent_plane_normals.clear()
        group.mpr_crosshair_angles.clear()
        group.rotation_drag = None
        view.is_initialized = True
        return True

    def _ensure_mpr_independent_plane_normals(
        self,
        group: ViewGroupRecord,
        poses: dict[str, PlanePose],
    ) -> None:
        next_normals = self._normal_records_from_poses(poses)
        for viewport_key in (MPR_VIEWPORT_AXIAL, MPR_VIEWPORT_CORONAL, MPR_VIEWPORT_SAGITTAL):
            existing_normal = self._normalize_plane_normal_record(group.mpr_independent_plane_normals.get(viewport_key))
            if existing_normal is not None:
                next_normals[viewport_key] = existing_normal
        group.mpr_independent_plane_normals = next_normals

    def _reorthogonalize_mpr_group_from_pose_context(
        self,
        group: ViewGroupRecord,
        pose_context: MprPoseContext,
        volume_shape: tuple[int, int, int],
    ) -> None:
        active_viewport = (
            group.active_viewport
            if group.active_viewport in (MPR_VIEWPORT_AXIAL, MPR_VIEWPORT_CORONAL, MPR_VIEWPORT_SAGITTAL)
            else MPR_VIEWPORT_AXIAL
        )
        active_plane = pose_context.poses[active_viewport]
        active_normal = np.asarray(active_plane.normal_world, dtype=np.float64)
        horizontal_angle, _ = self._get_mpr_visible_crosshair_line_angles(
            group,
            pose_context.poses,
            active_viewport,
        )
        horizontal_line_world = mpr_geometry.direction_from_screen_angle(
            np.asarray(active_plane.row_world, dtype=np.float64),
            np.asarray(active_plane.col_world, dtype=np.float64),
            horizontal_angle,
        )
        vertical_line_world = mpr_geometry.direction_from_screen_angle(
            np.asarray(active_plane.row_world, dtype=np.float64),
            np.asarray(active_plane.col_world, dtype=np.float64),
            horizontal_angle + float(np.pi / 2.0),
        )

        normal_updates: dict[str, np.ndarray] = {
            active_viewport: active_normal,
        }
        for line, line_world in (("horizontal", horizontal_line_world), ("vertical", vertical_line_world)):
            target_viewport = self._resolve_mpr_oblique_target_viewport(active_viewport, line)
            target_plane = pose_context.poses[target_viewport]
            next_normal = mpr_geometry.normalize_oblique_vector(
                np.cross(line_world, active_normal),
                fallback=tuple(target_plane.normal_world),
            )
            if float(np.dot(next_normal, np.asarray(target_plane.normal_world, dtype=np.float64))) < 0.0:
                next_normal = -next_normal
            normal_updates[target_viewport] = next_normal

        next_cursor = self._replace_mpr_cursor_plane_normals(pose_context.cursor, normal_updates)
        self._sync_group_from_mpr_cursor(group, next_cursor, pose_context.geometry, volume_shape)

    def _extract_mpr_plane(
        self,
        view: ViewRecord,
        volume: np.ndarray,
        viewport_key: str | None = None,
        output_shape: tuple[int, int] | None = None,
        interpolation_order: int = 1,
    ) -> tuple[np.ndarray, int, int]:
        target_viewport = viewport_key or self._resolve_mpr_viewport(view)
        full_plane_shape = self._get_mpr_plane_shape(volume.shape, target_viewport)
        effective_output_shape = tuple(int(value) for value in output_shape) if output_shape is not None else full_plane_shape
        cache_key = self._get_mpr_plane_cache_key(
            view,
            target_viewport,
            effective_output_shape,
            interpolation_order,
        )
        cached_plane = self._mpr_plane_cache.get(cache_key)
        if cached_plane is not None:
            self._mpr_plane_cache.move_to_end(cache_key)
            plane_pixels, current, total = cached_plane
            if target_viewport == MPR_VIEWPORT_AXIAL:
                view.current_index = current
            return plane_pixels, current, total

        try:
            series = compat.series_registry.get(view.series_id)
        except Exception:
            series = None
        geometry = self._get_series_volume_geometry(series, volume.shape) if series is not None else build_identity_geometry(volume.shape)
        cursor = self._get_mpr_cursor_state(view, geometry, volume.shape)
        plane_pose = self._derive_mpr_plane_pose(
            cursor,
            target_viewport,
            geometry,
            OutputShapePolicy(viewport_shapes={target_viewport: full_plane_shape}),
            self._get_independent_plane_normal_overrides(view.view_group),
            use_display_basis_for_cursor_offsets=self._should_use_mpr_display_basis_for_cursor_offsets(view.view_group),
        )
        if output_shape is not None and tuple(output_shape) != full_plane_shape:
            sample_height = max(1, int(output_shape[0]))
            sample_width = max(1, int(output_shape[1]))
            plane_pose = replace(
                plane_pose,
                output_shape=(sample_height, sample_width),
                pixel_spacing_row_mm=float(plane_pose.pixel_spacing_row_mm) * float(full_plane_shape[0]) / float(sample_height),
                pixel_spacing_col_mm=float(plane_pose.pixel_spacing_col_mm) * float(full_plane_shape[1]) / float(sample_width),
            )
        sampling_geometry = self._build_mpr_model_sampling_geometry(
            view,
            geometry,
            pivot_world=cursor.center_world,
        )
        mip_config = self._build_reslice_mip_config(view.mpr_mip, target_viewport)
        if output_shape is not None and mip_config.enabled:
            mip_config = replace(mip_config, max_samples=3)
        plane = compat.reslice_plane(
            volume,
            sampling_geometry,
            plane_pose,
            mip_config,
            interpolation_order=interpolation_order,
        )
        current, total = self._get_mpr_viewport_index_info(view, volume.shape, target_viewport, cursor=cursor, geometry=geometry)
        if target_viewport == MPR_VIEWPORT_AXIAL:
            view.current_index = current
        plane_pixels = plane.astype(np.float32, copy=False)
        self._store_mpr_plane_cache(cache_key, plane_pixels, current, total)
        return plane_pixels, current, total

    def _get_mpr_plane_cache_key(
        self,
        view: ViewRecord,
        viewport_key: str,
        output_shape: tuple[int, int],
        interpolation_order: int,
    ) -> tuple[object, ...]:
        group = view.view_group
        mip_state = view.mpr_mip.viewports.get(viewport_key, MprMipViewportState())
        model_rotation = (
            tuple(tuple(float(value) for value in row) for row in group.mpr_model_rotation_world)
            if group is not None
            else None
        )
        independent_normals = (
            tuple(
                (key, tuple(float(value) for value in group.mpr_independent_plane_normals[key]))
                for key in sorted(group.mpr_independent_plane_normals)
            )
            if group is not None
            else None
        )
        return (
            view.workspace_id,
            view.series_id,
            group.group_id if group is not None else view.view_id,
            self._get_mpr_revision(group),
            self._should_use_mpr_display_basis_for_cursor_offsets(group),
            None if group is not None else int(view.mpr_axial_index),
            None if group is not None else int(view.mpr_coronal_index),
            None if group is not None else int(view.mpr_sagittal_index),
            viewport_key,
            int(output_shape[0]),
            int(output_shape[1]),
            int(interpolation_order),
            bool(view.mpr_mip.enabled),
            str(view.mpr_mip.algorithm or "maximum"),
            max(0, min(100, int(mip_state.thickness))),
            model_rotation,
            independent_normals,
        )

    def _store_mpr_plane_cache(
        self,
        cache_key: tuple[object, ...],
        plane_pixels: np.ndarray,
        current: int,
        total: int,
    ) -> None:
        self._mpr_plane_cache[cache_key] = (plane_pixels, int(current), int(total))
        self._mpr_plane_cache.move_to_end(cache_key)
        while len(self._mpr_plane_cache) > MPR_PLANE_CACHE_MAX_ITEMS:
            self._mpr_plane_cache.popitem(last=False)

    def _extract_oblique_mpr_plane(
        self,
        view: ViewRecord,
        volume: np.ndarray,
        viewport_key: str,
        plane_state: MprObliquePlaneState,
    ) -> tuple[np.ndarray, int, int]:
        del plane_state
        return self._extract_mpr_plane(view, volume, viewport_key)

    def _build_mpr_model_sampling_geometry(
        self,
        view: ViewRecord,
        geometry: VolumeGeometry,
        *,
        pivot_world: np.ndarray,
    ) -> VolumeGeometry:
        group = view.view_group
        if group is None:
            return geometry

        rotation_world = self._get_mpr_model_rotation_matrix(group)
        if np.allclose(rotation_world, np.eye(3, dtype=np.float64), atol=1e-8):
            return geometry

        if group.mpr_model_rotation_pivot_world is None:
            self._set_mpr_model_rotation_pivot_world(group, pivot_world)
        pivot = self._get_mpr_model_rotation_pivot_world(group, pivot_world)
        inverse_rotation = rotation_world.T
        inverse_model_transform = np.eye(4, dtype=np.float64)
        inverse_model_transform[:3, :3] = inverse_rotation
        inverse_model_transform[:3, 3] = pivot - inverse_rotation @ pivot
        world_to_ijk = np.asarray(geometry.world_to_ijk, dtype=np.float64) @ inverse_model_transform
        return VolumeGeometry(
            shape_ijk=geometry.shape_ijk,
            ijk_to_world=np.linalg.inv(world_to_ijk),
            world_to_ijk=world_to_ijk,
            spacing_hint_mm=geometry.spacing_hint_mm,
        )

    @staticmethod
    def _get_mpr_model_rotation_matrix(group: ViewGroupRecord) -> np.ndarray:
        matrix = np.asarray(group.mpr_model_rotation_world, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            return np.eye(3, dtype=np.float64)
        return orthonormalize_matrix(matrix)

    @staticmethod
    def _get_mpr_model_rotation_pivot_world(group: ViewGroupRecord, fallback_world: np.ndarray) -> np.ndarray:
        if group.mpr_model_rotation_pivot_world is not None:
            pivot = np.asarray(group.mpr_model_rotation_pivot_world, dtype=np.float64)
            if pivot.shape == (3,) and np.all(np.isfinite(pivot)):
                return pivot
        return np.asarray(fallback_world, dtype=np.float64)

    @staticmethod
    def _set_mpr_model_rotation_pivot_world(group: ViewGroupRecord, pivot_world: np.ndarray) -> None:
        pivot = np.asarray(pivot_world, dtype=np.float64)
        if pivot.shape != (3,) or not np.all(np.isfinite(pivot)):
            return
        group.mpr_model_rotation_pivot_world = tuple(float(value) for value in pivot)

    @staticmethod
    def _set_mpr_model_rotation_matrix(
        group: ViewGroupRecord,
        matrix: np.ndarray,
        *,
        pivot_world: np.ndarray | None = None,
    ) -> None:
        normalized = orthonormalize_matrix(np.asarray(matrix, dtype=np.float64))
        group.mpr_model_rotation_world = tuple(
            tuple(float(value) for value in normalized[row_index])
            for row_index in range(3)
        )
        if np.allclose(normalized, np.eye(3, dtype=np.float64), atol=1e-8):
            group.mpr_model_rotation_pivot_world = None
        elif pivot_world is not None and group.mpr_model_rotation_pivot_world is None:
            compat.ViewerService._set_mpr_model_rotation_pivot_world(group, pivot_world)

    @staticmethod
    def _get_mpr_model_source_direction(group: ViewGroupRecord | None, direction_world: np.ndarray) -> np.ndarray:
        direction = np.asarray(direction_world, dtype=np.float64)
        if group is None:
            return direction
        return compat.ViewerService._get_mpr_model_rotation_matrix(group).T @ direction

    @staticmethod
    def _should_apply_mpr_model_rotation_to_plane_labels(
        group: ViewGroupRecord | None,
        plane_pose: PlanePose | None,
    ) -> bool:
        if group is None or plane_pose is None:
            return False
        rotation = compat.ViewerService._get_mpr_model_rotation_matrix(group)
        if np.allclose(rotation, np.eye(3, dtype=np.float64), atol=1e-8):
            return False
        normal = mpr_geometry.normalize_oblique_vector(
            np.asarray(plane_pose.normal_world, dtype=np.float64),
            fallback=(1.0, 0.0, 0.0),
        )
        return not np.allclose(rotation @ normal, normal, atol=1e-6)

    @staticmethod
    def _normalize_oblique_vector(
        value: tuple[float, float, float] | np.ndarray,
        *,
        fallback: tuple[float, float, float],
    ) -> np.ndarray:
        return mpr_geometry.normalize_oblique_vector(value, fallback=fallback)

    def _build_default_mpr_frame_state(self, volume_shape: tuple[int, int, int]) -> MprFrameState:
        return mpr_geometry.default_mpr_frame_state(volume_shape)

    def _ensure_mpr_reference_center(
        self,
        group: ViewGroupRecord,
        volume_shape: tuple[int, int, int],
    ) -> tuple[float, float, float]:
        if group.mpr_reference_center is None:
            group.mpr_reference_center = tuple(
                float(value)
                for value in self._build_default_mpr_frame_state(volume_shape).center
            )
        return group.mpr_reference_center

    @staticmethod
    def _reset_mpr_rotation_state(group: ViewGroupRecord) -> None:
        group.rotation_drag = None

    @staticmethod
    def _get_mpr_viewport_index_info(
        view: ViewRecord,
        volume_shape: tuple[int, int, int],
        viewport_key: str,
        *,
        cursor: MprCursorState | None = None,
        geometry: VolumeGeometry | None = None,
    ) -> tuple[int, int]:
        depth, height, width = volume_shape
        if view.view_group is not None and cursor is not None and geometry is not None:
            center = world_to_ijk_point(geometry, cursor.center_world)
            if viewport_key == MPR_VIEWPORT_CORONAL:
                return max(0, min(int(np.round(center[1])), height - 1)), height
            if viewport_key == MPR_VIEWPORT_SAGITTAL:
                return max(0, min(int(np.round(center[2])), width - 1)), width
            return max(0, min(int(np.round(center[0])), depth - 1)), depth
        if view.view_group is not None:
            if viewport_key == MPR_VIEWPORT_CORONAL:
                return max(0, min(view.view_group.coronal_index, height - 1)), height
            if viewport_key == MPR_VIEWPORT_SAGITTAL:
                return max(0, min(view.view_group.sagittal_index, width - 1)), width
            return max(0, min(view.view_group.axial_index, depth - 1)), depth
        if viewport_key == MPR_VIEWPORT_CORONAL:
            return max(0, min(view.mpr_coronal_index, height - 1)), height
        if viewport_key == MPR_VIEWPORT_SAGITTAL:
            return max(0, min(view.mpr_sagittal_index, width - 1)), width
        return max(0, min(view.mpr_axial_index, depth - 1)), depth

    @staticmethod
    def _clamp_3d_zoom(zoom: float) -> float:
        return min(max(float(zoom), ZOOM_MIN_3D), ZOOM_MAX_3D)
