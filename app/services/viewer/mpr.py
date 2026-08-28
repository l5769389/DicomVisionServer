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
        source_volume = self._get_series_volume(series, progress_callback=progress_callback)
        if not view.is_initialized:
            # PET MPR must resolve its preferred quantitative unit, initial
            # range, cursor and fit before the first reslice is generated.
            # Building a Source/BQML display first made the initial response
            # disagree with the immediately-following SUV state update.
            self._emit_render_progress(progress_callback, "initialize", progress_percent=72)
            self._initialize_mpr_viewport(view)
            view.is_initialized = True
        pet_display: FusionPetDisplayVolume | None = None
        if self._is_pet_series(series):
            requested_unit = view.view_group.pet_unit if view.view_group is not None else view.pet_unit
            pet_display = self._build_fusion_pet_display_volume(series, source_volume, requested_unit)
            volume = pet_display.volume
            view.pet_unit = pet_display.unit
            view.pet_unit_label = pet_display.unit_label
            if view.view_group is not None:
                view.view_group.pet_unit = pet_display.unit
                view.view_group.pet_unit_label = pet_display.unit_label
                view.pseudocolor_preset = view.view_group.pet_pseudocolor_preset
        else:
            volume = source_volume
        volume_ms = (perf_counter() - volume_started_at) * 1000.0

        target_viewport = self._resolve_mpr_viewport(view)
        is_crosshair_preview = fast_preview and metadata_mode == "mpr-crosshair-preview"
        is_window_pixel_preview = (
            fast_preview
            and fast_preview_full_resolution
            and metadata_mode == "mpr-pixel-preview"
        )
        # Crosshair batches may be throttled, but every presented image must
        # use the final pixel pipeline. Mixing the lightweight preview renderer
        # with the final renderer makes mouse-up visibly change the image.
        use_fast_pixel_path = fast_preview and not is_crosshair_preview and not is_window_pixel_preview
        self._emit_render_progress(progress_callback, "render", progress_percent=82)
        preview_plane_shape = (
            self._get_mpr_fast_preview_plane_shape(
                volume.shape,
                target_viewport,
                viewport_size=(view.height or 0, view.width or 0),
            )
            if use_fast_pixel_path and not fast_preview_full_resolution
            else None
        )
        reslice_started_at = perf_counter()
        plane_pixels, current, total = self._extract_mpr_plane(
            view,
            volume,
            target_viewport,
            output_shape=preview_plane_shape,
            interpolation_order=0 if use_fast_pixel_path and not fast_preview_full_resolution else 1,
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
            allow_downsample=False,
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
            canvas_width=view.width or 0,
            canvas_height=view.height or 0,
            view=view,
            pixel_aspect_x=pixel_aspect_x,
            pixel_aspect_y=pixel_aspect_y,
        )
        scale_bar = self._build_scale_bar_info(
            view,
            metadata_image_transform,
            self._get_mpr_spacing_xy_from_pose(target_plane_pose),
        )
        plane_min = float(np.min(plane_pixels))
        plane_max = float(np.max(plane_pixels))
        mpr_crosshair_overlay = self._build_mpr_crosshair_overlay(
            view,
            volume.shape,
            target_plane_pose.output_shape,
            metadata_image_transform,
        )
        include_static_preview_metadata = not (
            fast_preview
            and metadata_mode
            in {
                "mpr-pan-zoom-preview",
                "mpr-zoom-preview",
                "mpr-crosshair-preview",
                "mpr-model-rotate-preview",
            }
        )
        reference_instance, reference_cached = (
            (None, None)
            if use_fast_pixel_path
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
        if slice_corner_info is not None and pet_display is not None:
            slice_corner_info = self._with_pet_window_corner_info(
                slice_corner_info,
                pet_display,
                view.window_width,
                view.window_center,
            )
        include_mpr_measurement_payloads = not use_fast_pixel_path or metadata_mode in {
            "mpr-pan-zoom-preview",
            "mpr-zoom-preview",
            "mpr-model-rotate-preview",
        }
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
        if use_fast_pixel_path:
            image = self._render_fast_mpr_preview(
                context,
                order=1 if fast_preview_full_resolution else 0,
            )
        else:
            image = compat.layered_renderer.render(context)
        is_mpr_model_rotate_preview = fast_preview and metadata_mode == "mpr-model-rotate-preview"
        include_mpr_segmentation_overlay = (
            not use_fast_pixel_path
            or metadata_mode in {"mpr-segmentation-preview", "mpr-model-rotate-preview"}
        )
        model_rotation_world = np.eye(3, dtype=np.float64)
        model_rotation_pivot_world = np.asarray(segmentation_plane_pose.center_world, dtype=np.float64)
        if view.view_group is not None:
            model_rotation_world = self._get_mpr_model_rotation_matrix(view.view_group)
            model_rotation_pivot_world = self._get_mpr_model_rotation_pivot_world(
                view.view_group,
                model_rotation_pivot_world,
            )
        has_mpr_model_rotation = self._mpr_model_rotation_is_active(
            model_rotation_world,
            model_rotation_pivot_world,
        )
        mpr_segmentation_overlay = (
            self._build_mpr_segmentation_overlay_payload(
                plane_pixels,
                view.mpr_segmentation,
                target_viewport,
                segmentation_plane_pose,
                display_shape=target_plane_pose.output_shape,
                include_samples=not use_fast_pixel_path
                or metadata_mode
                in {"mpr-segmentation-preview", "mpr-model-rotate-preview"},
                sample_limit=(
                    MPR_SEGMENTATION_OVERLAY_PREVIEW_SAMPLE_LIMIT
                    if use_fast_pixel_path
                    and metadata_mode
                    in {"mpr-segmentation-preview", "mpr-model-rotate-preview"}
                    else MPR_SEGMENTATION_OVERLAY_SAMPLE_LIMIT
                ),
                model_rotation_world=model_rotation_world,
                model_rotation_pivot_world=model_rotation_pivot_world,
                guide_authoritative=is_mpr_model_rotate_preview or has_mpr_model_rotation,
            )
            if include_mpr_segmentation_overlay
            else None
        )
        has_local_segmentation_samples = bool(
            mpr_segmentation_overlay
            and any(
                region.samples is not None
                and region.samples.sampled_count >= region.samples.total_count
                for region in mpr_segmentation_overlay.regions
            )
        )
        if (
            include_mpr_segmentation_overlay
            and not is_mpr_model_rotate_preview
            and not has_local_segmentation_samples
        ):
            image = self._apply_mpr_segmentation_overlay(
                image,
                view.mpr_segmentation,
                plane_pixels,
                target_viewport,
                segmentation_plane_pose,
                render_image_transform,
                render_plan.render_view.width or 0,
                render_plan.render_view.height or 0,
                model_rotation_world=model_rotation_world,
                model_rotation_pivot_world=model_rotation_pivot_world,
            )
        image_ms = (perf_counter() - image_started_at) * 1000.0

        self._emit_render_progress(progress_callback, "encode", progress_percent=96)
        encode_started_at = perf_counter()
        image_bytes = self._encode_image(
            image,
            image_format,
            fast_preview=use_fast_pixel_path and not fast_preview_full_resolution,
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
                petInfo=(
                    None
                    if pet_display is None
                    else self._build_pet_info(
                        series,
                        pet_display,
                        window_width=view.window_width,
                        window_center=view.window_center,
                        pseudocolor_preset=view.pseudocolor_preset,
                        control_window_max=(
                            view.view_group.pet_control_window_max
                            if view.view_group is not None
                            else view.pet_control_window_max
                        ),
                    )
                ),
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
                    canvas_width=view.width or 0,
                    canvas_height=view.height or 0,
                ),
                annotations=[] if not include_mpr_measurement_payloads else self._serialize_annotations(
                    tuple(visible_annotations),
                    image_transform=metadata_image_transform,
                    canvas_width=view.width or 0,
                    canvas_height=view.height or 0,
                ),
                transform=self._build_view_transform_payload(view),
                orientation=None if not include_static_preview_metadata else self._serialize_orientation_overlay(
                    self._build_mpr_orientation_overlay(
                        view,
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
        source_pixels = context.source_pixels
        if source_pixels.ndim == 3 and source_pixels.shape[-1] in (3, 4):
            base_pixels = self._window_array(
                source_pixels,
                context.view.window_width,
                context.view.window_center,
                pixel_min=context.pixel_min,
                pixel_max=context.pixel_max,
            )
            transformed = compat.viewport_transformer.apply_affine_array(
                base_pixels,
                context.view.width or 0,
                context.view.height or 0,
                context.image_transform,
                order=order,
                cval=context.background_cval,
            )
            return Image.fromarray(transformed)

        window_width = context.view.window_width
        window_center = context.view.window_center
        if window_width is not None and window_width > 0 and window_center is not None:
            lower = float(window_center) - float(window_width) / 2.0
            upper = float(window_center) + float(window_width) / 2.0
        else:
            lower = float(context.pixel_min)
            upper = float(context.pixel_max)
        dataset = context.cached.dataset if context.cached is not None else None
        is_monochrome1 = str(getattr(dataset, "PhotometricInterpretation", "") or "").upper() == "MONOCHROME1"
        transformed_scalar = self._get_cached_fast_base_pixels(
            context,
            order=order,
            scalar_padding=upper if is_monochrome1 else lower,
        )
        base_pixels = self._window_array(
            transformed_scalar,
            window_width,
            window_center,
            pixel_min=context.pixel_min,
            pixel_max=context.pixel_max,
        )
        if is_monochrome1:
            base_pixels = np.subtract(255, base_pixels, dtype=np.uint8)
        if context.view.pseudocolor_preset != DEFAULT_PSEUDOCOLOR_PRESET:
            base_pixels = apply_pseudocolor(base_pixels, context.view.pseudocolor_preset)
        return Image.fromarray(base_pixels)

    def _get_cached_fast_base_pixels(
        self,
        context: RenderContext,
        *,
        order: int,
        scalar_padding: float,
    ) -> np.ndarray:
        cache_key = self._build_fast_base_pixels_cache_key(
            context,
            order=order,
            scalar_padding=scalar_padding,
        )
        cached = self._fast_base_pixels_cache.get(cache_key)
        if cached is not None:
            self._fast_base_pixels_cache.move_to_end(cache_key)
            return cached

        base_pixels = compat.viewport_transformer.apply_affine_array(
            np.asarray(context.source_pixels, dtype=np.float32),
            context.view.width or 0,
            context.view.height or 0,
            context.image_transform,
            order=order,
            cval=scalar_padding,
        )
        self._fast_base_pixels_cache[cache_key] = base_pixels
        self._fast_base_pixels_cache.move_to_end(cache_key)
        while len(self._fast_base_pixels_cache) > FAST_BASE_PIXELS_CACHE_MAX_ITEMS:
            self._fast_base_pixels_cache.popitem(last=False)
        return base_pixels

    @staticmethod
    def _build_fast_base_pixels_cache_key(
        context: RenderContext,
        *,
        order: int,
        scalar_padding: float,
    ) -> tuple[object, ...]:
        return (
            id(context.source_pixels),
            tuple(context.source_pixels.shape),
            str(context.source_pixels.dtype),
            int(context.view.width or 0),
            int(context.view.height or 0),
            int(order),
            float(scalar_padding),
            tuple(float(value) for value in np.asarray(context.image_transform.matrix).reshape(-1)),
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
        allow_downsample: bool = True,
    ) -> RenderPlan:
        if not allow_downsample:
            return RenderPlan(render_view=view, render_ratio=1.0)

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
        def serialize_context(context: MprIntensityContextState) -> MprIntensityContext:
            return MprIntensityContext(
                modality=context.modality,
                valueType=context.value_type,
                unit=context.unit,
                label=context.label,
                quantitative=context.quantitative,
                warnings=list(context.warnings),
            )

        def serialize_stats(stats: MprThresholdRegionStatsState | None) -> MprThresholdRegionStats | None:
            if stats is None:
                return None
            return MprThresholdRegionStats(
                status=stats.status,
                message=stats.message,
                valueMean=stats.value_mean,
                valueMin=stats.value_min,
                valueMax=stats.value_max,
                valueStdDev=stats.value_std_dev,
                huMean=stats.hu_mean,
                huMin=stats.hu_min,
                huMax=stats.hu_max,
                huStdDev=stats.hu_std_dev,
                volumeCm3=float(stats.volume_cm3),
                sampleCount=int(stats.sample_count),
                effectiveThresholdHu=stats.effective_threshold_hu,
                effectiveThresholdValue=stats.effective_threshold_value,
                intensityContext=serialize_context(stats.intensity_context),
                uptakePeak=stats.uptake_peak,
                uptakePeakReason=stats.uptake_peak_reason,
                mtvCm3=stats.mtv_cm3,
                tlg=stats.tlg,
                tlgAvailable=stats.tlg_available,
                tlgReason=stats.tlg_reason,
            )

        def serialize_region(region: MprThresholdRegionState) -> MprThresholdRegion:
            return MprThresholdRegion(
                id=str(region.id),
                enabled=bool(region.enabled),
                label=str(region.label or ""),
                thresholdValue=float(region.threshold_value),
                thresholdHu=float(region.threshold_value),
                thresholdMode=str(region.threshold_mode or "absolute"),
                thresholdPercentMax=float(region.threshold_percent_max),
                thresholdPercentile=float(region.threshold_percentile),
                componentMode=(
                    "hotspotConnected"
                    if str(region.component_mode) == "hotspotConnected"
                    else "all"
                ),
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
                valueMean=stats.value_mean,
                valueMin=stats.value_min,
                valueMax=stats.value_max,
                valueStdDev=stats.value_std_dev,
                huMean=stats.hu_mean,
                huMin=stats.hu_min,
                huMax=stats.hu_max,
                huStdDev=stats.hu_std_dev,
                volumeCm3=float(stats.volume_cm3),
                sampleCount=int(stats.sample_count),
                intensityContext=serialize_context(stats.intensity_context),
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
            lowerValue=float(state.lower_value),
            upperValue=float(state.upper_value),
            lowerHu=float(state.lower_value),
            upperHu=float(state.upper_value),
            intensityContext=serialize_context(state.intensity_context),
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
        previous_regions = {
            str(region.id): region
            for region in view.view_group.mpr_segmentation.threshold_regions
        }
        next_state = self._normalize_mpr_segmentation_state(payload.mpr_segmentation_config)
        self._normalize_changed_mpr_segmentation_regions_to_model_space(
            next_state,
            previous_regions,
            view.view_group,
        )
        if refresh_stats:
            self._refresh_mpr_segmentation_stats_for_view(view, next_state, series=series)
        view.view_group.mpr_segmentation = next_state
        return True

    @classmethod
    def _normalize_changed_mpr_segmentation_regions_to_model_space(
        cls,
        state: MprSegmentationState,
        previous_regions: dict[str, MprThresholdRegionState],
        group: ViewGroupRecord,
    ) -> None:
        rotation = cls._get_mpr_model_rotation_matrix(group)
        fallback_pivot = np.zeros(3, dtype=np.float64)
        pivot = cls._get_mpr_model_rotation_pivot_world(group, fallback_pivot)
        if not cls._mpr_model_rotation_is_active(rotation, pivot):
            return
        inverse_rotation = rotation.T

        def normalized(vector: np.ndarray, fallback: tuple[float, float, float]) -> tuple[float, float, float]:
            transformed = inverse_rotation @ vector
            length = float(np.linalg.norm(transformed))
            if not np.all(np.isfinite(transformed)) or not np.isfinite(length) or length <= 1e-12:
                return fallback
            return tuple(float(value) for value in transformed / length)

        for region in state.threshold_regions:
            previous_region = previous_regions.get(str(region.id))
            if previous_region is not None and cls._mpr_threshold_region_boxes_equal(
                region.box,
                previous_region.box,
            ):
                continue
            box = region.box
            display_center = np.asarray(box.center_world, dtype=np.float64)
            source_center = pivot + inverse_rotation @ (display_center - pivot)
            if source_center.shape == (3,) and np.all(np.isfinite(source_center)):
                box.center_world = tuple(float(value) for value in source_center)
            box.row_world = normalized(np.asarray(box.row_world, dtype=np.float64), box.row_world)
            box.col_world = normalized(np.asarray(box.col_world, dtype=np.float64), box.col_world)
            box.normal_world = normalized(np.asarray(box.normal_world, dtype=np.float64), box.normal_world)

    @staticmethod
    def _mpr_threshold_region_boxes_equal(
        first: MprThresholdRegionBoxState,
        second: MprThresholdRegionBoxState,
    ) -> bool:
        return bool(
            np.allclose(first.center_world, second.center_world, atol=1e-6)
            and np.allclose(first.row_world, second.row_world, atol=1e-6)
            and np.allclose(first.col_world, second.col_world, atol=1e-6)
            and np.allclose(first.normal_world, second.normal_world, atol=1e-6)
            and abs(float(first.width_mm) - float(second.width_mm)) <= 1e-6
            and abs(float(first.height_mm) - float(second.height_mm)) <= 1e-6
            and abs(float(first.depth_mm) - float(second.depth_mm)) <= 1e-6
            and str(first.source_viewport) == str(second.source_viewport)
        )

    @classmethod
    def _normalize_mpr_segmentation_state(cls, config: MprSegmentationConfig) -> MprSegmentationState:
        lower_source = config.lower_value if config.lower_value is not None else config.lower_hu
        upper_source = config.upper_value if config.upper_value is not None else config.upper_hu
        lower_hu = cls._finite_float_or_default(lower_source, 300.0)
        upper_hu = cls._finite_float_or_default(upper_source, 3071.0)
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
                config.lower_value is not None
                or config.upper_value is not None
                or config.lower_hu is not None
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
            lower_value=lower_hu,
            upper_value=upper_hu,
            intensity_context=cls._normalize_mpr_intensity_context(config.intensity_context),
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
            threshold_value=cls._finite_float_or_default(
                getattr(region, "threshold_value", None)
                if getattr(region, "threshold_value", None) is not None
                else getattr(region, "threshold_hu", 300.0),
                300.0,
            ),
            threshold_mode=cls._normalize_mpr_threshold_mode(getattr(region, "threshold_mode", "absolute")),
            threshold_percent_max=cls._clamp_float(
                getattr(region, "threshold_percent_max", None)
                if getattr(region, "threshold_percent_max", None) is not None
                else getattr(region, "threshold_percentile", 80.0),
                0.0,
                100.0,
                80.0,
            ),
            component_mode=(
                "hotspotConnected"
                if str(getattr(region, "component_mode", "hotspotConnected")) == "hotspotConnected"
                else "all"
            ),
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
            status=str(getattr(stats, "status", "") or ("empty" if sample_count == 0 else "ready")),
            message=str(getattr(stats, "message", "") or "").strip() or None,
            value_mean=cls._optional_finite_float(
                getattr(stats, "value_mean", None)
                if getattr(stats, "value_mean", None) is not None
                else getattr(stats, "hu_mean", None)
            ),
            value_min=cls._optional_finite_float(
                getattr(stats, "value_min", None)
                if getattr(stats, "value_min", None) is not None
                else getattr(stats, "hu_min", None)
            ),
            value_max=cls._optional_finite_float(
                getattr(stats, "value_max", None)
                if getattr(stats, "value_max", None) is not None
                else getattr(stats, "hu_max", None)
            ),
            value_std_dev=cls._optional_finite_float(
                getattr(stats, "value_std_dev", None)
                if getattr(stats, "value_std_dev", None) is not None
                else getattr(stats, "hu_std_dev", None)
            ),
            volume_cm3=cls._clamp_float(getattr(stats, "volume_cm3", 0.0), 0.0, float("inf"), 0.0),
            sample_count=sample_count,
            effective_threshold_value=cls._optional_finite_float(
                getattr(stats, "effective_threshold_value", None)
                if getattr(stats, "effective_threshold_value", None) is not None
                else getattr(stats, "effective_threshold_hu", None)
            ),
            intensity_context=cls._normalize_mpr_intensity_context(getattr(stats, "intensity_context", None)),
            uptake_peak=cls._optional_finite_float(getattr(stats, "uptake_peak", None)),
            uptake_peak_reason=getattr(stats, "uptake_peak_reason", None),
            mtv_cm3=cls._optional_finite_float(getattr(stats, "mtv_cm3", None)),
            tlg=cls._optional_finite_float(getattr(stats, "tlg", None)),
            tlg_available=bool(getattr(stats, "tlg_available", False)),
            tlg_reason=getattr(stats, "tlg_reason", None),
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
            value_mean=cls._optional_finite_float(
                getattr(stats, "value_mean", None)
                if getattr(stats, "value_mean", None) is not None
                else getattr(stats, "hu_mean", None)
            ),
            value_min=cls._optional_finite_float(
                getattr(stats, "value_min", None)
                if getattr(stats, "value_min", None) is not None
                else getattr(stats, "hu_min", None)
            ),
            value_max=cls._optional_finite_float(
                getattr(stats, "value_max", None)
                if getattr(stats, "value_max", None) is not None
                else getattr(stats, "hu_max", None)
            ),
            value_std_dev=cls._optional_finite_float(
                getattr(stats, "value_std_dev", None)
                if getattr(stats, "value_std_dev", None) is not None
                else getattr(stats, "hu_std_dev", None)
            ),
            volume_cm3=cls._clamp_float(getattr(stats, "volume_cm3", 0.0), 0.0, float("inf"), 0.0),
            sample_count=sample_count,
            intensity_context=cls._normalize_mpr_intensity_context(getattr(stats, "intensity_context", None)),
        )

    @staticmethod
    def _normalize_mpr_threshold_mode(value: object) -> str:
        normalized = str(value or "absolute").strip().lower()
        if normalized in {"percentmax", "percent-max", "percent_max", "percent-suvmax"}:
            return "percentMax"
        if normalized in {"percentile", "percent"}:
            return "percentile"
        return "absolute"

    @staticmethod
    def _finite_float_or_default(value: object, fallback: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return fallback
        return parsed if np.isfinite(parsed) else fallback

    @classmethod
    def _normalize_mpr_intensity_context(cls, value: object | None) -> MprIntensityContextState:
        if isinstance(value, MprIntensityContextState):
            return deepcopy(value)
        modality = str(getattr(value, "modality", "CT") or "CT").strip().upper()
        unit = str(getattr(value, "unit", "HU") or "HU").strip()
        label = str(getattr(value, "label", unit) or unit).strip()
        warnings = tuple(str(item) for item in (getattr(value, "warnings", None) or []) if str(item).strip())
        return MprIntensityContextState(
            modality=modality,
            value_type=str(getattr(value, "value_type", unit) or unit),
            unit=unit,
            label=label,
            quantitative=bool(getattr(value, "quantitative", modality == "CT")),
            warnings=warnings,
        )

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
                raise ValueError("No decodable DICOM instances are available for segmentation statistics")
            volume = self._get_series_volume(series_record)
            is_fdg = False
            if self._is_pet_series(series_record):
                requested_unit = (
                    view.view_group.pet_unit
                    if view.view_group is not None
                    else view.pet_unit
                )
                pet_display = self._build_fusion_pet_display_volume(series_record, volume, requested_unit)
                volume = pet_display.volume
                context = pet_display.context
                state.intensity_context = MprIntensityContextState(
                    modality="PT",
                    value_type=pet_display.unit,
                    unit=pet_display.unit,
                    label=pet_display.unit_label,
                    quantitative=bool(context.quantitative) if context is not None else False,
                    warnings=tuple(context.warnings) if context is not None else (),
                )
                is_fdg = bool(context.is_fdg) if context is not None else False
            else:
                state.intensity_context = MprIntensityContextState()
                state.lower_value = self._clamp_float(state.lower_value, -1024.0, 3071.0, 300.0)
                state.upper_value = self._clamp_float(state.upper_value, -1024.0, 3071.0, 3071.0)
                for region in state.threshold_regions:
                    region.threshold_value = self._clamp_float(
                        region.threshold_value,
                        -1024.0,
                        3071.0,
                        300.0,
                    )
            if state.intensity_context.modality == "PT":
                for region in state.threshold_regions:
                    if self._normalize_mpr_threshold_mode(region.threshold_mode) == "percentile":
                        region.threshold_mode = "percentMax"
            geometry = self._get_series_volume_geometry(series_record, volume.shape)
            self._refresh_mpr_segmentation_stats(
                state,
                volume,
                geometry,
                intensity_context=state.intensity_context,
                is_fdg=is_fdg,
            )
        except Exception as exc:
            message = str(exc).strip() or "Unable to calculate segmentation statistics"
            for region in state.threshold_regions:
                region.stats = MprThresholdRegionStatsState(
                    status="error",
                    message=message,
                    intensity_context=deepcopy(state.intensity_context),
                    uptake_peak_reason=message if state.intensity_context.modality == "PT" else None,
                    mtv_cm3=0.0 if state.intensity_context.modality == "PT" else None,
                )
            logger.warning(
                "failed to refresh MPR segmentation stats view_id=%s: %s",
                view.view_id,
                message,
                exc_info=True,
            )

    @classmethod
    def _refresh_mpr_segmentation_stats(
        cls,
        state: MprSegmentationState,
        volume: np.ndarray,
        geometry: VolumeGeometry,
        *,
        intensity_context: MprIntensityContextState | None = None,
        is_fdg: bool = False,
    ) -> None:
        context = intensity_context or state.intensity_context
        cls._refresh_mpr_segmentation_region_stats(
            state,
            volume,
            geometry,
            intensity_context=context,
            is_fdg=is_fdg,
        )
        for sphere in cls._get_mpr_voi_spheres(state):
            sphere.stats = (
                cls._empty_mpr_voi_sphere_stats(context)
                if not sphere.enabled
                else cls._compute_mpr_voi_sphere_stats(volume, geometry, sphere, intensity_context=context)
            )

    @classmethod
    def _refresh_mpr_segmentation_region_stats(
        cls,
        state: MprSegmentationState,
        volume: np.ndarray,
        geometry: VolumeGeometry,
        *,
        intensity_context: MprIntensityContextState | None = None,
        is_fdg: bool = False,
    ) -> None:
        if not state.threshold_regions:
            return
        context = intensity_context or state.intensity_context
        for region in state.threshold_regions:
            region.stats = (
                cls._empty_mpr_threshold_region_stats(intensity_context=context)
                if not region.enabled
                else cls._compute_mpr_threshold_region_stats(
                    volume,
                    geometry,
                    region,
                    intensity_context=context,
                    is_fdg=is_fdg,
                )
            )

    @classmethod
    def _empty_mpr_threshold_region_stats(
        cls,
        effective_threshold_hu: float | None = None,
        *,
        intensity_context: MprIntensityContextState | None = None,
        uptake_peak_reason: str | None = None,
    ) -> MprThresholdRegionStatsState:
        return MprThresholdRegionStatsState(
            status="empty",
            message="No voxels satisfy the current threshold",
            value_mean=None,
            value_min=None,
            value_max=None,
            value_std_dev=None,
            volume_cm3=0.0,
            sample_count=0,
            effective_threshold_value=effective_threshold_hu,
            intensity_context=deepcopy(intensity_context or MprIntensityContextState()),
            uptake_peak_reason=uptake_peak_reason,
            mtv_cm3=0.0 if (intensity_context and intensity_context.modality == "PT") else None,
        )

    @classmethod
    def _empty_mpr_voi_sphere_stats(
        cls,
        intensity_context: MprIntensityContextState | None = None,
    ) -> MprVoiSphereStatsState:
        return MprVoiSphereStatsState(
            value_mean=None,
            value_min=None,
            value_max=None,
            value_std_dev=None,
            volume_cm3=0.0,
            sample_count=0,
            intensity_context=deepcopy(intensity_context or MprIntensityContextState()),
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
        if cls._normalize_mpr_threshold_mode(region.threshold_mode) in {"percentile", "percentMax"}:
            stats_threshold = None if region.stats is None else region.stats.effective_threshold_hu
            if stats_threshold is not None and np.isfinite(stats_threshold):
                return float(stats_threshold)
        threshold = cls._finite_float_or_default(region.threshold_value, 300.0)
        if region.stats is None or region.stats.intensity_context.modality == "CT":
            return cls._clamp_float(threshold, -1024.0, 3071.0, 300.0)
        return threshold

    @classmethod
    def _compute_mpr_threshold_region_stats(
        cls,
        volume: np.ndarray,
        geometry: VolumeGeometry,
        region: MprThresholdRegionState,
        *,
        intensity_context: MprIntensityContextState | None = None,
        is_fdg: bool = False,
    ) -> MprThresholdRegionStatsState:
        context = intensity_context or MprIntensityContextState()
        threshold_mode = cls._normalize_mpr_threshold_mode(region.threshold_mode)
        threshold_hu = cls._finite_float_or_default(region.threshold_value, 300.0)
        if context.modality == "CT":
            threshold_hu = cls._clamp_float(threshold_hu, -1024.0, 3071.0, 300.0)
        empty_stats = cls._empty_mpr_threshold_region_stats(
            threshold_hu,
            intensity_context=context,
        )
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
        if context.modality == "PT":
            return cls._compute_pet_connected_threshold_region_stats(
                voxels,
                geometry,
                region,
                min_index=min_index,
                max_index=max_index,
                threshold_mode=threshold_mode,
                threshold_value=threshold_hu,
                intensity_context=context,
                is_fdg=is_fdg,
            )

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
            if threshold_mode in {"percentile", "percentMax"}:
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
        if threshold_mode in {"percentile", "percentMax"}:
            if not inside_value_blocks:
                return empty_stats
            inside_values = np.concatenate(inside_value_blocks)
            if inside_values.size <= 0:
                return empty_stats
            if threshold_mode == "percentMax":
                effective_threshold_hu = float(np.max(inside_values)) * (
                    cls._clamp_float(region.threshold_percent_max, 0.0, 100.0, 40.0) / 100.0
                )
            else:
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
            return cls._empty_mpr_threshold_region_stats(
                effective_threshold_hu,
                intensity_context=context,
            )
        hu_mean = value_sum / float(sample_count)
        variance = max(0.0, value_sum_sq / float(sample_count) - hu_mean * hu_mean)
        volume_cm3 = float(sample_count) * voxel_volume_mm3 / 1000.0
        stats = MprThresholdRegionStatsState(
            status="ready",
            value_mean=hu_mean,
            value_min=hu_min,
            value_max=hu_max,
            value_std_dev=float(np.sqrt(variance)),
            volume_cm3=volume_cm3,
            sample_count=sample_count,
            effective_threshold_value=effective_threshold_hu,
            intensity_context=deepcopy(context),
            mtv_cm3=volume_cm3 if context.modality == "PT" else None,
        )
        if context.modality == "PT" and context.unit in {
            FUSION_PET_UNIT_SUV_BW,
            FUSION_PET_UNIT_SUV_BSA,
            FUSION_PET_UNIT_SUL,
        }:
            peak_mask = np.isfinite(voxels) & (voxels > float(effective_threshold_hu))
            peak, peak_reason = cls._compute_pet_uptake_peak_from_component(
                voxels,
                peak_mask,
                geometry,
            )
            stats.uptake_peak = peak
            stats.uptake_peak_reason = peak_reason
            if is_fdg and context.unit == FUSION_PET_UNIT_SUV_BW:
                stats.tlg = float(stats.value_mean or 0.0) * volume_cm3
                stats.tlg_available = True
            else:
                stats.tlg_reason = (
                    "TLG requires confirmed FDG and SUVbw"
                    if not is_fdg or context.unit != FUSION_PET_UNIT_SUV_BW
                    else None
                )
        return stats

    @classmethod
    def _compute_pet_connected_threshold_region_stats(
        cls,
        volume: np.ndarray,
        geometry: VolumeGeometry,
        region: MprThresholdRegionState,
        *,
        min_index: np.ndarray,
        max_index: np.ndarray,
        threshold_mode: str,
        threshold_value: float,
        intensity_context: MprIntensityContextState,
        is_fdg: bool,
    ) -> MprThresholdRegionStatsState:
        slices = tuple(
            slice(int(min_index[axis]), int(max_index[axis]) + 1)
            for axis in range(3)
        )
        cropped = np.asarray(volume[slices], dtype=np.float64)
        empty_stats = cls._empty_mpr_threshold_region_stats(
            threshold_value,
            intensity_context=intensity_context,
        )
        if cropped.size == 0:
            return empty_stats

        indices = np.indices(cropped.shape, dtype=np.float64)
        ii = indices[0] + float(min_index[0])
        jj = indices[1] + float(min_index[1])
        kk = indices[2] + float(min_index[2])
        affine = np.asarray(geometry.ijk_to_world, dtype=np.float64)
        world_x = affine[0, 0] * ii + affine[0, 1] * jj + affine[0, 2] * kk + affine[0, 3]
        world_y = affine[1, 0] * ii + affine[1, 1] * jj + affine[1, 2] * kk + affine[1, 3]
        world_z = affine[2, 0] * ii + affine[2, 1] * jj + affine[2, 2] * kk + affine[2, 3]

        box = region.box
        center = np.asarray(box.center_world, dtype=np.float64)
        row = np.asarray(box.row_world, dtype=np.float64)
        col = np.asarray(box.col_world, dtype=np.float64)
        normal = np.asarray(box.normal_world, dtype=np.float64)
        delta_x = world_x - center[0]
        delta_y = world_y - center[1]
        delta_z = world_z - center[2]
        inside_box = (
            (np.abs(delta_x * row[0] + delta_y * row[1] + delta_z * row[2]) <= float(box.height_mm) / 2.0 + 1e-6)
            & (np.abs(delta_x * col[0] + delta_y * col[1] + delta_z * col[2]) <= float(box.width_mm) / 2.0 + 1e-6)
            & (
                np.abs(delta_x * normal[0] + delta_y * normal[1] + delta_z * normal[2])
                <= float(box.depth_mm) / 2.0 + 1e-6
            )
            & np.isfinite(cropped)
        )
        inside_values = cropped[inside_box]
        if inside_values.size == 0:
            return empty_stats

        effective_threshold = float(threshold_value)
        if threshold_mode == "percentMax":
            effective_threshold = float(np.max(inside_values)) * (
                cls._clamp_float(region.threshold_percent_max, 0.0, 100.0, 40.0) / 100.0
            )
        elif threshold_mode == "percentile":
            effective_threshold = float(
                np.percentile(
                    inside_values,
                    cls._clamp_float(region.threshold_percentile, 0.0, 100.0, 80.0),
                )
            )

        candidate_mask = inside_box & (cropped > effective_threshold)
        if not bool(np.any(candidate_mask)):
            return cls._empty_mpr_threshold_region_stats(
                effective_threshold,
                intensity_context=intensity_context,
            )

        component_mask = candidate_mask
        if str(region.component_mode or "hotspotConnected") == "hotspotConnected":
            labels, component_count = ndimage.label(
                candidate_mask,
                structure=np.ones((3, 3, 3), dtype=np.uint8),
            )
            if component_count <= 0:
                return cls._empty_mpr_threshold_region_stats(
                    effective_threshold,
                    intensity_context=intensity_context,
                )
            seed_flat_index = int(np.argmax(np.where(candidate_mask, cropped, -np.inf)))
            seed_index = np.unravel_index(seed_flat_index, cropped.shape)
            component_label = int(labels[seed_index])
            if component_label <= 0:
                return cls._empty_mpr_threshold_region_stats(
                    effective_threshold,
                    intensity_context=intensity_context,
                )
            component_mask = labels == component_label

        values = np.asarray(cropped[component_mask], dtype=np.float64)
        sample_count = int(values.size)
        if sample_count <= 0:
            return cls._empty_mpr_threshold_region_stats(
                effective_threshold,
                intensity_context=intensity_context,
            )

        voxel_volume_mm3 = cls._get_geometry_voxel_volume_mm3(geometry)
        volume_cm3 = float(sample_count) * voxel_volume_mm3 / 1000.0
        region.authoritative_mask = np.asarray(component_mask, dtype=bool).copy()
        region.authoritative_mask_origin = tuple(int(min_index[axis]) for axis in range(3))
        region.authoritative_geometry = geometry
        peak: float | None = None
        peak_reason: str | None = None
        if intensity_context.unit in {
            FUSION_PET_UNIT_SUV_BW,
            FUSION_PET_UNIT_SUV_BSA,
            FUSION_PET_UNIT_SUL,
        }:
            peak, peak_reason = cls._compute_pet_uptake_peak_from_component(
                np.asarray(volume, dtype=np.float32),
                component_mask,
                geometry,
                origin_index=min_index,
            )
        mean_value = float(np.mean(values, dtype=np.float64))
        stats = MprThresholdRegionStatsState(
            status="ready",
            value_mean=mean_value,
            value_min=float(np.min(values)),
            value_max=float(np.max(values)),
            value_std_dev=float(np.std(values, dtype=np.float64)),
            volume_cm3=volume_cm3,
            sample_count=sample_count,
            effective_threshold_value=effective_threshold,
            intensity_context=deepcopy(intensity_context),
            mtv_cm3=volume_cm3,
            uptake_peak=peak,
            uptake_peak_reason=peak_reason,
        )
        if is_fdg and intensity_context.unit == FUSION_PET_UNIT_SUV_BW:
            stats.tlg = mean_value * volume_cm3
            stats.tlg_available = True
        else:
            stats.tlg_reason = "TLG requires confirmed FDG and SUVbw"
        return stats

    @classmethod
    def _compute_pet_uptake_peak_from_component(
        cls,
        volume: np.ndarray,
        component_mask: np.ndarray,
        geometry: VolumeGeometry,
        *,
        origin_index: np.ndarray | tuple[int, int, int] | list[int] | None = None,
    ) -> tuple[float | None, str | None]:
        voxels = np.asarray(volume, dtype=np.float32)
        lesion_mask = np.asarray(component_mask, dtype=bool)
        voxel_volume_mm3 = cls._get_geometry_voxel_volume_mm3(geometry)
        if voxels.ndim != 3 or lesion_mask.ndim != 3 or voxel_volume_mm3 <= 0.0:
            return None, "SUVpeak requires a valid three-dimensional PET component"
        origin = np.zeros(3, dtype=np.int64)
        if lesion_mask.shape != voxels.shape:
            if origin_index is None:
                return None, "SUVpeak requires a valid three-dimensional PET component"
            origin = np.asarray(origin_index, dtype=np.int64)
            if origin.shape != (3,) or not np.all(np.isfinite(origin)):
                return None, "SUVpeak requires a valid three-dimensional PET component"

        affine = np.asarray(geometry.ijk_to_world, dtype=np.float64)
        spacing = np.asarray(
            [
                np.linalg.norm(affine[:3, 0]),
                np.linalg.norm(affine[:3, 1]),
                np.linalg.norm(affine[:3, 2]),
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(spacing)) or np.any(spacing <= 0.0):
            return None, "SUVpeak requires valid physical voxel spacing"

        sphere_radius_mm = float((3.0 * 1000.0 / (4.0 * np.pi)) ** (1.0 / 3.0))
        radii_voxels = np.ceil(sphere_radius_mm / spacing).astype(np.int64)
        kernel_axes = [
            np.arange(-int(radius), int(radius) + 1, dtype=np.float64) * spacing[axis]
            for axis, radius in enumerate(radii_voxels)
        ]
        grid = np.meshgrid(*kernel_axes, indexing="ij")
        sphere_kernel = (
            grid[0] * grid[0] + grid[1] * grid[1] + grid[2] * grid[2]
            <= sphere_radius_mm * sphere_radius_mm + 1e-6
        )
        kernel_count = int(np.count_nonzero(sphere_kernel))
        if kernel_count <= 0:
            return None, "SUVpeak sphere cannot be sampled at this voxel spacing"

        lesion_indices = np.argwhere(lesion_mask)
        if lesion_indices.size <= 0:
            return None, "SUVpeak requires a valid three-dimensional PET component"
        lesion_min = np.min(lesion_indices, axis=0).astype(np.int64)
        lesion_max = np.max(lesion_indices, axis=0).astype(np.int64)
        global_min = origin + lesion_min
        global_max = origin + lesion_max
        volume_shape = np.asarray(voxels.shape, dtype=np.int64)
        expanded_min = np.maximum(0, global_min - radii_voxels)
        expanded_max = np.minimum(volume_shape - 1, global_max + radii_voxels)
        if bool(np.any(expanded_min > expanded_max)):
            return None, "No 1.0 ml SUVpeak sphere fits inside the acquired PET volume"

        expanded_slices = tuple(
            slice(int(expanded_min[axis]), int(expanded_max[axis]) + 1)
            for axis in range(3)
        )
        expanded_voxels = np.asarray(voxels[expanded_slices], dtype=np.float32)
        if expanded_voxels.size <= 0:
            return None, "No 1.0 ml SUVpeak sphere fits inside the acquired PET volume"

        kernel = sphere_kernel.astype(np.float32)
        finite_mask = np.isfinite(expanded_voxels)
        sample_values = np.where(finite_mask, expanded_voxels, 0.0)
        sample_sums = ndimage.convolve(sample_values, kernel, mode="constant", cval=0.0)
        sample_counts = ndimage.convolve(finite_mask.astype(np.float32), kernel, mode="constant", cval=0.0)
        candidate_centers = np.zeros(expanded_voxels.shape, dtype=bool)
        candidate_indices = origin + lesion_indices - expanded_min
        in_expanded = np.all((candidate_indices >= 0) & (candidate_indices < np.asarray(expanded_voxels.shape)), axis=1)
        if not bool(np.any(in_expanded)):
            return None, "No 1.0 ml SUVpeak sphere fits inside the acquired PET volume"
        candidate_indices = candidate_indices[in_expanded]
        candidate_centers[
            candidate_indices[:, 0],
            candidate_indices[:, 1],
            candidate_indices[:, 2],
        ] = True
        valid_centers = candidate_centers & (sample_counts >= float(kernel_count) - 0.5)
        if not bool(np.any(valid_centers)):
            return None, "No 1.0 ml SUVpeak sphere fits inside the acquired PET volume"
        means = sample_sums[valid_centers] / float(kernel_count)
        return float(np.max(means)), None

    @classmethod
    def _compute_pet_uptake_peak(
        cls,
        volume: np.ndarray,
        geometry: VolumeGeometry,
        region: MprThresholdRegionState,
        effective_threshold: float,
    ) -> tuple[float | None, str | None]:
        voxels = np.asarray(volume, dtype=np.float32)
        voxel_volume_mm3 = cls._get_geometry_voxel_volume_mm3(geometry)
        if voxels.ndim != 3 or voxel_volume_mm3 <= 0.0:
            return None, "SUVpeak requires a valid three-dimensional PET volume"

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
            corners_ijk = np.asarray(
                [world_to_ijk_point(geometry, point) for point in corners_world],
                dtype=np.float64,
            )
        except (TypeError, ValueError):
            return None, "SUVpeak region is outside the valid PET geometry"
        shape = np.asarray(voxels.shape, dtype=np.int64)
        min_index = np.maximum(0, np.floor(np.min(corners_ijk, axis=0) - 1.0).astype(np.int64))
        max_index = np.minimum(shape - 1, np.ceil(np.max(corners_ijk, axis=0) + 1.0).astype(np.int64))
        if bool(np.any(min_index > max_index)):
            return None, "SUVpeak region does not intersect the PET volume"

        slices = tuple(
            slice(int(min_index[axis]), int(max_index[axis]) + 1)
            for axis in range(3)
        )
        cropped = np.asarray(voxels[slices], dtype=np.float32)
        if cropped.size == 0:
            return None, "SUVpeak region does not contain PET voxels"

        indices = np.indices(cropped.shape, dtype=np.float64)
        ii = indices[0] + float(min_index[0])
        jj = indices[1] + float(min_index[1])
        kk = indices[2] + float(min_index[2])
        affine = np.asarray(geometry.ijk_to_world, dtype=np.float64)
        world_x = affine[0, 0] * ii + affine[0, 1] * jj + affine[0, 2] * kk + affine[0, 3]
        world_y = affine[1, 0] * ii + affine[1, 1] * jj + affine[1, 2] * kk + affine[1, 3]
        world_z = affine[2, 0] * ii + affine[2, 1] * jj + affine[2, 2] * kk + affine[2, 3]
        delta_x = world_x - center[0]
        delta_y = world_y - center[1]
        delta_z = world_z - center[2]
        inside_box = (
            (np.abs(delta_x * row[0] + delta_y * row[1] + delta_z * row[2]) <= float(box.height_mm) / 2.0 + 1e-6)
            & (np.abs(delta_x * col[0] + delta_y * col[1] + delta_z * col[2]) <= float(box.width_mm) / 2.0 + 1e-6)
            & (
                np.abs(delta_x * normal[0] + delta_y * normal[1] + delta_z * normal[2])
                <= float(box.depth_mm) / 2.0 + 1e-6
            )
        )
        lesion_mask = inside_box & np.isfinite(cropped) & (cropped > float(effective_threshold))
        return cls._compute_pet_uptake_peak_from_component(
            voxels,
            lesion_mask,
            geometry,
            origin_index=min_index,
        )

    @classmethod
    def _compute_mpr_voi_sphere_stats(
        cls,
        volume: np.ndarray,
        geometry: VolumeGeometry,
        sphere: MprVoiSphereState,
        *,
        intensity_context: MprIntensityContextState | None = None,
    ) -> MprVoiSphereStatsState:
        context = intensity_context or MprIntensityContextState()
        empty_stats = cls._empty_mpr_voi_sphere_stats(context)
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
            value_mean=hu_mean,
            value_min=hu_min,
            value_max=hu_max,
            value_std_dev=float(np.sqrt(variance)),
            volume_cm3=float(sample_count) * voxel_volume_mm3 / 1000.0,
            sample_count=sample_count,
            intensity_context=deepcopy(context),
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
        *,
        model_rotation_world: np.ndarray | None = None,
        model_rotation_pivot_world: np.ndarray | None = None,
    ) -> Image.Image:
        if canvas_width <= 0 or canvas_height <= 0:
            return image
        masks = cls._build_mpr_segmentation_region_plane_masks(
            source_pixels,
            state,
            viewport_key,
            plane_pose,
            model_rotation_world=model_rotation_world,
            model_rotation_pivot_world=model_rotation_pivot_world,
        )
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
        *,
        model_rotation_world: np.ndarray | None = None,
        model_rotation_pivot_world: np.ndarray | None = None,
    ) -> np.ndarray | None:
        masks = cls._build_mpr_segmentation_region_plane_masks(
            source_pixels,
            state,
            viewport_key,
            plane_pose,
            model_rotation_world=model_rotation_world,
            model_rotation_pivot_world=model_rotation_pivot_world,
        )
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
        display_shape: tuple[int, int] | None = None,
        include_samples: bool = True,
        sample_limit: int = MPR_SEGMENTATION_OVERLAY_SAMPLE_LIMIT,
        model_rotation_world: np.ndarray | None = None,
        model_rotation_pivot_world: np.ndarray | None = None,
        guide_authoritative: bool = False,
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
        resolved_display_shape = (
            (max(1, int(display_shape[0])), max(1, int(display_shape[1])))
            if display_shape is not None
            else pixels.shape[:2]
        )
        masks = cls._build_mpr_segmentation_region_plane_masks(
            source_pixels,
            state,
            viewport_key,
            plane_pose,
            model_rotation_world=model_rotation_world,
            model_rotation_pivot_world=model_rotation_pivot_world,
        )
        masks_by_region_id = {mask.region_id: mask.mask for mask in masks}
        regions: list[MprSegmentationOverlayRegion] = []
        for region in state.threshold_regions:
            mask = masks_by_region_id.get(str(region.id))
            rect = cls._build_mpr_segmentation_mask_rect(mask) if mask is not None else None
            geometry_mask: np.ndarray | None = None
            guide_points: list[MprSegmentationOverlayPoint] = []
            guide_world_points: list[MprSegmentationOverlayWorldPoint] = []
            contour_world_points: list[list[MprSegmentationOverlayWorldPoint]] = []
            display_box: MprThresholdRegionBox | None = None
            guide_intersects_plane = True
            requested_authoritative_guide = bool(guide_authoritative)
            samples: MprSegmentationOverlaySamples | None = None
            sample_revision = 0
            if region.enabled and plane_pose is not None and plane_grid is not None:
                geometry_mask = cls._build_mpr_threshold_region_plane_mask(
                    region,
                    plane_pose,
                    pixels.shape[:2],
                    plane_grid,
                    model_rotation_world=model_rotation_world,
                    model_rotation_pivot_world=model_rotation_pivot_world,
                )
                guide_points = cls._build_mpr_segmentation_mask_guide_points(geometry_mask)
                display_box = cls._build_mpr_threshold_region_display_box(
                    region,
                    model_rotation_world=model_rotation_world,
                    model_rotation_pivot_world=model_rotation_pivot_world,
                )
                guide_world_points = cls._build_mpr_threshold_region_plane_world_points(
                    region,
                    plane_pose,
                    model_rotation_world=model_rotation_world,
                    model_rotation_pivot_world=model_rotation_pivot_world,
                )
                guide_intersects_plane = bool(guide_world_points)
                if mask is not None:
                    contour_world_points = cls._build_mpr_segmentation_mask_contour_world_points(
                        mask,
                        plane_pose,
                    )
                sample_revision = cls._build_mpr_segmentation_sample_revision(
                    region,
                    plane_pose,
                    resolved_display_shape,
                    model_rotation_world=model_rotation_world,
                    model_rotation_pivot_world=model_rotation_pivot_world,
                )
                if include_samples:
                    samples = cls._build_mpr_segmentation_overlay_samples(
                        pixels,
                        mask if mask is not None else geometry_mask,
                        display_shape=resolved_display_shape,
                        sample_limit=sample_limit,
                    )
            regions.append(
                MprSegmentationOverlayRegion(
                    regionId=str(region.id),
                    visible=bool(guide_world_points) or bool(contour_world_points),
                    rect=rect,
                    displayBox=display_box,
                    guidePoints=guide_points,
                    guideWorldPoints=guide_world_points,
                    contourWorldPoints=contour_world_points,
                    guideAuthoritative=requested_authoritative_guide,
                    guideIntersectsPlane=guide_intersects_plane,
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
        *,
        model_rotation_world: np.ndarray | None = None,
        model_rotation_pivot_world: np.ndarray | None = None,
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
            "modelRotation": (
                [
                    [round(float(value), 8) for value in row]
                    for row in np.asarray(model_rotation_world, dtype=np.float64)
                ]
                if model_rotation_world is not None
                else None
            ),
            "modelRotationPivot": (
                vector_payload(np.asarray(model_rotation_pivot_world, dtype=np.float64))
                if model_rotation_pivot_world is not None
                else None
            ),
        }
        encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        return int.from_bytes(hashlib.blake2b(encoded, digest_size=4).digest(), "big")

    @staticmethod
    def _build_mpr_segmentation_overlay_samples(
        pixels: np.ndarray,
        geometry_mask: np.ndarray,
        *,
        display_shape: tuple[int, int] | None = None,
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
        source_height, source_width = mask_array.shape
        display_height, display_width = display_shape or mask_array.shape
        scale_x = np.float32(float(display_width) / float(max(1, source_width)))
        scale_y = np.float32(float(display_height) / float(max(1, source_height)))
        points[0::3] = (cols.astype(np.float32, copy=False) + np.float32(0.5)) * scale_x
        points[1::3] = (rows.astype(np.float32, copy=False) + np.float32(0.5)) * scale_y
        points[2::3] = values
        return MprSegmentationOverlaySamples(
            points=points.tolist(),
            totalCount=total_count,
            sampledCount=int(values.size),
        )

    @classmethod
    def _build_mpr_segmentation_mask_guide_points(
        cls,
        geometry_mask: np.ndarray | None,
    ) -> list[MprSegmentationOverlayPoint]:
        mask = cls._coerce_mpr_segmentation_mask(geometry_mask)
        if mask is None:
            return []
        height, width = mask.shape
        hull = cls._build_mpr_segmentation_mask_guide_cell_points(mask)
        if len(hull) < 3:
            return []

        return [
            MprSegmentationOverlayPoint(
                x=float(np.clip(point[0] / float(width), 0.0, 1.0)),
                y=float(np.clip(point[1] / float(height), 0.0, 1.0)),
            )
            for point in hull
        ]

    @classmethod
    def _build_mpr_segmentation_mask_guide_world_points(
        cls,
        geometry_mask: np.ndarray | None,
        plane_pose: PlanePose,
    ) -> list[MprSegmentationOverlayWorldPoint]:
        mask = cls._coerce_mpr_segmentation_mask(geometry_mask)
        if mask is None:
            return []
        hull = cls._build_mpr_segmentation_mask_guide_cell_points(mask)
        if len(hull) < 3:
            return []
        return [
            cls._mpr_plane_mask_boundary_point_to_world_point(plane_pose, point)
            for point in hull
        ]

    @classmethod
    def _build_mpr_segmentation_mask_contour_world_points(
        cls,
        geometry_mask: np.ndarray | None,
        plane_pose: PlanePose,
        *,
        point_limit: int = MPR_SEGMENTATION_OVERLAY_CONTOUR_POINT_LIMIT,
    ) -> list[list[MprSegmentationOverlayWorldPoint]]:
        mask = cls._coerce_mpr_segmentation_mask(geometry_mask)
        if mask is None:
            return []
        contours = cls._build_mpr_segmentation_mask_boundary_contours(mask)
        if not contours:
            return []
        resolved_point_limit = max(3, int(point_limit))
        world_contours: list[list[MprSegmentationOverlayWorldPoint]] = []
        for contour in contours:
            limited_contour = cls._limit_mpr_segmentation_contour_points(contour, resolved_point_limit)
            world_points = [
                cls._mpr_plane_mask_boundary_point_to_world_point(plane_pose, point)
                for point in limited_contour
            ]
            if len(world_points) >= 3:
                world_contours.append(world_points)
        return world_contours

    @staticmethod
    def _coerce_mpr_segmentation_mask(geometry_mask: np.ndarray | None) -> np.ndarray | None:
        if geometry_mask is None:
            return None
        mask = np.asarray(geometry_mask, dtype=bool)
        if mask.ndim != 2 or not bool(np.any(mask)):
            return None
        return mask

    @staticmethod
    def _build_mpr_segmentation_mask_guide_cell_points(
        mask: np.ndarray,
    ) -> list[tuple[float, float]]:
        height, width = mask.shape
        padded = np.pad(mask, 1, mode="constant", constant_values=False)
        interior = (
            mask
            & padded[:-2, 1:-1]
            & padded[2:, 1:-1]
            & padded[1:-1, :-2]
            & padded[1:-1, 2:]
        )
        rows, cols = np.nonzero(mask & ~interior)
        if rows.size == 0:
            return []

        corners: set[tuple[float, float]] = set()
        for row, col in zip(rows.tolist(), cols.tolist(), strict=True):
            corners.update(
                {
                    (float(col), float(row)),
                    (float(col + 1), float(row)),
                    (float(col + 1), float(row + 1)),
                    (float(col), float(row + 1)),
                }
            )

        def cross(
            origin: tuple[float, float],
            first: tuple[float, float],
            second: tuple[float, float],
        ) -> float:
            return (
                (first[0] - origin[0]) * (second[1] - origin[1])
                - (first[1] - origin[1]) * (second[0] - origin[0])
            )

        sorted_points = sorted(corners)
        lower: list[tuple[float, float]] = []
        for point in sorted_points:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
                lower.pop()
            lower.append(point)
        upper: list[tuple[float, float]] = []
        for point in reversed(sorted_points):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
                upper.pop()
            upper.append(point)
        hull = lower[:-1] + upper[:-1]
        return hull if len(hull) >= 3 else []

    @staticmethod
    def _build_mpr_segmentation_mask_boundary_contours(
        mask: np.ndarray,
    ) -> list[list[tuple[float, float]]]:
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.ndim != 2 or not bool(np.any(mask_array)):
            return []
        height, width = mask_array.shape
        edges: list[tuple[tuple[float, float], tuple[float, float]]] = []

        rows, cols = np.nonzero(mask_array)
        for row, col in zip(rows.tolist(), cols.tolist(), strict=True):
            if row <= 0 or not bool(mask_array[row - 1, col]):
                edges.append(((float(col), float(row)), (float(col + 1), float(row))))
            if col >= width - 1 or not bool(mask_array[row, col + 1]):
                edges.append(((float(col + 1), float(row)), (float(col + 1), float(row + 1))))
            if row >= height - 1 or not bool(mask_array[row + 1, col]):
                edges.append(((float(col + 1), float(row + 1)), (float(col), float(row + 1))))
            if col <= 0 or not bool(mask_array[row, col - 1]):
                edges.append(((float(col), float(row + 1)), (float(col), float(row))))

        edges_by_start: dict[tuple[float, float], list[int]] = {}
        for index, (start, _end) in enumerate(edges):
            edges_by_start.setdefault(start, []).append(index)

        used: set[int] = set()
        contours: list[list[tuple[float, float]]] = []
        for start_edge_index, (start, _end) in enumerate(edges):
            if start_edge_index in used:
                continue
            contour: list[tuple[float, float]] = [start]
            edge_index = start_edge_index
            for _guard in range(len(edges) + 1):
                if edge_index in used:
                    break
                used.add(edge_index)
                _edge_start, edge_end = edges[edge_index]
                contour.append(edge_end)
                if edge_end == start:
                    break
                next_edge_index = next(
                    (
                        candidate
                        for candidate in edges_by_start.get(edge_end, [])
                        if candidate not in used
                    ),
                    None,
                )
                if next_edge_index is None:
                    break
                edge_index = next_edge_index
            if len(contour) < 4:
                continue
            if contour[0] == contour[-1]:
                contour.pop()
            if len(contour) >= 3:
                contours.append(contour)
        return contours

    @staticmethod
    def _limit_mpr_segmentation_contour_points(
        contour: list[tuple[float, float]],
        point_limit: int,
    ) -> list[tuple[float, float]]:
        if len(contour) <= point_limit:
            return contour
        stride = max(1, int(np.ceil(float(len(contour)) / float(point_limit))))
        return contour[::stride][:point_limit]

    @staticmethod
    def _mpr_plane_mask_boundary_point_to_world_point(
        plane_pose: PlanePose,
        point: tuple[float, float],
    ) -> MprSegmentationOverlayWorldPoint:
        height, width = plane_pose.output_shape
        col_offset_mm = (float(point[0]) - float(width) / 2.0) * float(plane_pose.pixel_spacing_col_mm)
        row_offset_mm = (float(point[1]) - float(height) / 2.0) * float(plane_pose.pixel_spacing_row_mm)
        world = (
            np.asarray(plane_pose.center_world, dtype=np.float64)
            + np.asarray(plane_pose.col_world, dtype=np.float64) * col_offset_mm
            + np.asarray(plane_pose.row_world, dtype=np.float64) * row_offset_mm
        )
        return MprSegmentationOverlayWorldPoint(
            x=float(world[0]),
            y=float(world[1]),
            z=float(world[2]),
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
        *,
        model_rotation_world: np.ndarray | None = None,
        model_rotation_pivot_world: np.ndarray | None = None,
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
                region_mask = cls._build_mpr_threshold_region_plane_mask(
                    region,
                    plane_pose,
                    pixels.shape[:2],
                    plane_grid,
                    model_rotation_world=model_rotation_world,
                    model_rotation_pivot_world=model_rotation_pivot_world,
                )
                if not bool(np.any(region_mask)):
                    continue
                authoritative_mask = cls._reslice_authoritative_mpr_region_mask(
                    region,
                    plane_pose,
                    pixels.shape[:2],
                    plane_grid,
                    model_rotation_world=model_rotation_world,
                    model_rotation_pivot_world=model_rotation_pivot_world,
                )
                if authoritative_mask is not None:
                    mask = authoritative_mask & region_mask
                    if bool(np.any(mask)):
                        masks.append(
                            MprThresholdPlaneMask(
                                region_id=str(region.id),
                                mask=mask,
                                color=region.color,
                            )
                        )
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
    def _reslice_authoritative_mpr_region_mask(
        region: MprThresholdRegionState,
        plane_pose: PlanePose,
        shape: tuple[int, int],
        plane_grid: MprThresholdPlaneGrid,
        *,
        model_rotation_world: np.ndarray | None = None,
        model_rotation_pivot_world: np.ndarray | None = None,
    ) -> np.ndarray | None:
        authoritative_mask = getattr(region, "authoritative_mask", None)
        geometry = getattr(region, "authoritative_geometry", None)
        if authoritative_mask is None or not isinstance(geometry, VolumeGeometry):
            return None
        mask_volume = np.asarray(authoritative_mask, dtype=np.uint8)
        if mask_volume.ndim != 3:
            return None
        origin = np.zeros(3, dtype=np.float64)
        origin_value = getattr(region, "authoritative_mask_origin", None)
        if origin_value is not None:
            origin = np.asarray(origin_value, dtype=np.float64)
            if origin.shape != (3,) or not np.all(np.isfinite(origin)):
                return None
        elif tuple(mask_volume.shape) != tuple(geometry.shape_ijk):
            return None

        world_x = (
            plane_grid.center_world[0]
            + plane_grid.row_world[0] * plane_grid.row_grid_mm
            + plane_grid.col_world[0] * plane_grid.col_grid_mm
        )
        world_y = (
            plane_grid.center_world[1]
            + plane_grid.row_world[1] * plane_grid.row_grid_mm
            + plane_grid.col_world[1] * plane_grid.col_grid_mm
        )
        world_z = (
            plane_grid.center_world[2]
            + plane_grid.row_world[2] * plane_grid.row_grid_mm
            + plane_grid.col_world[2] * plane_grid.col_grid_mm
        )
        if ViewerMprMixin._mpr_model_rotation_is_active(model_rotation_world, model_rotation_pivot_world):
            rotation = np.asarray(model_rotation_world, dtype=np.float64)
            pivot = np.asarray(model_rotation_pivot_world, dtype=np.float64)
            display_points = np.stack((world_x, world_y, world_z), axis=0)
            flat_display_points = display_points.reshape(3, -1)
            source_points = pivot[:, None] + rotation.T @ (flat_display_points - pivot[:, None])
            source_points = source_points.reshape(display_points.shape)
            world_x, world_y, world_z = source_points[0], source_points[1], source_points[2]
        world_to_ijk = np.asarray(geometry.world_to_ijk, dtype=np.float64)
        coordinates = np.asarray(
            [
                world_to_ijk[0, 0] * world_x
                + world_to_ijk[0, 1] * world_y
                + world_to_ijk[0, 2] * world_z
                + world_to_ijk[0, 3],
                world_to_ijk[1, 0] * world_x
                + world_to_ijk[1, 1] * world_y
                + world_to_ijk[1, 2] * world_z
                + world_to_ijk[1, 3],
                world_to_ijk[2, 0] * world_x
                + world_to_ijk[2, 1] * world_y
                + world_to_ijk[2, 2] * world_z
                + world_to_ijk[2, 3],
            ],
            dtype=np.float64,
        )
        coordinates[0] -= origin[0]
        coordinates[1] -= origin[1]
        coordinates[2] -= origin[2]
        sampled = ndimage.map_coordinates(
            mask_volume,
            coordinates,
            order=0,
            mode="constant",
            cval=0,
            prefilter=False,
        )
        return np.asarray(sampled, dtype=np.uint8).reshape(shape) > 0

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

    @staticmethod
    def _mpr_model_rotation_is_active(
        model_rotation_world: np.ndarray | None,
        model_rotation_pivot_world: np.ndarray | None,
    ) -> bool:
        rotation = np.asarray(model_rotation_world, dtype=np.float64)
        pivot = np.asarray(model_rotation_pivot_world, dtype=np.float64)
        return bool(
            rotation.shape == (3, 3)
            and pivot.shape == (3,)
            and np.all(np.isfinite(rotation))
            and np.all(np.isfinite(pivot))
            and not np.allclose(rotation, np.eye(3, dtype=np.float64), atol=1e-8)
        )

    @classmethod
    def _resolve_display_mpr_threshold_region_box(
        cls,
        region: MprThresholdRegionState,
        model_rotation_world: np.ndarray | None,
        model_rotation_pivot_world: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        def normalized(values: tuple[float, float, float] | np.ndarray) -> np.ndarray | None:
            vector = np.asarray(values, dtype=np.float64)
            if vector.shape != (3,) or not np.all(np.isfinite(vector)):
                return None
            length = float(np.linalg.norm(vector))
            if not np.isfinite(length) or length <= 1e-12:
                return None
            return vector / length

        box = region.box
        center = np.asarray(box.center_world, dtype=np.float64)
        row = normalized(box.row_world)
        col = normalized(box.col_world)
        normal = normalized(box.normal_world)
        if (
            center.shape != (3,)
            or not np.all(np.isfinite(center))
            or row is None
            or col is None
            or normal is None
        ):
            return None

        if not cls._mpr_model_rotation_is_active(model_rotation_world, model_rotation_pivot_world):
            return center, row, col, normal

        rotation = np.asarray(model_rotation_world, dtype=np.float64)
        pivot = np.asarray(model_rotation_pivot_world, dtype=np.float64)
        display_center = pivot + rotation @ (center - pivot)
        display_row = normalized(rotation @ row)
        display_col = normalized(rotation @ col)
        display_normal = normalized(rotation @ normal)
        if display_row is None or display_col is None or display_normal is None:
            return None
        return display_center, display_row, display_col, display_normal

    @classmethod
    def _build_mpr_threshold_region_plane_world_points(
        cls,
        region: MprThresholdRegionState,
        plane_pose: PlanePose,
        *,
        model_rotation_world: np.ndarray | None = None,
        model_rotation_pivot_world: np.ndarray | None = None,
    ) -> list[MprSegmentationOverlayWorldPoint]:
        geometry = cls._resolve_display_mpr_threshold_region_box(
            region,
            model_rotation_world,
            model_rotation_pivot_world,
        )
        if geometry is None:
            return []
        center, row, col, normal = geometry
        plane_center = np.asarray(plane_pose.center_world, dtype=np.float64)
        plane_normal = np.asarray(plane_pose.normal_world, dtype=np.float64)
        plane_normal_length = float(np.linalg.norm(plane_normal))
        if (
            plane_center.shape != (3,)
            or plane_normal.shape != (3,)
            or not np.all(np.isfinite(plane_center))
            or not np.all(np.isfinite(plane_normal))
            or not np.isfinite(plane_normal_length)
            or plane_normal_length <= 1e-12
        ):
            return []
        plane_normal = plane_normal / plane_normal_length

        half_axes = (
            row * (float(region.box.height_mm) / 2.0),
            col * (float(region.box.width_mm) / 2.0),
            normal * (float(region.box.depth_mm) / 2.0),
        )
        vertices = np.asarray(
            [
                center
                + row_sign * half_axes[0]
                + col_sign * half_axes[1]
                + normal_sign * half_axes[2]
                for row_sign in (-1.0, 1.0)
                for col_sign in (-1.0, 1.0)
                for normal_sign in (-1.0, 1.0)
            ],
            dtype=np.float64,
        )
        signed_distances = (vertices - plane_center) @ plane_normal
        epsilon = 1e-6
        edge_indices = (
            (0, 1), (0, 2), (0, 4),
            (1, 3), (1, 5),
            (2, 3), (2, 6),
            (3, 7),
            (4, 5), (4, 6),
            (5, 7), (6, 7),
        )
        intersections: list[np.ndarray] = []

        def append_unique(point: np.ndarray) -> None:
            if not np.all(np.isfinite(point)):
                return
            if any(float(np.linalg.norm(point - existing)) <= epsilon for existing in intersections):
                return
            intersections.append(np.asarray(point, dtype=np.float64))

        for start_index, end_index in edge_indices:
            start = vertices[start_index]
            end = vertices[end_index]
            start_distance = float(signed_distances[start_index])
            end_distance = float(signed_distances[end_index])
            start_on_plane = abs(start_distance) <= epsilon
            end_on_plane = abs(end_distance) <= epsilon
            if start_on_plane:
                append_unique(start)
            if end_on_plane:
                append_unique(end)
            if start_on_plane or end_on_plane or start_distance * end_distance >= 0.0:
                continue
            fraction = start_distance / (start_distance - end_distance)
            append_unique(start + fraction * (end - start))

        if len(intersections) < 3:
            return []
        centroid = np.mean(np.asarray(intersections), axis=0)
        plane_row = np.asarray(plane_pose.row_world, dtype=np.float64)
        plane_col = np.asarray(plane_pose.col_world, dtype=np.float64)
        intersections.sort(
            key=lambda point: float(
                np.arctan2(
                    np.dot(point - centroid, plane_row),
                    np.dot(point - centroid, plane_col),
                )
            )
        )
        return [
            MprSegmentationOverlayWorldPoint(
                x=float(point[0]),
                y=float(point[1]),
                z=float(point[2]),
            )
            for point in intersections
        ]

    @classmethod
    def _build_mpr_threshold_region_display_box(
        cls,
        region: MprThresholdRegionState,
        *,
        model_rotation_world: np.ndarray | None = None,
        model_rotation_pivot_world: np.ndarray | None = None,
    ) -> MprThresholdRegionBox | None:
        geometry = cls._resolve_display_mpr_threshold_region_box(
            region,
            model_rotation_world,
            model_rotation_pivot_world,
        )
        if geometry is None:
            return None
        center, row, col, normal = geometry
        return MprThresholdRegionBox(
            centerWorld=tuple(float(value) for value in center),
            rowWorld=tuple(float(value) for value in row),
            colWorld=tuple(float(value) for value in col),
            normalWorld=tuple(float(value) for value in normal),
            widthMm=float(region.box.width_mm),
            heightMm=float(region.box.height_mm),
            depthMm=float(region.box.depth_mm),
            sourceViewport=str(region.box.source_viewport),
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
        if state.intensity_context.modality == "CT":
            lower_hu = cls._clamp_float(state.lower_value, -1024.0, 3071.0, 300.0)
            upper_hu = cls._clamp_float(state.upper_value, -1024.0, 3071.0, 3071.0)
        else:
            lower_hu = cls._finite_float_or_default(state.lower_value, 0.0)
            upper_hu = cls._finite_float_or_default(state.upper_value, lower_hu + 1.0)
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
        *,
        model_rotation_world: np.ndarray | None = None,
        model_rotation_pivot_world: np.ndarray | None = None,
    ) -> np.ndarray:
        height, width = int(shape[0]), int(shape[1])
        if height <= 0 or width <= 0:
            return np.zeros((max(0, height), max(0, width)), dtype=bool)
        grid = plane_grid or cls._build_mpr_threshold_plane_grid(plane_pose, (height, width))
        box = region.box
        geometry = cls._resolve_display_mpr_threshold_region_box(
            region,
            model_rotation_world,
            model_rotation_pivot_world,
        )
        if geometry is None:
            return np.zeros((height, width), dtype=bool)
        box_center, box_row, box_col, box_normal = geometry
        delta_center = grid.center_world - box_center

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
