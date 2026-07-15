from __future__ import annotations

"""Stack and standalone PET rendering."""

from app.services.viewer.shared import *  # noqa: F403


class ViewerStackMixin:
    def _render_pet_view(
        self,
        view: ViewRecord,
        image_format: ImageFormat = "webp",
        *,
        fast_preview: bool = False,
        metadata_mode: str = "full",
        progress_callback: ViewRenderProgressCallback | None = None,
    ) -> RenderedImageResult:
        render_started_at = perf_counter()
        ensure_view_size(view)

        series = compat.series_registry.get(view.series_id)
        if not self._is_pet_series(series):
            raise HTTPException(status_code=400, detail="PET view requires a PT/PET series")
        self._emit_render_progress(progress_callback, "volume", progress_percent=8)
        pet_volume = self._get_series_volume(series, progress_callback=progress_callback)
        if not view.is_initialized:
            self._emit_render_progress(progress_callback, "initialize", progress_percent=72)
            self._initialize_pet_viewport(view)
            view.is_initialized = True
        if view.pseudocolor_preset != PET_STANDALONE_PSEUDOCOLOR_PRESET:
            view.pseudocolor_preset = PET_STANDALONE_PSEUDOCOLOR_PRESET

        pet_display = self._build_fusion_pet_display_volume(series, pet_volume, view.pet_unit)
        view.pet_unit = pet_display.unit
        view.pet_unit_label = pet_display.unit_label
        view.current_index = max(0, min(int(view.current_index), pet_display.volume.shape[0] - 1))
        instance, cached = self._get_indexed_instance_and_cache(series, view.current_index)
        if instance is None or cached is None:
            raise HTTPException(status_code=400, detail="PET series does not contain renderable DICOM instances")

        source_pixels = self._prepare_pet_standalone_source_pixels(
            np.asarray(pet_display.volume[view.current_index], dtype=np.float32),
            view.window_width,
            view.window_center,
        )
        pixel_min = float(np.nanmin(source_pixels)) if source_pixels.size else 0.0
        pixel_max = float(np.nanmax(source_pixels)) if source_pixels.size else 1.0
        if not np.isfinite(pixel_min):
            pixel_min = 0.0
        if not np.isfinite(pixel_max) or pixel_max <= pixel_min:
            pixel_max = pixel_min + 1.0

        metadata_started_at = perf_counter()
        render_plan = self._build_render_plan_for_shape(view, *source_pixels.shape[:2])
        image_transform = compat.viewport_transformer.build_image_to_canvas_transform(
            image_width=source_pixels.shape[1],
            image_height=source_pixels.shape[0],
            canvas_width=render_plan.render_view.width or 0,
            canvas_height=render_plan.render_view.height or 0,
            view=render_plan.render_view,
        )
        scale_bar = self._build_scale_bar_info(
            render_plan.render_view,
            image_transform,
            self._get_stack_spacing_xy(cached.dataset),
        )
        slice_corner_info = self._build_slice_corner_info_overlay(
            view,
            series,
            cached.dataset,
            current_index=view.current_index,
            total_slices=len(series.instances),
            viewport_label="PET",
        )
        slice_corner_info = self._with_pet_window_corner_info(
            slice_corner_info,
            pet_display,
            view.window_width,
            view.window_center,
        )
        include_stack_overlay_payloads = not (
            fast_preview
            and metadata_mode in {"stack-preview-lite", "stack-pixel-preview"}
        )
        visible_measurements = self._build_visible_measurements(view) if include_stack_overlay_payloads else ()
        visible_annotations = self._build_visible_annotations(view) if include_stack_overlay_payloads else ()
        context = RenderContext(
            view=render_plan.render_view,
            source_pixels=source_pixels,
            pixel_min=pixel_min,
            pixel_max=pixel_max,
            instance=instance,
            cached=cached,
            image_transform=image_transform,
            measurements=visible_measurements,
            corner_info=None,
            orientation=None,
            background_cval=FUSION_PET_STANDALONE_BACKGROUND_CVAL,
        )
        visible_presentation_measurements = (
            self._build_visible_presentation_measurements(series, instance)
            if include_stack_overlay_payloads
            else ()
        )
        visible_presentation_annotations = (
            self._build_visible_presentation_annotations(series, instance)
            if include_stack_overlay_payloads
            else ()
        )
        metadata_ms = (perf_counter() - metadata_started_at) * 1000.0

        image_started_at = perf_counter()
        if fast_preview:
            image = self._render_fast_preview(context)
        else:
            image = compat.layered_renderer.render(context)
        image_ms = (perf_counter() - image_started_at) * 1000.0

        encode_started_at = perf_counter()
        image_bytes = self._encode_image(image, image_format, fast_preview=fast_preview)
        encode_ms = (perf_counter() - encode_started_at) * 1000.0

        logger.debug(
            "PET render timing view_id=%s index=%s unit=%s fast_preview=%s image_format=%s viewport=%sx%s render=%sx%s zoom=%.4f ww=%s wl=%s metadata_ms=%.1f image_ms=%.1f encode_ms=%.1f total_ms=%.1f",
            view.view_id,
            view.current_index,
            view.pet_unit,
            fast_preview,
            image_format,
            view.width,
            view.height,
            render_plan.render_view.width,
            render_plan.render_view.height,
            view.zoom,
            view.window_width,
            view.window_center,
            metadata_ms,
            image_ms,
            encode_ms,
            (perf_counter() - render_started_at) * 1000.0,
        )

        return RenderedImageResult(
            meta=ViewImageResponse(
                slice_info=SliceInfo(current=view.current_index, total=len(series.instances)),
                window_info=WindowInfo(ww=view.window_width, wl=view.window_center),
                imageFormat=image_format,
                viewId=view.view_id,
                color=ViewColorInfo(pseudocolorPreset=view.pseudocolor_preset),
                petInfo=PetInfo(
                    seriesId=series.series_id,
                    petUnit=pet_display.unit,
                    petUnitLabel=pet_display.unit_label,
                    petWindowMin=self._resolve_window_min(view.window_width, view.window_center),
                    petWindowMax=self._resolve_window_max(view.window_width, view.window_center),
                    pseudocolorPreset=view.pseudocolor_preset,
                ),
                scaleBar=scale_bar,
                cornerInfo=self._serialize_corner_info_overlay(slice_corner_info),
                measurements=[] if not include_stack_overlay_payloads else self._serialize_measurements(
                    (*visible_measurements, *visible_presentation_measurements),
                    image_transform=image_transform,
                    canvas_width=render_plan.render_view.width or 0,
                    canvas_height=render_plan.render_view.height or 0,
                ),
                annotations=[] if not include_stack_overlay_payloads else self._serialize_annotations(
                    (*visible_annotations, *visible_presentation_annotations),
                    image_transform=image_transform,
                    canvas_width=render_plan.render_view.width or 0,
                    canvas_height=render_plan.render_view.height or 0,
                ),
                transform=self._build_view_transform_payload(view),
                orientation=self._serialize_orientation_overlay(
                    self._build_stack_orientation_overlay(render_plan.render_view, cached.dataset)
                ),
            ),
            image_bytes=image_bytes,
        )

    def _render_view(
        self,
        view: ViewRecord,
        image_format: ImageFormat = "webp",
        *,
        fast_preview: bool = False,
        metadata_mode: str = "full",
    ) -> RenderedImageResult:
        render_started_at = perf_counter()
        ensure_view_size(view)

        series = compat.series_registry.get(view.series_id)
        instance = series.instances[view.current_index]
        if not instance.sop_instance_uid:
            raise HTTPException(status_code=400, detail="DICOM instance does not contain SOPInstanceUID")

        cache_started_at = perf_counter()
        cached = compat.dicom_cache.get(instance.sop_instance_uid, instance.path)
        cache_ms = (perf_counter() - cache_started_at) * 1000.0
        metadata_started_at = perf_counter()
        render_plan = self._build_render_plan_for_shape(view, *cached.source_pixels.shape[:2])
        image_transform = compat.viewport_transformer.build_image_to_canvas_transform(
            image_width=cached.source_pixels.shape[1],
            image_height=cached.source_pixels.shape[0],
            canvas_width=render_plan.render_view.width or 0,
            canvas_height=render_plan.render_view.height or 0,
            view=render_plan.render_view,
        )
        scale_bar = self._build_scale_bar_info(
            render_plan.render_view,
            image_transform,
            self._get_stack_spacing_xy(cached.dataset),
        )
        slice_corner_info = self._build_slice_corner_info_overlay(
            view,
            series,
            cached.dataset,
            current_index=view.current_index,
            total_slices=len(series.instances),
            viewport_label="Stack",
        )
        include_stack_overlay_payloads = not (
            fast_preview
            and metadata_mode in {"stack-preview-lite", "stack-pixel-preview"}
        )
        visible_measurements = self._build_visible_measurements(view) if include_stack_overlay_payloads else ()
        visible_annotations = self._build_visible_annotations(view) if include_stack_overlay_payloads else ()
        context = RenderContext(
            view=render_plan.render_view,
            source_pixels=cached.source_pixels,
            pixel_min=cached.pixel_min,
            pixel_max=cached.pixel_max,
            instance=instance,
            cached=cached,
            image_transform=image_transform,
            measurements=visible_measurements,
            corner_info=None,
            orientation=None,
        )
        visible_presentation_measurements = (
            self._build_visible_presentation_measurements(series, instance)
            if include_stack_overlay_payloads
            else ()
        )
        visible_presentation_annotations = (
            self._build_visible_presentation_annotations(series, instance)
            if include_stack_overlay_payloads
            else ()
        )
        metadata_ms = (perf_counter() - metadata_started_at) * 1000.0

        image_started_at = perf_counter()
        if fast_preview:
            image = self._render_fast_preview(context)
        else:
            image = compat.layered_renderer.render(context)
        image_ms = (perf_counter() - image_started_at) * 1000.0

        encode_started_at = perf_counter()
        image_bytes = self._encode_image(image, image_format, fast_preview=fast_preview)
        encode_ms = (perf_counter() - encode_started_at) * 1000.0

        logger.debug(
            "stack render timing view_id=%s index=%s fast_preview=%s image_format=%s viewport=%sx%s render=%sx%s ratio=%.4f zoom=%.4f ww=%s wl=%s cache_ms=%.1f metadata_ms=%.1f image_ms=%.1f encode_ms=%.1f total_ms=%.1f",
            view.view_id,
            view.current_index,
            fast_preview,
            image_format,
            view.width,
            view.height,
            render_plan.render_view.width,
            render_plan.render_view.height,
            render_plan.render_ratio,
            view.zoom,
            view.window_width,
            view.window_center,
            cache_ms,
            metadata_ms,
            image_ms,
            encode_ms,
            (perf_counter() - render_started_at) * 1000.0,
        )

        return RenderedImageResult(
            meta=ViewImageResponse(
                slice_info=SliceInfo(current=view.current_index, total=len(series.instances)),
                window_info=WindowInfo(ww=view.window_width, wl=view.window_center),
                imageFormat=image_format,
                viewId=view.view_id,
                color=ViewColorInfo(pseudocolorPreset=view.pseudocolor_preset),
                scaleBar=scale_bar,
                cornerInfo=self._serialize_corner_info_overlay(slice_corner_info),
                measurements=[] if not include_stack_overlay_payloads else self._serialize_measurements(
                    (*visible_measurements, *visible_presentation_measurements),
                    image_transform=image_transform,
                    canvas_width=render_plan.render_view.width or 0,
                    canvas_height=render_plan.render_view.height or 0,
                ),
                annotations=[] if not include_stack_overlay_payloads else self._serialize_annotations(
                    (*visible_annotations, *visible_presentation_annotations),
                    image_transform=image_transform,
                    canvas_width=render_plan.render_view.width or 0,
                    canvas_height=render_plan.render_view.height or 0,
                ),
                transform=self._build_view_transform_payload(view),
                orientation=self._serialize_orientation_overlay(
                    self._build_stack_orientation_overlay(render_plan.render_view, cached.dataset)
                ),
            ),
            image_bytes=image_bytes,
        )
