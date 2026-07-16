from __future__ import annotations

"""3D volume and surface preprocessing and rendering."""

from app.services.viewer.shared import *  # noqa: F403


class ViewerVolumeMixin:
    def _build_volume_render_request(
        self,
        view: ViewRecord,
        *,
        volume: np.ndarray,
        spacing_xyz: tuple[float, float, float],
        fast_preview: bool,
        scale_fast_preview_canvas: bool = True,
        volume_token: str | None = None,
    ) -> VolumeRenderRequest:
        """Build the shared VTK request payload used by 3D render and drag paths."""

        canvas_width = view.width or 0
        canvas_height = view.height or 0
        offset_x = float(view.offset_x)
        offset_y = float(view.offset_y)
        if fast_preview and scale_fast_preview_canvas:
            preview_plan = self._resolve_volume_fast_preview_render_plan(
                canvas_width,
                canvas_height,
            )
            canvas_width = preview_plan.width
            canvas_height = preview_plan.height
            offset_x *= preview_plan.ratio
            offset_y *= preview_plan.ratio

        return VolumeRenderRequest(
            view_id=view.view_id,
            volume=volume,
            spacing_xyz=spacing_xyz,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            window_width=float(view.window_width or WINDOW_WIDTH_MIN),
            window_center=float(view.window_center or 0.0),
            zoom=float(view.zoom),
            offset_x=offset_x,
            offset_y=offset_y,
            rotation_quaternion=tuple(float(value) for value in view.rotation_quaternion),
            volume_preset=str(view.volume_preset or "bone"),
            volume_config=view.volume_render_config,
            fast_preview=fast_preview,
            volume_token=volume_token,
        )

    @staticmethod
    def _resolve_volume_fast_preview_render_plan(width: int, height: int) -> VolumePreviewRenderPlan:
        """Cap expensive 3D volume preview frames while final frames stay full size."""

        if width <= 0 or height <= 0:
            return VolumePreviewRenderPlan(width=width, height=height, ratio=1.0)
        max_dimension = max(width, height)
        if max_dimension <= VOLUME_FAST_PREVIEW_MAX_DIMENSION:
            return VolumePreviewRenderPlan(width=width, height=height, ratio=1.0)
        scale = VOLUME_FAST_PREVIEW_MAX_DIMENSION / float(max_dimension)
        return VolumePreviewRenderPlan(
            width=max(1, int(round(width * scale))),
            height=max(1, int(round(height * scale))),
            ratio=scale,
        )

    def _build_surface_render_request(
        self,
        view: ViewRecord,
        *,
        volume: np.ndarray,
        spacing_xyz: tuple[float, float, float],
        fast_preview: bool,
        volume_token: str | None = None,
        progress_callback: ViewRenderProgressCallback | None = None,
    ) -> SurfaceRenderRequest:
        """Build the shared VTK request payload used by 3D surface render and drag paths."""

        return SurfaceRenderRequest(
            view_id=view.view_id,
            volume=volume,
            spacing_xyz=spacing_xyz,
            canvas_width=view.width or 0,
            canvas_height=view.height or 0,
            zoom=float(view.zoom),
            offset_x=float(view.offset_x),
            offset_y=float(view.offset_y),
            rotation_quaternion=tuple(float(value) for value in view.rotation_quaternion),
            surface_config=view.surface_render_config,
            fast_preview=fast_preview,
            volume_token=volume_token,
            progress_callback=progress_callback,
        )

    def _prepare_3d_render_volume(
        self,
        view: ViewRecord,
        series: SeriesRecord,
        volume: np.ndarray,
        spacing_xyz: tuple[float, float, float],
        volume_token: str | None,
        progress_callback: ViewRenderProgressCallback | None = None,
    ) -> tuple[np.ndarray, str | None]:
        options_token = self._build_volume_render_options_token(view)
        if options_token == "default":
            return volume, volume_token

        base_identity = volume_token or series.volume_cache_key or series.series_id or id(volume)
        cache_key = (
            str(base_identity),
            tuple(int(size) for size in volume.shape),
            tuple(round(float(value), 6) for value in spacing_xyz),
            options_token,
        )
        cached = self._volume_render_preprocess_cache.get(cache_key)
        if cached is not None:
            self._volume_render_preprocess_cache.move_to_end(cache_key)
            logger.info(
                "3d preprocess cache hit view_id=%s remove_bed=%s clip_mode=%s options=%s",
                view.view_id,
                bool(view.volume_remove_bed),
                view.volume_clip_mode,
                options_token,
            )
            return cached, self._build_preprocessed_volume_token(volume_token, options_token)

        preprocess_started_at = perf_counter()
        prepared = np.asarray(volume).copy()
        if bool(view.volume_remove_bed):
            self._emit_render_progress(progress_callback, "preprocess", progress_percent=74, message="正在过滤床板...")
            prepared = self._remove_bed_from_render_volume(prepared)
        if view.volume_clip_mode in {"inside", "outside"} and len(view.volume_clip_points) >= 3:
            self._emit_render_progress(progress_callback, "preprocess", progress_percent=78, message="正在应用 3D 裁剪...")
            prepared = self._apply_3d_volume_clip(
                prepared,
                spacing_xyz=spacing_xyz,
                mode=str(view.volume_clip_mode),
                points=tuple(view.volume_clip_points),
                rotation_quaternion=tuple(float(value) for value in view.volume_clip_rotation_quaternion),
            )

        self._volume_render_preprocess_cache[cache_key] = prepared
        self._volume_render_preprocess_cache.move_to_end(cache_key)
        while len(self._volume_render_preprocess_cache) > 4:
            self._volume_render_preprocess_cache.popitem(last=False)
        logger.info(
            "3d preprocess complete view_id=%s remove_bed=%s clip_mode=%s options=%s elapsed_ms=%.1f shape=%s",
            view.view_id,
            bool(view.volume_remove_bed),
            view.volume_clip_mode,
            options_token,
            (perf_counter() - preprocess_started_at) * 1000.0,
            tuple(int(value) for value in prepared.shape),
        )
        return prepared, self._build_preprocessed_volume_token(volume_token, options_token)

    @staticmethod
    def _build_preprocessed_volume_token(volume_token: str | None, options_token: str) -> str:
        base = str(volume_token or "ndarray")
        digest = hashlib.sha1(options_token.encode("utf-8")).hexdigest()[:12]
        return f"{base}:render-options:{digest}"

    @staticmethod
    def _build_volume_render_options_token(view: ViewRecord) -> str:
        clip_payload: dict[str, object] | None = None
        if view.volume_clip_mode in {"inside", "outside"} and len(view.volume_clip_points) >= 3:
            clip_payload = {
                "mode": view.volume_clip_mode,
                "points": [
                    [round(float(point[0]), 5), round(float(point[1]), 5)]
                    for point in view.volume_clip_points
                ],
                "rotation": [round(float(value), 6) for value in view.volume_clip_rotation_quaternion],
            }
        if not bool(view.volume_remove_bed) and clip_payload is None:
            return "default"
        return json.dumps(
            {
                "removeBed": bool(view.volume_remove_bed),
                "clip": clip_payload,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    @staticmethod
    def _render_air_value(volume: np.ndarray) -> float:
        finite = np.asarray(volume)[np.isfinite(volume)]
        if finite.size == 0:
            return 0.0
        source_min = float(np.min(finite))
        source_max = float(np.max(finite))
        if source_min <= -300.0:
            return min(source_min, -1000.0)
        span = max(source_max - source_min, 1.0)
        return source_min - span * 0.05

    def _remove_bed_from_render_volume(self, volume: np.ndarray) -> np.ndarray:
        array = np.asarray(volume)
        if array.ndim != 3 or array.size == 0:
            return volume

        finite = np.isfinite(array)
        if not np.any(finite):
            return volume

        seed_upper, candidate_upper = self._resolve_bed_hu_thresholds(array, finite)
        seed_mask = finite & (array >= -520.0) & (array <= seed_upper)
        candidate_mask = finite & (array >= -520.0) & (array <= candidate_upper)
        high_density = finite & (array >= 350.0)
        protected_foreground = self._build_protected_foreground_mask(
            array,
            finite,
            seed_upper=seed_upper,
            candidate_upper=candidate_upper,
            high_density=high_density,
        )

        remove_mask = self._build_connected_bed_removal_mask(seed_mask, candidate_mask, high_density)
        remove_mask |= self._build_axis_bed_slab_mask(candidate_mask, high_density)
        candidate_count = int(np.count_nonzero(remove_mask))
        remove_mask &= candidate_mask & ~high_density & ~protected_foreground

        if not np.any(remove_mask):
            return volume

        filtered = np.asarray(volume).copy()
        filtered[remove_mask] = self._render_air_value(filtered)
        logger.info(
            "3d remove bed candidate_voxels=%s protected_voxels=%s filtered_voxels=%s fraction=%.6f shape=%s",
            candidate_count,
            int(np.count_nonzero(protected_foreground)),
            int(np.count_nonzero(remove_mask)),
            float(np.count_nonzero(remove_mask) / max(1, remove_mask.size)),
            tuple(int(value) for value in filtered.shape),
        )
        return filtered

    @staticmethod
    def _resolve_bed_hu_thresholds(volume: np.ndarray, finite: np.ndarray) -> tuple[float, float]:
        foreground = np.asarray(volume[finite & (volume > -700.0)], dtype=np.float32)
        if foreground.size < 32:
            foreground = np.asarray(volume[finite], dtype=np.float32)
        if foreground.size == 0:
            return 130.0, 180.0
        p25, p50, p75 = (float(value) for value in np.percentile(foreground, [25.0, 50.0, 75.0]))
        seed_upper = max(80.0, min(180.0, max(p25 + 28.0, p50 + 12.0)))
        candidate_upper = max(seed_upper + 24.0, min(230.0, max(p75 + 10.0, seed_upper + 36.0)))
        return seed_upper, candidate_upper

    def _build_protected_foreground_mask(
        self,
        volume: np.ndarray,
        finite: np.ndarray,
        *,
        seed_upper: float,
        candidate_upper: float,
        high_density: np.ndarray,
    ) -> np.ndarray:
        foreground_floor = max(88.0, min(125.0, seed_upper - 15.0))
        foreground_ceiling = max(candidate_upper + 45.0, seed_upper + 58.0)
        protected_seed = finite & (volume >= foreground_floor) & (volume <= foreground_ceiling)
        if not np.any(protected_seed):
            return high_density.copy()

        structure = ndimage.generate_binary_structure(3, 1)
        clean_seed = ndimage.binary_closing(protected_seed, structure=structure, iterations=1)
        clean_seed = ndimage.binary_opening(clean_seed, structure=structure, iterations=1)
        labeled, component_count = ndimage.label(clean_seed, structure=structure)
        if component_count <= 0:
            return high_density.copy()

        keep_labels: list[int] = []
        objects = ndimage.find_objects(labeled)
        shape = tuple(int(value) for value in volume.shape)
        min_component_voxels = max(64, int(round(volume.size * 0.00005)))
        for label_index, component_slice in enumerate(objects, start=1):
            if component_slice is None:
                continue
            component = labeled[component_slice] == label_index
            component_voxels = int(np.count_nonzero(component))
            if component_voxels < min_component_voxels:
                continue

            starts = [int(item.start or 0) for item in component_slice]
            stops = [int(item.stop or shape[axis]) for axis, item in enumerate(component_slice)]
            lengths = [max(1, stop - start) for start, stop in zip(starts, stops, strict=True)]
            span_fractions = [length / max(1, shape[axis]) for axis, length in enumerate(lengths)]
            bbox_volume = max(1, int(np.prod(lengths)))
            fill_fraction = component_voxels / bbox_volume
            sorted_spans = sorted(span_fractions)
            centroid = [
                (start + stop) / 2.0 / max(1, shape[axis])
                for axis, (start, stop) in enumerate(zip(starts, stops, strict=True))
            ]
            touches_boundary = any(
                start <= 1 or stop >= shape[axis] - 1
                for axis, (start, stop) in enumerate(zip(starts, stops, strict=True))
            )
            centered = sum(0.12 <= value <= 0.88 for value in centroid) >= 2
            thick_body = sorted_spans[1] >= 0.16 and sorted_spans[0] >= 0.055 and fill_fraction >= 0.08
            compact_body = sorted_spans[1] >= 0.12 and fill_fraction >= 0.16 and centered
            if (not touches_boundary and (thick_body or compact_body)) or (centered and thick_body and fill_fraction >= 0.12):
                keep_labels.append(label_index)

        if not keep_labels:
            return high_density.copy()

        keep_mask = np.isin(labeled, np.asarray(keep_labels, dtype=labeled.dtype))
        protected = ndimage.binary_dilation(keep_mask, structure=structure, iterations=2)
        return protected | high_density

    def _build_connected_bed_removal_mask(
        self,
        seed_mask: np.ndarray,
        candidate_mask: np.ndarray,
        high_density: np.ndarray,
    ) -> np.ndarray:
        if not np.any(seed_mask):
            return np.zeros(seed_mask.shape, dtype=bool)

        structure = ndimage.generate_binary_structure(3, 1)
        smoothed_seed = ndimage.binary_closing(seed_mask, structure=structure, iterations=1)
        labeled, component_count = ndimage.label(smoothed_seed, structure=structure)
        if component_count <= 0:
            return np.zeros(seed_mask.shape, dtype=bool)

        objects = ndimage.find_objects(labeled)
        keep_labels: list[int] = []
        shape = tuple(int(value) for value in seed_mask.shape)
        min_component_voxels = max(48, int(round(seed_mask.size * 0.00003)))
        for label_index, component_slice in enumerate(objects, start=1):
            if component_slice is None:
                continue
            component = labeled[component_slice] == label_index
            component_voxels = int(np.count_nonzero(component))
            if component_voxels < min_component_voxels:
                continue

            starts = [int(item.start or 0) for item in component_slice]
            stops = [int(item.stop or shape[axis]) for axis, item in enumerate(component_slice)]
            lengths = [max(1, stop - start) for start, stop in zip(starts, stops, strict=True)]
            span_fractions = [length / max(1, shape[axis]) for axis, length in enumerate(lengths)]
            bbox_volume = max(1, int(np.prod(lengths)))
            fill_fraction = component_voxels / bbox_volume
            max_span = max(span_fractions)
            min_span = min(span_fractions)
            sorted_spans = sorted(span_fractions)
            touches_boundary = any(start <= 1 or stop >= shape[axis] - 1 for axis, (start, stop) in enumerate(zip(starts, stops, strict=True)))
            near_shell = any(min(start, shape[axis] - stop) / max(1, shape[axis]) <= 0.08 for axis, (start, stop) in enumerate(zip(starts, stops, strict=True)))
            plane_like = max_span >= 0.48 and min_span <= 0.18 and fill_fraction <= 0.68
            rail_like = max_span >= 0.40 and sorted_spans[1] <= 0.24 and fill_fraction <= 0.58
            broad_edge_support = touches_boundary and max_span >= 0.55 and fill_fraction <= 0.44
            if (touches_boundary and (plane_like or rail_like or broad_edge_support)) or (near_shell and (plane_like or rail_like)):
                keep_labels.append(label_index)

        if not keep_labels:
            return np.zeros(seed_mask.shape, dtype=bool)

        keep_mask = np.isin(labeled, np.asarray(keep_labels, dtype=labeled.dtype))
        grown = ndimage.binary_dilation(keep_mask, structure=structure, iterations=1)
        return grown & candidate_mask & ~high_density

    def _build_axis_bed_slab_mask(self, candidate_mask: np.ndarray, high_density: np.ndarray) -> np.ndarray:
        remove_mask = np.zeros(candidate_mask.shape, dtype=bool)
        for axis in range(3):
            other_axes = tuple(index for index in range(3) if index != axis)
            plane_candidate_fraction = np.mean(candidate_mask, axis=other_axes)
            plane_high_fraction = np.mean(high_density, axis=other_axes)
            axis_length = int(candidate_mask.shape[axis])
            if axis_length < 8 or plane_candidate_fraction.size != axis_length:
                continue

            dynamic_threshold = max(0.07, float(np.nanpercentile(plane_candidate_fraction, 84.0)) * 0.78)
            candidates = (
                (plane_candidate_fraction >= dynamic_threshold)
                & (plane_high_fraction <= 0.018)
            )
            max_thickness = max(2, int(round(axis_length * 0.12)))
            for start, end in self._boolean_runs(candidates):
                if end - start > max_thickness:
                    continue
                edge_distance = min(start, axis_length - end) / max(1, axis_length)
                if edge_distance > 0.28:
                    continue
                selector: list[slice | np.ndarray] = [slice(None), slice(None), slice(None)]
                selector[axis] = slice(start, end)
                slab_selector = tuple(selector)
                remove_mask[slab_selector] |= candidate_mask[slab_selector]
        return remove_mask & ~high_density

    @staticmethod
    def _boolean_runs(mask: np.ndarray) -> list[tuple[int, int]]:
        runs: list[tuple[int, int]] = []
        start: int | None = None
        for index, value in enumerate(np.asarray(mask, dtype=bool).tolist()):
            if value and start is None:
                start = index
            elif not value and start is not None:
                runs.append((start, index))
                start = None
        if start is not None:
            runs.append((start, int(mask.size)))
        return runs

    def _apply_3d_volume_clip(
        self,
        volume: np.ndarray,
        *,
        spacing_xyz: tuple[float, float, float],
        mode: str,
        points: tuple[tuple[float, float], ...],
        rotation_quaternion: tuple[float, float, float, float],
    ) -> np.ndarray:
        array = np.asarray(volume)
        if array.ndim != 3 or len(points) < 3:
            return volume

        polygon = np.asarray(points, dtype=np.float64)
        if polygon.ndim != 2 or polygon.shape[1] != 2:
            return volume
        polygon = np.clip(polygon, 0.0, 1.0)
        if float(np.max(polygon[:, 0]) - np.min(polygon[:, 0])) < 1e-4:
            return volume
        if float(np.max(polygon[:, 1]) - np.min(polygon[:, 1])) < 1e-4:
            return volume
        polygon_min_x = float(np.min(polygon[:, 0]))
        polygon_max_x = float(np.max(polygon[:, 0]))
        polygon_min_y = float(np.min(polygon[:, 1]))
        polygon_max_y = float(np.max(polygon[:, 1]))

        depth, height, width = array.shape
        sx, sy, sz = (max(abs(float(value)), 1e-6) for value in spacing_xyz)
        col_coords = (np.arange(width, dtype=np.float64) - (width - 1.0) / 2.0) * sx
        row_coords = (np.arange(height, dtype=np.float64) - (height - 1.0) / 2.0) * sy
        slice_coords = (np.arange(depth, dtype=np.float64) - (depth - 1.0) / 2.0) * sz
        rotation = quaternion_to_rotation_matrix(rotation_quaternion)
        screen_bounds = self._build_3d_clip_screen_bounds(col_coords, row_coords, slice_coords, rotation)
        min_screen_x, max_screen_x, min_screen_up, max_screen_up = screen_bounds
        screen_width = max(max_screen_x - min_screen_x, 1e-6)
        screen_height = max(max_screen_up - min_screen_up, 1e-6)

        clipped = array.copy()
        air_value = self._render_air_value(clipped)
        keep_inside = str(mode).strip().lower() == "inside"
        chunk_depth = max(1, min(12, depth))
        clip_started_at = perf_counter()
        candidate_voxels = 0
        modified_voxels = 0
        for start in range(0, depth, chunk_depth):
            end = min(depth, start + chunk_depth)
            z = slice_coords[start:end][:, None, None]
            y = row_coords[None, :, None]
            x = col_coords[None, None, :]
            screen_x = rotation[0, 0] * x + rotation[0, 1] * y + rotation[0, 2] * z
            screen_up = rotation[2, 0] * x + rotation[2, 1] * y + rotation[2, 2] * z
            norm_x = (screen_x - min_screen_x) / screen_width
            norm_y = 1.0 - (screen_up - min_screen_up) / screen_height
            chunk = clipped[start:end]
            candidate = (
                (norm_x >= polygon_min_x)
                & (norm_x <= polygon_max_x)
                & (norm_y >= polygon_min_y)
                & (norm_y <= polygon_max_y)
            )
            if not np.any(candidate):
                if keep_inside:
                    modified_voxels += int(chunk.size)
                    chunk[...] = air_value
                continue

            candidate_shape = candidate.shape
            candidate_voxels += int(np.count_nonzero(candidate))
            candidate_x = np.broadcast_to(norm_x, candidate_shape)[candidate]
            candidate_y = np.broadcast_to(norm_y, candidate_shape)[candidate]
            candidate_inside = self._points_inside_polygon(candidate_x, candidate_y, polygon)
            if keep_inside:
                keep_mask = np.zeros(chunk.shape, dtype=bool)
                keep_mask[candidate] = candidate_inside
                modified_voxels += int(np.count_nonzero(~keep_mask))
                chunk[~keep_mask] = air_value
            else:
                remove_mask = np.zeros(chunk.shape, dtype=bool)
                remove_mask[candidate] = candidate_inside
                modified_voxels += int(np.count_nonzero(remove_mask))
                chunk[remove_mask] = air_value

        logger.info(
            "3d volume clip mode=%s points=%s candidate_voxels=%s modified_voxels=%s fraction=%.6f elapsed_ms=%.1f shape=%s",
            mode,
            len(points),
            candidate_voxels,
            modified_voxels,
            float(modified_voxels / max(1, array.size)),
            (perf_counter() - clip_started_at) * 1000.0,
            tuple(int(value) for value in array.shape),
        )
        return clipped

    @staticmethod
    def _build_3d_clip_screen_bounds(
        col_coords: np.ndarray,
        row_coords: np.ndarray,
        slice_coords: np.ndarray,
        rotation: np.ndarray,
    ) -> tuple[float, float, float, float]:
        corners = np.asarray(
            [
                (x, y, z)
                for x in (float(col_coords[0]), float(col_coords[-1]))
                for y in (float(row_coords[0]), float(row_coords[-1]))
                for z in (float(slice_coords[0]), float(slice_coords[-1]))
            ],
            dtype=np.float64,
        )
        rotated = corners @ rotation.T
        return (
            float(np.min(rotated[:, 0])),
            float(np.max(rotated[:, 0])),
            float(np.min(rotated[:, 2])),
            float(np.max(rotated[:, 2])),
        )

    @staticmethod
    def _points_inside_polygon(x: np.ndarray, y: np.ndarray, polygon: np.ndarray) -> np.ndarray:
        inside = np.zeros(np.broadcast_shapes(x.shape, y.shape), dtype=bool)
        x_values = np.broadcast_to(x, inside.shape)
        y_values = np.broadcast_to(y, inside.shape)
        j = polygon.shape[0] - 1
        for i in range(polygon.shape[0]):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            crosses = (yi > y_values) != (yj > y_values)
            denominator = yj - yi
            if abs(float(denominator)) < 1e-12:
                denominator = 1e-12
            x_intersection = (xj - xi) * (y_values - yi) / denominator + xi
            inside ^= crosses & (x_values < x_intersection)
            j = i
        return inside

    def _render_3d_view(
        self,
        view: ViewRecord,
        image_format: ImageFormat = "webp",
        *,
        fast_preview: bool = False,
        progress_callback: ViewRenderProgressCallback | None = None,
    ) -> RenderedImageResult:
        render_started_at = perf_counter()
        ensure_view_size(view)

        volume_started_at = perf_counter()
        series = compat.series_registry.get(view.series_id)
        self._emit_render_progress(progress_callback, "volume", progress_percent=6)
        volume = self._get_series_volume(series, progress_callback=progress_callback)
        volume_token = self._build_series_volume_cache_key(series)
        volume_ms = (perf_counter() - volume_started_at) * 1000.0
        if not view.is_initialized:
            self._emit_render_progress(progress_callback, "initialize", progress_percent=72)
            self._initialize_3d_viewport(view)
            view.is_initialized = True

        spacing_xyz = self._get_3d_spacing_xyz(series)
        render_volume, render_volume_token = self._prepare_3d_render_volume(
            view,
            series,
            volume,
            spacing_xyz,
            volume_token,
            progress_callback=progress_callback,
        )
        render_3d_mode = self._normalize_render_3d_mode(view.render_3d_mode)
        image_started_at = perf_counter()
        if render_3d_mode == "surface":
            self._emit_render_progress(
                progress_callback,
                "preprocess",
                progress_percent=80,
                message="正在准备 Surface 数据...",
            )
            self._resolve_surface_render_config_for_render(
                view,
                series=series,
                volume=volume,
                volume_token=volume_token,
            )
            surface_request = self._build_surface_render_request(
                view,
                volume=render_volume,
                spacing_xyz=spacing_xyz,
                fast_preview=fast_preview,
                volume_token=render_volume_token,
                progress_callback=progress_callback,
            )
            image = compat._get_vtk_surface_renderer().render(surface_request)
            vtk_timings = compat._get_vtk_surface_renderer().get_last_timings(view.view_id)
            if not fast_preview:
                self._warm_surface_preview_session(surface_request)
            viewport_label = "3D SR"
            render_width = surface_request.canvas_width
            render_height = surface_request.canvas_height
        else:
            self._emit_render_progress(progress_callback, "render", progress_percent=82)
            self._resolve_volume_render_config_for_render(
                view,
                series=series,
                volume=volume,
                volume_token=volume_token,
            )
            volume_request = self._build_volume_render_request(
                view,
                volume=render_volume,
                spacing_xyz=spacing_xyz,
                fast_preview=fast_preview,
                volume_token=render_volume_token,
            )
            image = compat._get_vtk_volume_renderer().render(volume_request)
            vtk_timings = compat._get_vtk_volume_renderer().get_last_timings(view.view_id)
            viewport_label = "3D VR"
            render_width = volume_request.canvas_width
            render_height = volume_request.canvas_height
        image_ms = (perf_counter() - image_started_at) * 1000.0

        metadata_started_at = perf_counter()
        corner_info = self._build_slice_corner_info_overlay(
            view,
            series,
            None,
            current_index=view.current_index,
            total_slices=max(1, volume.shape[0]),
            viewport_label=viewport_label,
        )
        metadata_ms = (perf_counter() - metadata_started_at) * 1000.0

        self._emit_render_progress(progress_callback, "encode", progress_percent=96)
        encode_started_at = perf_counter()
        image_bytes = self._encode_3d_image(image, image_format, fast_preview=fast_preview)
        encode_ms = (perf_counter() - encode_started_at) * 1000.0

        performance_timings = vtk_timings.as_dict()
        performance_timings["webp_encode_ms"] = encode_ms
        performance_timings["viewer_render_ms"] = (perf_counter() - render_started_at) * 1000.0

        log_method = logger.debug if fast_preview else logger.info
        log_method(
            "3d render timing view_id=%s mode=%s fast_preview=%s image_format=%s source_shape=%s source_dtype=%s vtk_dtype=%s viewport=%sx%s render=%sx%s image=%sx%s volume_ms=%.1f session_ms=%.1f configure_ms=%.1f vtk_render_ms=%.1f gpu_readback_ms=%.1f gpu_ipc_ms=%.1f metadata_ms=%.1f webp_encode_ms=%.1f total_ms=%.1f",
            view.view_id,
            render_3d_mode,
            fast_preview,
            image_format,
            volume.shape,
            performance_timings.get("source_dtype", ""),
            performance_timings.get("vtk_dtype", ""),
            view.width,
            view.height,
            render_width,
            render_height,
            image.width,
            image.height,
            volume_ms,
            float(performance_timings.get("session_ms", 0.0)),
            float(performance_timings.get("configure_ms", 0.0)),
            float(performance_timings.get("vtk_render_ms", 0.0)),
            float(performance_timings.get("gpu_readback_ms", 0.0)),
            float(performance_timings.get("ipc_ms", 0.0)),
            metadata_ms,
            encode_ms,
            (perf_counter() - render_started_at) * 1000.0,
        )

        return RenderedImageResult(
            meta=ViewImageResponse(
                slice_info=SliceInfo(current=view.current_index, total=max(1, volume.shape[0])),
                window_info=WindowInfo(ww=view.window_width, wl=view.window_center),
                imageFormat=image_format,
                viewId=view.view_id,
                color=ViewColorInfo(pseudocolorPreset=view.pseudocolor_preset),
                cornerInfo=self._serialize_corner_info_overlay(corner_info),
                orientation=self._build_3d_orientation_overlay(view),
                transform=self._build_view_transform_payload(view),
                volumePreset=str(view.volume_preset or "bone"),
                volumeConfig=view.volume_render_config,
                render3dMode=render_3d_mode,
                surfaceConfig=view.surface_render_config,
                volumeRenderOptions=self._build_volume_render_options_response(view),
            ),
            image_bytes=image_bytes,
            performance_timings=performance_timings,
        )

    def _resolve_surface_render_config_for_render(
        self,
        view: ViewRecord,
        *,
        series: SeriesRecord,
        volume: np.ndarray,
        volume_token: str | None,
    ) -> dict[str, object]:
        existing_preset = "bone"
        if isinstance(view.surface_render_config, dict):
            existing_preset = str(view.surface_render_config.get("preset") or existing_preset)
        preset = normalize_surface_preset_name(existing_preset)
        source = str(view.surface_render_config_source or "manual").strip().lower()
        config_token = self._build_volume_render_config_token(
            preset=f"surface:{preset}",
            series=series,
            volume=volume,
            volume_token=volume_token,
        )

        if source == "preset":
            if view.surface_render_config is not None and view.surface_render_config_token == config_token:
                return view.surface_render_config
            view.surface_render_config = create_adaptive_surface_render_config(
                preset,
                volume,
                modality=series.modality,
            )
            view.surface_render_config_source = "preset"
            view.surface_render_config_token = config_token
            return view.surface_render_config

        if view.surface_render_config is None:
            view.surface_render_config = create_adaptive_surface_render_config(
                preset,
                volume,
                modality=series.modality,
            )
            view.surface_render_config_source = "preset"
            view.surface_render_config_token = config_token
            return view.surface_render_config

        view.surface_render_config = normalize_surface_render_config(view.surface_render_config, preset)
        view.surface_render_config_source = "manual"
        view.surface_render_config_token = None
        return view.surface_render_config

    def _resolve_volume_render_config_for_render(
        self,
        view: ViewRecord,
        *,
        series: SeriesRecord,
        volume: np.ndarray,
        volume_token: str | None,
    ) -> dict[str, object]:
        preset = normalize_volume_preset_name(view.volume_preset or "bone")
        source = str(view.volume_render_config_source or "manual").strip().lower()
        config_token = self._build_volume_render_config_token(
            preset=preset,
            series=series,
            volume=volume,
            volume_token=volume_token,
        )

        if source == "preset":
            if view.volume_render_config is not None and view.volume_render_config_token == config_token:
                return view.volume_render_config
            view.volume_render_config = create_adaptive_volume_render_config(
                preset,
                volume,
                modality=series.modality,
            )
            view.volume_preset = preset
            view.volume_render_config_source = "preset"
            view.volume_render_config_token = config_token
            return view.volume_render_config

        if view.volume_render_config is None:
            view.volume_render_config = create_default_volume_render_config(preset)
            view.volume_render_config_source = "preset"
            view.volume_render_config_token = config_token
            return view.volume_render_config

        view.volume_render_config = normalize_volume_render_config(view.volume_render_config, preset)
        view.volume_preset = str(view.volume_render_config.get("preset", preset))
        view.volume_render_config_source = "manual"
        view.volume_render_config_token = None
        return view.volume_render_config

    @staticmethod
    def _build_volume_render_config_token(
        *,
        preset: str,
        series: SeriesRecord,
        volume: np.ndarray,
        volume_token: str | None,
    ) -> str:
        identity = volume_token or series.volume_cache_key or series.series_id
        shape = "x".join(str(int(size)) for size in volume.shape)
        modality = str(series.modality or "").strip().upper()
        return f"{preset}:{identity}:{shape}:{modality}"

    def _warm_surface_preview_session(self, request: SurfaceRenderRequest) -> None:
        try:
            compat._get_vtk_surface_renderer().warm_preview_session(request)
        except Exception:
            logger.debug("failed to schedule surface preview warmup view_id=%s", request.view_id, exc_info=True)
