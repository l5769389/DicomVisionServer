from __future__ import annotations

"""View initialization, reset, PET, and shared state handling."""

from app.services.viewer.shared import *  # noqa: F403


class ViewerStateMixin:
    def _render_by_view_type(
        self,
        view: ViewRecord,
        image_format: ImageFormat = "webp",
        *,
        fast_preview: bool = False,
        fast_preview_full_resolution: bool = False,
        metadata_mode: str = "full",
        progress_callback: ViewRenderProgressCallback | None = None,
        raw_3d_output: bool = False,
    ) -> RenderedImageResult:
        return render_by_view_type(
            self,
            view,
            image_format=image_format,
            fast_preview=fast_preview,
            fast_preview_full_resolution=fast_preview_full_resolution,
            metadata_mode=metadata_mode,
            progress_callback=progress_callback,
            raw_3d_output=raw_3d_output,
        )

    def _emit_render_progress(
        self,
        progress_callback: ViewRenderProgressCallback | None,
        phase: str,
        *,
        progress_percent: int | float | None = None,
        loaded_count: int | None = None,
        total_count: int | None = None,
        message: str | None = None,
    ) -> None:
        if progress_callback is None:
            return

        payload: dict[str, object] = {"phase": phase}
        if progress_percent is not None:
            payload["progressPercent"] = max(0, min(100, int(round(float(progress_percent)))))
        if loaded_count is not None:
            payload["loadedCount"] = max(0, int(loaded_count))
        if total_count is not None:
            payload["totalCount"] = max(0, int(total_count))
        if message:
            payload["message"] = message

        try:
            progress_callback(payload)
        except Exception:
            logger.debug("render progress callback failed", exc_info=True)

    def _handle_scroll(self, view: ViewRecord, series: SeriesRecord, scroll: int) -> None:
        if not self._is_mpr_view_type(view.view_type):
            next_index = view.current_index + scroll
            view.current_index = max(0, min(next_index, len(series.instances) - 1))
            return

        volume = self._get_series_volume(series)
        target_viewport = self._resolve_mpr_viewport(view)
        if view.view_group is not None:
            group = view.view_group
            pose_context = self._build_mpr_pose_context(view, volume.shape, series=series)
            plane_pose = pose_context.poses[target_viewport]
            delta_world = (
                np.asarray(plane_pose.normal_world, dtype=np.float64)
                * spacing_along_world_direction(pose_context.geometry, plane_pose.normal_world)
                * float(scroll)
            )
            next_cursor = translate_cursor(pose_context.cursor, delta_world, pose_context.geometry)
            self._sync_group_from_mpr_cursor(group, next_cursor, pose_context.geometry, volume.shape)
        else:
            depth, height, width = volume.shape
            if target_viewport == MPR_VIEWPORT_CORONAL:
                view.mpr_coronal_index = max(0, min(view.mpr_coronal_index + scroll, height - 1))
            elif target_viewport == MPR_VIEWPORT_SAGITTAL:
                view.mpr_sagittal_index = max(0, min(view.mpr_sagittal_index + scroll, width - 1))
            else:
                view.mpr_axial_index = max(0, min(view.mpr_axial_index + scroll, depth - 1))
        view.is_initialized = True

    def _initialize_viewport(self, view: ViewRecord) -> None:
        ensure_view_size(view)

        series = compat.series_registry.get(view.series_id)
        view.current_index = self._resolve_representative_stack_index(series)
        instance = series.instances[view.current_index]
        if not instance.sop_instance_uid:
            raise HTTPException(status_code=400, detail="DICOM instance does not contain SOPInstanceUID")

        cached = compat.dicom_cache.get(instance.sop_instance_uid, instance.path)
        image_height, image_width = cached.source_pixels.shape[:2]
        view.zoom = compat.viewport_transformer.calculate_contain_zoom(
            image_width=image_width,
            image_height=image_height,
            canvas_width=view.width,
            canvas_height=view.height,
        )
        view.offset_x = 0.0
        view.offset_y = 0.0
        view.rotation_degrees = 0
        view.pseudocolor_preset = DEFAULT_PSEUDOCOLOR_PRESET
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

    @staticmethod
    def _is_pet_series(series: SeriesRecord | None) -> bool:
        return str(series.modality or "").strip().upper() in {"PT", "PET"} if series is not None else False

    def _initialize_pet_viewport(self, view: ViewRecord) -> None:
        ensure_view_size(view)

        series = compat.series_registry.get(view.series_id)
        if not self._is_pet_series(series):
            raise HTTPException(status_code=400, detail="PET view requires a PT/PET series")
        if not series.instances:
            raise HTTPException(status_code=400, detail="PET series does not contain image instances")

        pet_volume = self._get_series_volume(series)
        pet_display = self._build_fusion_pet_display_volume(series, pet_volume, view.pet_unit)
        view.pet_unit = pet_display.unit
        view.pet_unit_label = pet_display.unit_label
        view.current_index = max(0, min(self._resolve_representative_stack_index(series), pet_display.volume.shape[0] - 1))
        image_height = int(pet_display.volume.shape[1]) if pet_display.volume.ndim >= 2 else 1
        image_width = int(pet_display.volume.shape[2]) if pet_display.volume.ndim >= 3 else 1
        view.zoom = compat.viewport_transformer.calculate_contain_zoom(
            image_width=image_width,
            image_height=image_height,
            canvas_width=view.width,
            canvas_height=view.height,
        )
        view.offset_x = 0.0
        view.offset_y = 0.0
        view.rotation_degrees = 0
        view.hor_flip = False
        view.ver_flip = False
        view.pseudocolor_preset = PET_STANDALONE_PSEUDOCOLOR_PRESET
        view.window_width = FUSION_DEFAULT_SUV_WINDOW_MAX - FUSION_DEFAULT_SUV_WINDOW_MIN
        view.window_center = (FUSION_DEFAULT_SUV_WINDOW_MAX + FUSION_DEFAULT_SUV_WINDOW_MIN) / 2.0
        self._reset_drag_state(view)
        logger.info(
            "PET viewport initialized view_id=%s volume=%s unit=%s zoom=%.4f ww=%s wl=%s",
            view.view_id,
            tuple(int(value) for value in pet_display.volume.shape),
            view.pet_unit,
            view.zoom,
            view.window_width,
            view.window_center,
        )

    def _initialize_mpr_viewport(self, view: ViewRecord) -> None:
        ensure_view_size(view)

        series = compat.series_registry.get(view.series_id)
        volume = self._get_series_volume(series)
        if view.view_group is not None:
            if view.view_group.mpr_cursor is None:
                self._reset_mpr_group_geometry(view.view_group, volume.shape, series=series)
        else:
            depth, height, width = volume.shape
            view.mpr_axial_index = depth // 2
            view.mpr_coronal_index = height // 2
            view.mpr_sagittal_index = width // 2
        self._reset_mpr_view_display_state(view)
        self._reset_mpr_view_window(view, series, volume)
        self._fit_mpr_view_to_plane(view, series, volume)
        logger.info(
            "mpr viewport initialized view_id=%s volume=%s axial=%s coronal=%s sagittal=%s zoom=%.4f",
            view.view_id,
            volume.shape,
            view.mpr_axial_index,
            view.mpr_coronal_index,
            view.mpr_sagittal_index,
            view.zoom,
        )

    def _sync_mpr_state_from_source_view(
        self,
        target_view: ViewRecord,
        source_view_id: str,
        workspace_id: str | None = None,
    ) -> bool:
        if not self._is_mpr_view_type(target_view.view_type) or target_view.view_group is None:
            return False

        source_view = (
            compat.view_registry.get(source_view_id)
            if workspace_id is None
            else compat.view_registry.get(source_view_id, workspace_id=workspace_id)
        )
        if not self._is_mpr_view_type(source_view.view_type) or source_view.view_group is None:
            return False
        if source_view.view_group.group_id == target_view.view_group.group_id:
            return False

        source_series = (
            compat.series_registry.get(source_view.series_id)
            if workspace_id is None
            else compat.series_registry.get(source_view.series_id, workspace_id=workspace_id)
        )
        target_series = (
            compat.series_registry.get(target_view.series_id)
            if workspace_id is None
            else compat.series_registry.get(target_view.series_id, workspace_id=workspace_id)
        )
        logger.info(
            "mpr state sync source_view_id=%s source_series_id=%s target_view_id=%s target_series_id=%s",
            source_view.view_id,
            source_view.series_id,
            target_view.view_id,
            target_view.series_id,
        )
        source_volume = self._get_series_volume(source_series)
        target_volume = self._get_series_volume(target_series)
        source_context = self._build_mpr_pose_context(source_view, source_volume.shape, series=source_series)
        target_geometry = self._get_series_volume_geometry(target_series, target_volume.shape)
        source_group = source_view.view_group
        target_group = target_view.view_group

        target_group.active_viewport = source_group.active_viewport
        target_group.crosshair_drag_active = False
        target_group.crosshair_drag_origin_center = None
        target_group.crosshair_drag_origin_image = None
        target_group.rotation_drag = None
        target_group.mpr_crosshair_angles = deepcopy(source_group.mpr_crosshair_angles)
        target_group.mpr_crosshair_mode = self._normalize_mpr_crosshair_mode(source_group.mpr_crosshair_mode)
        target_group.mpr_independent_plane_normals = deepcopy(source_group.mpr_independent_plane_normals)
        target_group.mpr_mip = deepcopy(source_group.mpr_mip)
        target_group.mpr_segmentation = deepcopy(source_group.mpr_segmentation)
        target_group.mpr_use_display_basis_for_cursor_offsets = bool(source_group.mpr_use_display_basis_for_cursor_offsets)
        target_group.mpr_model_rotation_world = deepcopy(source_group.mpr_model_rotation_world)
        target_group.mpr_model_rotation_pivot_world = deepcopy(source_group.mpr_model_rotation_pivot_world)
        self._sync_group_from_mpr_cursor(target_group, source_context.cursor, target_geometry, target_volume.shape)
        if target_view.width and target_view.height:
            target_view.is_initialized = True
        return True

    def _initialize_3d_viewport(self, view: ViewRecord) -> None:
        ensure_view_size(view)

        series = compat.series_registry.get(view.series_id)
        volume = self._get_series_volume(series)
        view.current_index = self._resolve_representative_stack_index(series)

        first_instance = next((instance for instance in series.instances if instance.sop_instance_uid), None)
        if first_instance is not None and first_instance.sop_instance_uid:
            cached = compat.dicom_cache.get(first_instance.sop_instance_uid, first_instance.path)
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
        view.rotation_quaternion = compat._get_vtk_volume_renderer().get_default_rotation_quaternion()
        view.pseudocolor_preset = DEFAULT_PSEUDOCOLOR_PRESET
        stats = build_volume_intensity_stats(volume, modality=series.modality)
        default_volume_preset = select_default_volume_preset(series, volume, stats=stats)
        view.volume_preset = default_volume_preset
        view.volume_render_config = create_adaptive_volume_render_config(
            default_volume_preset,
            volume,
            modality=series.modality,
            stats=stats,
        )
        view.volume_render_config_source = "preset"
        view.volume_render_config_token = None
        view.render_3d_mode = "volume"
        view.surface_render_config = create_adaptive_surface_render_config(
            "bone",
            volume,
            modality=series.modality,
            stats=stats,
        )
        view.surface_render_config_source = "preset"
        view.surface_render_config_token = None
        view.volume_remove_bed = False
        view.volume_clip_mode = None
        view.volume_clip_points = ()
        view.volume_clip_rotation_quaternion = compat._get_vtk_volume_renderer().get_default_rotation_quaternion()
        self._reset_drag_state(view)
        logger.info(
            "3d viewport initialized view_id=%s volume=%s preset=%s zoom=%.4f ww=%s wl=%s",
            view.view_id,
            volume.shape,
            view.volume_preset,
            view.zoom,
            view.window_width,
            view.window_center,
        )

    def _resolve_fusion_group_series(self, view: ViewRecord) -> tuple[ViewGroupRecord, SeriesRecord, SeriesRecord]:
        group = view.view_group
        if group is None or str(group.group_type).lower() != "fusion":
            raise HTTPException(status_code=400, detail="Fusion view is missing shared group state")
        ct_series_id = group.fusion_ct_series_id or view.series_id
        pet_series_id = group.fusion_pet_series_id or view.secondary_series_id
        if not pet_series_id:
            raise HTTPException(status_code=400, detail="Fusion view is missing PET series")
        ct_series = compat.series_registry.get(ct_series_id, workspace_id=view.workspace_id)
        pet_series = compat.series_registry.get(pet_series_id, workspace_id=view.workspace_id)
        return group, ct_series, pet_series

    @staticmethod
    def _normalize_fusion_pet_unit(value: str | None) -> str:
        normalized = str(value or FUSION_PET_UNIT_SUV_BW).strip()
        aliases = {
            "raw": FUSION_PET_UNIT_SOURCE,
            "source": FUSION_PET_UNIT_SOURCE,
            "BQML": FUSION_PET_UNIT_SOURCE,
            "kBq/ml": FUSION_PET_UNIT_KBQML,
            "kBqml": FUSION_PET_UNIT_KBQML,
            "uptake": FUSION_PET_UNIT_KBQML,
            "SUV": FUSION_PET_UNIT_SUV_BW,
            "SUVbw": FUSION_PET_UNIT_SUV_BW,
            "GML": FUSION_PET_UNIT_SUV_BW,
            "SUVbsa": FUSION_PET_UNIT_SUV_BSA,
            "SUL": FUSION_PET_UNIT_SUL,
            "%ID/g": FUSION_PET_UNIT_PERCENT_ID_G,
            "percentIDg": FUSION_PET_UNIT_PERCENT_ID_G,
        }
        return aliases.get(normalized, aliases.get(normalized.upper(), FUSION_PET_UNIT_SUV_BW))

    @staticmethod
    def _parse_dicom_datetime(date_value: object | None, time_value: object | None = None) -> datetime | None:
        if date_value is None and time_value is None:
            return None
        date_text = str(date_value or "").strip()
        time_text = str(time_value or "").strip()
        text = f"{date_text}{time_text}" if time_text else date_text
        text = text.replace(" ", "").replace(":", "")
        if "." in text:
            head, tail = text.split(".", 1)
            text = f"{head}.{''.join(ch for ch in tail if ch.isdigit())}"
        else:
            text = "".join(ch for ch in text if ch.isdigit())
        if time_text:
            date_digits = "".join(ch for ch in date_text if ch.isdigit())
            time_digits = "".join(ch for ch in time_text if ch.isdigit())
            text = f"{date_digits}{time_digits}"
        if not text:
            return None

        if "." in text:
            main_text, fractional_text = text.split(".", 1)
            text = f"{main_text}{fractional_text[:6].ljust(6, '0')}"
            formats = ("%Y%m%d%H%M%S%f",)
        else:
            formats = ("%Y%m%d%H%M%S", "%Y%m%d%H%M", "%Y%m%d")
        for fmt in formats:
            expected_len = len(datetime(2000, 1, 1, 1, 1, 1, 123456).strftime(fmt))
            try:
                return datetime.strptime(text[:expected_len], fmt)
            except Exception:
                continue
        return None

    @staticmethod
    def _get_first_sequence_item(dataset: Dataset | None, name: str) -> Dataset | None:
        if dataset is None:
            return None
        sequence = getattr(dataset, name, None)
        try:
            return sequence[0] if sequence else None
        except Exception:
            return None

    def _resolve_pet_decay_corrected_dose_bq(self, dataset: Dataset | None) -> float | None:
        radiopharmaceutical = self._get_first_sequence_item(dataset, "RadiopharmaceuticalInformationSequence")
        if dataset is None or radiopharmaceutical is None:
            return None
        dose = self._safe_float(getattr(radiopharmaceutical, "RadionuclideTotalDose", None))
        if dose is None or dose <= 0.0:
            return None

        corrected_value = getattr(dataset, "CorrectedImage", []) or []
        corrected_image = (
            str(corrected_value).upper()
            if isinstance(corrected_value, str)
            else " ".join(str(value).upper() for value in corrected_value)
        )
        decay_correction = str(getattr(dataset, "DecayCorrection", "") or "").upper()
        half_life = self._safe_float(getattr(radiopharmaceutical, "RadionuclideHalfLife", None))
        if "DECY" not in corrected_image or decay_correction not in {"START", "NONE"} or half_life is None or half_life <= 0.0:
            return float(dose)

        injection_datetime = self._parse_dicom_datetime(
            getattr(radiopharmaceutical, "RadiopharmaceuticalStartDateTime", None),
            None,
        ) or self._parse_dicom_datetime(
            getattr(dataset, "SeriesDate", None) or getattr(dataset, "AcquisitionDate", None) or getattr(dataset, "StudyDate", None),
            getattr(radiopharmaceutical, "RadiopharmaceuticalStartTime", None),
        )
        scan_datetime = self._parse_dicom_datetime(
            getattr(dataset, "AcquisitionDateTime", None),
            None,
        ) or self._parse_dicom_datetime(
            getattr(dataset, "AcquisitionDate", None) or getattr(dataset, "SeriesDate", None) or getattr(dataset, "StudyDate", None),
            getattr(dataset, "AcquisitionTime", None) or getattr(dataset, "SeriesTime", None) or getattr(dataset, "StudyTime", None),
        )
        if injection_datetime is None or scan_datetime is None:
            return float(dose)

        elapsed_seconds = max(0.0, (scan_datetime - injection_datetime).total_seconds())
        return float(dose) * float(np.exp(-np.log(2.0) * elapsed_seconds / float(half_life)))

    @staticmethod
    def _safe_float(value: object | None) -> float | None:
        if value is None:
            return None
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result if np.isfinite(result) else None

    def _resolve_pet_display_scale(self, dataset: Dataset | None, requested_unit: str) -> tuple[float, str, str]:
        source_units = str(getattr(dataset, "Units", "") or "").strip().upper() if dataset is not None else ""
        unit = self._normalize_fusion_pet_unit(requested_unit)
        if unit == FUSION_PET_UNIT_SOURCE:
            return (1.0, FUSION_PET_UNIT_SOURCE, source_units or FUSION_PET_UNIT_LABELS[FUSION_PET_UNIT_SOURCE])

        if unit == FUSION_PET_UNIT_KBQML:
            if source_units == "BQML":
                return (0.001, unit, FUSION_PET_UNIT_LABELS[unit])
            return (1.0, FUSION_PET_UNIT_SOURCE, source_units or FUSION_PET_UNIT_LABELS[FUSION_PET_UNIT_SOURCE])

        if source_units in {"GML", "SUVBW"} and unit == FUSION_PET_UNIT_SUV_BW:
            return (1.0, FUSION_PET_UNIT_SUV_BW, FUSION_PET_UNIT_LABELS[FUSION_PET_UNIT_SUV_BW])
        if source_units != "BQML":
            return (1.0, FUSION_PET_UNIT_SOURCE, source_units or FUSION_PET_UNIT_LABELS[FUSION_PET_UNIT_SOURCE])

        dose = self._resolve_pet_decay_corrected_dose_bq(dataset)
        if dose is None or dose <= 0.0:
            return (1.0, FUSION_PET_UNIT_SOURCE, source_units or FUSION_PET_UNIT_LABELS[FUSION_PET_UNIT_SOURCE])

        weight_kg = self._safe_float(getattr(dataset, "PatientWeight", None))
        height_m = self._safe_float(getattr(dataset, "PatientSize", None))
        sex = str(getattr(dataset, "PatientSex", "") or "").upper()
        if unit == FUSION_PET_UNIT_SUV_BW and weight_kg is not None and weight_kg > 0.0:
            return ((weight_kg * 1000.0) / dose, unit, FUSION_PET_UNIT_LABELS[unit])
        if unit == FUSION_PET_UNIT_SUV_BSA and weight_kg is not None and weight_kg > 0.0 and height_m is not None and height_m > 0.0:
            height_cm = height_m * 100.0
            bsa_cm2 = 0.007184 * (height_cm ** 0.725) * (weight_kg ** 0.425) * 10000.0
            return (bsa_cm2 / dose, unit, FUSION_PET_UNIT_LABELS[unit])
        if unit == FUSION_PET_UNIT_SUL and weight_kg is not None and weight_kg > 0.0 and height_m is not None and height_m > 0.0:
            height_cm = height_m * 100.0
            if sex == "F":
                lbm_kg = 1.07 * weight_kg - 148.0 * ((weight_kg / height_cm) ** 2)
            else:
                lbm_kg = 1.10 * weight_kg - 128.0 * ((weight_kg / height_cm) ** 2)
            if lbm_kg > 0.0:
                return ((lbm_kg * 1000.0) / dose, unit, FUSION_PET_UNIT_LABELS[unit])
        if unit == FUSION_PET_UNIT_PERCENT_ID_G:
            return (100.0 / dose, unit, FUSION_PET_UNIT_LABELS[unit])
        return (1.0, FUSION_PET_UNIT_SOURCE, source_units or FUSION_PET_UNIT_LABELS[FUSION_PET_UNIT_SOURCE])

    def _build_fusion_pet_display_volume(
        self,
        pet_series: SeriesRecord,
        pet_volume: np.ndarray,
        requested_unit: str | None,
    ) -> FusionPetDisplayVolume:
        _, cached = self._get_reference_instance_and_cache(pet_series)
        scale, actual_unit, actual_label = self._resolve_pet_display_scale(
            cached.dataset if cached is not None else None,
            requested_unit or FUSION_PET_UNIT_SUV_BW,
        )
        if abs(scale - 1.0) <= 1e-12:
            display_volume = pet_volume
        else:
            display_volume = np.asarray(pet_volume, dtype=np.float32) * np.float32(scale)
        source_units = str(getattr(cached.dataset, "Units", "") or "").strip() if cached is not None else None
        return FusionPetDisplayVolume(
            volume=display_volume,
            unit=actual_unit,
            unit_label=actual_label,
            source_units=source_units or None,
            scale=float(scale),
        )

    def _derive_default_pet_window_for_display_volume(
        self,
        display: FusionPetDisplayVolume,
    ) -> tuple[float, float]:
        finite = np.asarray(display.volume, dtype=np.float32)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return (1.0, 0.5)
        if display.unit in {FUSION_PET_UNIT_SUV_BW, FUSION_PET_UNIT_SUV_BSA, FUSION_PET_UNIT_SUL}:
            window_width = FUSION_DEFAULT_SUV_WINDOW_MAX - FUSION_DEFAULT_SUV_WINDOW_MIN
            window_center = (FUSION_DEFAULT_SUV_WINDOW_MAX + FUSION_DEFAULT_SUV_WINDOW_MIN) / 2.0
            return (window_width, window_center)
        positive = finite[finite > 0.0]
        pet_window_values = positive if positive.size else finite
        low = 0.0 if float(np.nanmin(finite)) >= 0.0 else float(np.nanpercentile(finite, 1.0))
        high = float(np.nanpercentile(pet_window_values, 99.5))
        if not np.isfinite(high) or high <= low:
            high = low + 1.0
        return (max(WINDOW_WIDTH_MIN, high - low), (high + low) / 2.0)

    @staticmethod
    def _prepare_pet_standalone_source_pixels(
        source_pixels: np.ndarray,
        window_width: float | None,
        window_center: float | None,
    ) -> np.ndarray:
        low = compat.ViewerService._resolve_window_min(window_width, window_center)
        high = compat.ViewerService._resolve_window_max(window_width, window_center)
        if low is None or high is None or not np.isfinite(low) or not np.isfinite(high) or high <= low:
            return source_pixels

        pixels = np.asarray(source_pixels, dtype=np.float32)
        if pixels.ndim < 2 or pixels.size == 0:
            return pixels

        edge_pixels = np.concatenate(
            (
                pixels[0, :].ravel(),
                pixels[-1, :].ravel(),
                pixels[:, 0].ravel(),
                pixels[:, -1].ravel(),
            )
        )
        edge_pixels = edge_pixels[np.isfinite(edge_pixels)]
        if edge_pixels.size == 0:
            return pixels

        window_span = float(high) - float(low)
        edge_threshold = float(np.nanpercentile(edge_pixels, 75.0))
        threshold = min(edge_threshold, float(low) + window_span * 0.35)
        if not np.isfinite(threshold) or threshold <= float(low):
            return pixels

        background_value = float(low) - max(1.0, window_span * 0.02)
        prepared = pixels.copy()
        prepared[prepared <= threshold] = background_value
        return prepared

    def _derive_default_window_for_volume(self, series: SeriesRecord, volume: np.ndarray) -> tuple[float, float]:
        first_instance = next((instance for instance in series.instances if instance.sop_instance_uid), None)
        if first_instance is not None and first_instance.sop_instance_uid:
            cached = compat.dicom_cache.get(first_instance.sop_instance_uid, first_instance.path)
            return (
                float(cached.window_width or self._derive_default_window_width(cached)),
                float(cached.window_center or self._derive_default_window_center(cached)),
            )
        pixel_min = float(np.min(volume))
        pixel_max = float(np.max(volume))
        return (max(WINDOW_WIDTH_MIN, pixel_max - pixel_min), (pixel_min + pixel_max) / 2.0)

    @staticmethod
    def _get_geometry_axis_spacing(geometry: VolumeGeometry, axis_index: int) -> float:
        axis = np.asarray(geometry.ijk_to_world[:3, axis_index], dtype=np.float64)
        spacing = float(np.linalg.norm(axis))
        if not np.isfinite(spacing) or spacing <= 0.0:
            return 1.0
        return max(spacing, 1e-6)

    def _get_fusion_source_shape_and_spacing(
        self,
        view: ViewRecord,
        *,
        ct_volume: np.ndarray,
        ct_geometry: VolumeGeometry,
        pet_volume: np.ndarray,
        pet_geometry: VolumeGeometry,
    ) -> tuple[int, int, tuple[float, float]]:
        role = self._resolve_fusion_pane_role(view)
        if role == FUSION_PANE_PET_CORONAL_MIP:
            image_height = int(pet_volume.shape[0])
            image_width = int(pet_volume.shape[2])
            spacing_x = self._get_geometry_axis_spacing(pet_geometry, 2)
            spacing_y = self._get_geometry_axis_spacing(pet_geometry, 0)
            return image_height, image_width, (spacing_x, spacing_y)

        group = view.view_group
        axial_index = group.fusion_axial_index if group is not None else int(ct_volume.shape[0]) // 2
        plane = build_ct_axial_plane(ct_geometry, ct_volume.shape, axial_index)
        return (
            int(plane.output_shape[0]),
            int(plane.output_shape[1]),
            (float(plane.pixel_spacing_col_mm), float(plane.pixel_spacing_row_mm)),
        )

    @staticmethod
    def _calculate_fusion_physical_contain_zoom(
        view: ViewRecord,
        *,
        width_mm: float,
        height_mm: float,
        canvas_width: int | None = None,
        canvas_height: int | None = None,
    ) -> float:
        width = max(float(width_mm), 1e-6)
        height = max(float(height_mm), 1e-6)
        return compat.viewport_transformer.calculate_contain_zoom(
            image_width=1,
            image_height=1,
            canvas_width=canvas_width or view.width or 1,
            canvas_height=canvas_height or view.height or 1,
            pixel_aspect_x=width,
            pixel_aspect_y=height,
        )

    def _build_fusion_axial_display_plane_for_view(
        self,
        view: ViewRecord,
        *,
        ct_volume: np.ndarray,
        ct_geometry: VolumeGeometry,
        pet_volume: np.ndarray,
        pet_geometry: VolumeGeometry,
    ) -> PlanePose:
        group = view.view_group
        axial_index = group.fusion_axial_index if group is not None else int(ct_volume.shape[0]) // 2
        registration = group.fusion_registration if group is not None else FusionRegistrationState()
        return build_fusion_axial_display_plane(
            ct_geometry=ct_geometry,
            ct_shape=tuple(int(value) for value in ct_volume.shape),
            pet_geometry=pet_geometry,
            pet_shape=tuple(int(value) for value in pet_volume.shape),
            axial_index=axial_index,
            registration=registration,
        )

    def _calculate_fusion_axial_shared_fit_zoom(
        self,
        view: ViewRecord,
        *,
        ct_volume: np.ndarray,
        ct_geometry: VolumeGeometry,
        pet_volume: np.ndarray,
        pet_geometry: VolumeGeometry,
        canvas_width: int | None = None,
        canvas_height: int | None = None,
    ) -> float:
        plane = self._build_fusion_axial_display_plane_for_view(
            view,
            ct_volume=ct_volume,
            ct_geometry=ct_geometry,
            pet_volume=pet_volume,
            pet_geometry=pet_geometry,
        )
        return self._calculate_fusion_physical_contain_zoom(
            view,
            width_mm=max(float(plane.output_shape[1]) * float(plane.pixel_spacing_col_mm), 1e-6),
            height_mm=max(float(plane.output_shape[0]) * float(plane.pixel_spacing_row_mm), 1e-6),
            canvas_width=canvas_width,
            canvas_height=canvas_height,
        )

    def _calculate_fusion_fit_zoom_for_size(
        self,
        view: ViewRecord,
        *,
        ct_volume: np.ndarray,
        ct_geometry: VolumeGeometry,
        pet_volume: np.ndarray,
        pet_geometry: VolumeGeometry,
        canvas_width: int | None = None,
        canvas_height: int | None = None,
    ) -> float:
        if self._resolve_fusion_pane_role(view) != FUSION_PANE_PET_CORONAL_MIP:
            return self._calculate_fusion_axial_shared_fit_zoom(
                view,
                ct_volume=ct_volume,
                ct_geometry=ct_geometry,
                pet_volume=pet_volume,
                pet_geometry=pet_geometry,
                canvas_width=canvas_width,
                canvas_height=canvas_height,
            )

        image_height, image_width, spacing_xy = self._get_fusion_source_shape_and_spacing(
            view,
            ct_volume=ct_volume,
            ct_geometry=ct_geometry,
            pet_volume=pet_volume,
            pet_geometry=pet_geometry,
        )
        pixel_aspect_x, pixel_aspect_y = self._get_display_aspect_xy_from_spacing(spacing_xy)
        return compat.viewport_transformer.calculate_contain_zoom(
            image_width=image_width,
            image_height=image_height,
            canvas_width=canvas_width or view.width or image_width,
            canvas_height=canvas_height or view.height or image_height,
            pixel_aspect_x=pixel_aspect_x,
            pixel_aspect_y=pixel_aspect_y,
        )

    def _is_fusion_view_at_auto_fit_size(
        self,
        view: ViewRecord,
        *,
        canvas_width: int | None,
        canvas_height: int | None,
    ) -> bool:
        if not canvas_width or not canvas_height:
            return False
        if (
            abs(float(view.offset_x)) > 1e-6
            or abs(float(view.offset_y)) > 1e-6
            or int(view.rotation_degrees) != 0
            or bool(view.hor_flip)
            or bool(view.ver_flip)
        ):
            return False
        _group, ct_series, pet_series = self._resolve_fusion_group_series(view)
        ct_volume = self._get_series_volume(ct_series)
        pet_volume = self._get_series_volume(pet_series)
        ct_geometry = self._get_series_volume_geometry(ct_series, ct_volume.shape)
        pet_geometry = self._get_series_volume_geometry(pet_series, pet_volume.shape)
        expected_zoom = self._calculate_fusion_fit_zoom_for_size(
            view,
            ct_volume=ct_volume,
            ct_geometry=ct_geometry,
            pet_volume=pet_volume,
            pet_geometry=pet_geometry,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
        )
        tolerance = max(1e-3, abs(float(expected_zoom)) * 1e-3)
        return abs(float(view.zoom) - float(expected_zoom)) <= tolerance

    def _fit_initialized_fusion_view_to_source(self, view: ViewRecord) -> None:
        _group, ct_series, pet_series = self._resolve_fusion_group_series(view)
        ct_volume = self._get_series_volume(ct_series)
        pet_volume = self._get_series_volume(pet_series)
        ct_geometry = self._get_series_volume_geometry(ct_series, ct_volume.shape)
        pet_geometry = self._get_series_volume_geometry(pet_series, pet_volume.shape)
        self._fit_fusion_view_to_source(
            view,
            ct_volume=ct_volume,
            ct_geometry=ct_geometry,
            pet_volume=pet_volume,
            pet_geometry=pet_geometry,
        )
        self._sync_fusion_view_state_from_group(view)

    def _fit_fusion_view_to_source(
        self,
        view: ViewRecord,
        *,
        ct_volume: np.ndarray,
        ct_geometry: VolumeGeometry,
        pet_volume: np.ndarray,
        pet_geometry: VolumeGeometry,
    ) -> None:
        if self._resolve_fusion_pane_role(view) != FUSION_PANE_PET_CORONAL_MIP:
            view.zoom = self._calculate_fusion_axial_shared_fit_zoom(
                view,
                ct_volume=ct_volume,
                ct_geometry=ct_geometry,
                pet_volume=pet_volume,
                pet_geometry=pet_geometry,
            )
            view.offset_x = 0.0
            view.offset_y = 0.0
            view.rotation_degrees = 0
            view.hor_flip = False
            view.ver_flip = False
            self._reset_drag_state(view)
            return

        image_height, image_width, spacing_xy = self._get_fusion_source_shape_and_spacing(
            view,
            ct_volume=ct_volume,
            ct_geometry=ct_geometry,
            pet_volume=pet_volume,
            pet_geometry=pet_geometry,
        )
        pixel_aspect_x, pixel_aspect_y = self._get_display_aspect_xy_from_spacing(spacing_xy)
        view.zoom = compat.viewport_transformer.calculate_contain_zoom(
            image_width=image_width,
            image_height=image_height,
            canvas_width=view.width or image_width,
            canvas_height=view.height or image_height,
            pixel_aspect_x=pixel_aspect_x,
            pixel_aspect_y=pixel_aspect_y,
        )
        view.offset_x = 0.0
        view.offset_y = 0.0
        view.rotation_degrees = 0
        view.hor_flip = False
        view.ver_flip = False
        self._reset_drag_state(view)

    def _initialize_fusion_viewport(self, view: ViewRecord) -> None:
        ensure_view_size(view)
        group, ct_series, pet_series = self._resolve_fusion_group_series(view)
        ct_volume = self._get_series_volume(ct_series)
        pet_volume = self._get_series_volume(pet_series)
        ct_geometry = self._get_series_volume_geometry(ct_series, ct_volume.shape)
        pet_geometry = self._get_series_volume_geometry(pet_series, pet_volume.shape)
        if not group.fusion_initialized:
            group.fusion_axial_index = ct_volume.shape[0] // 2
            ct_ww, ct_wl = self._derive_default_window_for_volume(ct_series, ct_volume)
            group.fusion_pet_unit = self._normalize_fusion_pet_unit(group.fusion_pet_unit)
            pet_display = self._build_fusion_pet_display_volume(pet_series, pet_volume, group.fusion_pet_unit)
            group.fusion_pet_unit = pet_display.unit
            pet_ww, pet_wl = self._derive_default_pet_window_for_display_volume(pet_display)
            group.window.window_width = ct_ww
            group.window.window_center = ct_wl
            group.fusion_pet_window.window_width = pet_ww
            group.fusion_pet_window.window_center = pet_wl
            group.fusion_pet_pseudocolor_preset = normalize_pseudocolor_preset(group.fusion_pet_pseudocolor_preset or "petct-rainbow")
            group.fusion_initialized = True
        self._fit_fusion_view_to_source(
            view,
            ct_volume=ct_volume,
            ct_geometry=ct_geometry,
            pet_volume=pet_volume,
            pet_geometry=pet_geometry,
        )
        self._sync_fusion_view_state_from_group(view)
        view.is_initialized = True

    def _sync_fusion_view_state_from_group(self, view: ViewRecord) -> None:
        group = view.view_group
        if group is None:
            return
        role = self._resolve_fusion_pane_role(view)
        view.current_index = int(group.fusion_axial_index)
        if role in {FUSION_PANE_PET_AXIAL, FUSION_PANE_PET_CORONAL_MIP}:
            view.window_width = group.fusion_pet_window.window_width
            view.window_center = group.fusion_pet_window.window_center
            view.pseudocolor_preset = FUSION_PET_STANDALONE_PSEUDOCOLOR_PRESET
        elif role == FUSION_PANE_OVERLAY_AXIAL:
            view.window_width = group.window.window_width
            view.window_center = group.window.window_center
            view.pseudocolor_preset = group.fusion_pet_pseudocolor_preset
        else:
            view.window_width = group.window.window_width
            view.window_center = group.window.window_center
            view.pseudocolor_preset = DEFAULT_PSEUDOCOLOR_PRESET

    def _reset_fusion_view_group(self, view: ViewRecord) -> None:
        group = view.view_group
        if group is None:
            self._initialize_fusion_viewport(view)
            return
        self._clear_fusion_registration_overlay_frame_locks(group)
        self._fusion_registration_preview_drags.pop(group.group_id, None)
        group.fusion_initialized = False
        group.fusion_pet_pseudocolor_preset = "petct-rainbow"
        group.fusion_pet_unit = FUSION_PET_UNIT_SUV_BW
        group.fusion_registration = FusionRegistrationState()
        group.fusion_revision += 1
        for group_view in self._get_group_views(view):
            group_view.offset_x = 0.0
            group_view.offset_y = 0.0
            group_view.zoom = 1.0
            group_view.rotation_degrees = 0
            group_view.hor_flip = False
            group_view.ver_flip = False
            self._reset_drag_state(group_view)
            self._initialize_fusion_viewport(group_view)

    def _reset_view(self, view: ViewRecord) -> None:
        if self._is_mpr_view_type(view.view_type):
            self._reset_mpr_view_group(view)
        elif self._is_3d_view_type(view.view_type):
            view.rotation_degrees = 0
            view.hor_flip = False
            view.ver_flip = False
            self._initialize_3d_viewport(view)
        elif self._is_fusion_view_type(view.view_type):
            self._reset_fusion_view_group(view)
        elif self._is_pet_view_type(view.view_type):
            view.pet_unit = FUSION_PET_UNIT_SUV_BW
            view.pet_unit_label = FUSION_PET_UNIT_LABELS[FUSION_PET_UNIT_SUV_BW]
            self._initialize_pet_viewport(view)
        else:
            view.rotation_degrees = 0
            view.hor_flip = False
            view.ver_flip = False
            self._initialize_viewport(view)

        view.is_initialized = True

    def _reset_mpr_view_group(self, view: ViewRecord) -> None:
        group_views = self._get_mpr_group_views(view)
        group = view.view_group
        if group is not None:
            series = compat.series_registry.get(view.series_id)
            volume = self._get_series_volume(series)
            self._reset_mpr_group_geometry(group, volume.shape, series=series)
        else:
            series = None
            volume = None

        for group_view in group_views:
            if group is not None and series is not None and volume is not None:
                self._reset_mpr_view_display_state(group_view)
                self._reset_mpr_view_window(group_view, series, volume)
                self._fit_mpr_view_to_plane(group_view, series, volume)
            else:
                self._initialize_mpr_viewport(group_view)
            group_view.is_initialized = True

    def _reset_mpr_crosshair_state(self, view: ViewRecord) -> bool:
        group = view.view_group
        if group is None:
            return False
        series = compat.series_registry.get(view.series_id)
        volume = self._get_series_volume(series)
        volume_shape = volume.shape
        default_frame = self._build_default_mpr_frame_state(volume_shape)
        geometry = self._get_series_volume_geometry(series, volume_shape)
        default_cursor = legacy_frame_to_cursor(default_frame, geometry, reference_center=default_frame.center)

        group.active_viewport = MPR_VIEWPORT_AXIAL
        group.crosshair_drag_active = False
        group.crosshair_drag_origin_center = None
        group.crosshair_drag_origin_image = None
        group.rotation_drag = None
        group.mpr_crosshair_angles.clear()
        group.mpr_crosshair_mode = MPR_CROSSHAIR_MODE_ORTHOGONAL
        group.mpr_independent_plane_normals.clear()
        group.mpr_use_display_basis_for_cursor_offsets = False
        self._sync_group_from_mpr_cursor(group, default_cursor, geometry, volume_shape)
        self._reset_mpr_rotation_state(group)

        for group_view in self._get_mpr_group_views(view):
            self._reset_drag_state(group_view)
            group_view.is_initialized = True
        return True

    def _reset_rotate_3d_state(self, view: ViewRecord) -> bool:
        if self._is_mpr_view_type(view.view_type):
            group = view.view_group
            if group is None:
                return False
            self._set_mpr_model_rotation_matrix(group, np.eye(3, dtype=np.float64))
            group.mpr_model_rotation_pivot_world = None
            group.rotation_drag = None
            for group_view in self._get_mpr_group_views(view):
                self._reset_drag_state(group_view)
                group_view.is_initialized = True
            return True

        if not self._is_3d_view_type(view.view_type):
            return False
        view.rotation_quaternion = compat._get_vtk_volume_renderer().get_default_rotation_quaternion()
        self._reset_drag_state(view)
        view.is_initialized = True
        return True

    def _reset_mpr_group_geometry(
        self,
        group: ViewGroupRecord,
        volume_shape: tuple[int, int, int],
        *,
        series: SeriesRecord | None = None,
    ) -> None:
        group.active_viewport = MPR_VIEWPORT_AXIAL
        group.crosshair_drag_active = False
        group.crosshair_drag_origin_center = None
        group.crosshair_drag_origin_image = None
        group.rotation_drag = None
        group.mpr_crosshair_angles.clear()
        group.mpr_crosshair_mode = MPR_CROSSHAIR_MODE_ORTHOGONAL
        group.mpr_independent_plane_normals.clear()
        group.mpr_mip = self._create_default_mpr_mip_state()
        group.mpr_segmentation = self._create_default_mpr_segmentation_state()
        group.mpr_use_display_basis_for_cursor_offsets = False
        self._set_mpr_model_rotation_matrix(group, np.eye(3, dtype=np.float64))
        group.mpr_model_rotation_pivot_world = None
        default_frame = self._build_default_mpr_frame_state(volume_shape)
        geometry = self._get_series_volume_geometry(series, volume_shape) if series is not None else build_identity_geometry(volume_shape)
        default_cursor = legacy_frame_to_cursor(default_frame, geometry, reference_center=default_frame.center)
        self._sync_group_from_mpr_cursor(group, default_cursor, geometry, volume_shape)
        self._reset_mpr_rotation_state(group)

    def _reset_mpr_view_display_state(self, view: ViewRecord) -> None:
        view.current_index = view.mpr_axial_index
        view.offset_x = 0.0
        view.offset_y = 0.0
        view.zoom = 1.0
        view.rotation_degrees = 0
        view.hor_flip = False
        view.ver_flip = False
        view.pseudocolor_preset = DEFAULT_PSEUDOCOLOR_PRESET
        self._reset_drag_state(view)

    def _reset_mpr_view_window(self, view: ViewRecord, series: SeriesRecord, volume: np.ndarray) -> None:
        first_instance = next((instance for instance in series.instances if instance.sop_instance_uid), None)
        if first_instance is not None and first_instance.sop_instance_uid:
            cached = compat.dicom_cache.get(first_instance.sop_instance_uid, first_instance.path)
            view.window_width = cached.window_width or self._derive_default_window_width(cached)
            view.window_center = cached.window_center or self._derive_default_window_center(cached)
            return
        pixel_min = float(np.min(volume))
        pixel_max = float(np.max(volume))
        view.window_width = max(WINDOW_WIDTH_MIN, pixel_max - pixel_min)
        view.window_center = (pixel_max + pixel_min) / 2.0

    def _fit_mpr_view_to_plane(self, view: ViewRecord, series: SeriesRecord, volume: np.ndarray) -> None:
        plane_pixels, _, _ = self._extract_mpr_plane(view, volume)
        target_viewport = self._resolve_mpr_viewport(view)
        pose_context = self._build_mpr_pose_context(view, volume.shape, series=series)
        pixel_aspect_x, pixel_aspect_y = self._get_mpr_display_aspect_xy_from_pose(
            pose_context.poses[target_viewport]
        )
        view.zoom = compat.viewport_transformer.calculate_contain_zoom(
            image_width=plane_pixels.shape[1],
            image_height=plane_pixels.shape[0],
            canvas_width=view.width or plane_pixels.shape[1],
            canvas_height=view.height or plane_pixels.shape[0],
            pixel_aspect_x=pixel_aspect_x,
            pixel_aspect_y=pixel_aspect_y,
        )
