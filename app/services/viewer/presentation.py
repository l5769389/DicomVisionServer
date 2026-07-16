from __future__ import annotations

"""DICOM export, corner overlays, orientation, and formatting."""

from app.services.viewer.shared import *  # noqa: F403


class ViewerPresentationMixin:
    def _get_export_reference_dataset(self, view: ViewRecord) -> Dataset | None:
        series = compat.series_registry.get(view.series_id)
        if self._is_mpr_view_type(view.view_type) or self._is_3d_view_type(view.view_type):
            _, cached = self._get_reference_instance_and_cache(series)
            return cached.dataset if cached is not None else None

        if 0 <= view.current_index < len(series.instances):
            instance = series.instances[view.current_index]
            if instance.sop_instance_uid:
                cached = compat.dicom_cache.get(instance.sop_instance_uid, instance.path)
                return cached.dataset

        _, cached = self._get_reference_instance_and_cache(series)
        return cached.dataset if cached is not None else None

    @staticmethod
    def _build_secondary_capture_dicom_bytes(view: ViewRecord, image: Image.Image, reference_dataset: Dataset | None) -> bytes:
        now = datetime.now()
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        file_meta.ImplementationClassUID = PYDICOM_IMPLEMENTATION_UID

        dataset = Dataset()
        dataset.file_meta = file_meta
        dataset.is_little_endian = True
        dataset.is_implicit_VR = False

        if reference_dataset is not None:
            for attribute in (
                "PatientName",
                "PatientID",
                "PatientBirthDate",
                "PatientSex",
                "StudyInstanceUID",
                "StudyID",
                "AccessionNumber",
                "StudyDate",
                "StudyTime",
                "ReferringPhysicianName",
                "InstitutionName",
                "Manufacturer",
            ):
                value = getattr(reference_dataset, attribute, None)
                if value not in (None, ""):
                    setattr(dataset, attribute, value)

        dataset.SOPClassUID = SecondaryCaptureImageStorage
        dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        dataset.SeriesInstanceUID = generate_uid()
        dataset.Modality = "OT"
        dataset.SeriesNumber = 999
        dataset.InstanceNumber = 1
        dataset.ImageType = ["DERIVED", "SECONDARY", "OTHER"]
        dataset.ConversionType = "WSD"
        dataset.SeriesDescription = f"Exported {view.view_type}"
        dataset.ContentDate = now.strftime("%Y%m%d")
        dataset.ContentTime = now.strftime("%H%M%S")
        dataset.InstanceCreationDate = dataset.ContentDate
        dataset.InstanceCreationTime = dataset.ContentTime
        dataset.BurnedInAnnotation = "YES"
        dataset.SpecificCharacterSet = "ISO_IR 192"

        rgb_image = image.convert("RGB")
        rows, cols = rgb_image.height, rgb_image.width
        dataset.SamplesPerPixel = 3
        dataset.PhotometricInterpretation = "RGB"
        dataset.PlanarConfiguration = 0
        dataset.Rows = rows
        dataset.Columns = cols
        dataset.BitsAllocated = 8
        dataset.BitsStored = 8
        dataset.HighBit = 7
        dataset.PixelRepresentation = 0
        dataset.PixelData = rgb_image.tobytes()

        output = io.BytesIO()
        dcmwrite(output, dataset, write_like_original=False)
        return output.getvalue()

    @staticmethod
    def _build_mpr_crosshair_info(overlay: MprCrosshairOverlay) -> MprCrosshairInfo | None:
        if overlay.center_x is None or overlay.center_y is None:
            return None

        canvas_width = float(overlay.width)
        canvas_height = float(overlay.height)
        min_canvas_dimension = min(canvas_width, canvas_height)
        normalized_radius = (
            CROSSHAIR_HIT_RADIUS / min_canvas_dimension
            if min_canvas_dimension > 0
            else 0.0
        )
        return MprCrosshairInfo(
            centerX=(
                float(overlay.center_x) / canvas_width
                if canvas_width > 0
                else 0.0
            ),
            centerY=(
                float(overlay.center_y) / canvas_height
                if canvas_height > 0
                else 0.0
            ),
            hitRadius=normalized_radius,
            horizontalPosition=(
                float(overlay.horizontal_position) / canvas_height
                if overlay.horizontal_position is not None and canvas_height > 0
                else None
            ),
            verticalPosition=(
                float(overlay.vertical_position) / canvas_width
                if overlay.vertical_position is not None and canvas_width > 0
                else None
            ),
            horizontalAngleRad=float(overlay.horizontal_angle_rad),
            verticalAngleRad=float(overlay.vertical_angle_rad),
            horizontalSlabOffsetX=(
                float(overlay.horizontal_slab_offset_x) / canvas_width
                if overlay.horizontal_slab_offset_x is not None and canvas_width > 0
                else None
            ),
            horizontalSlabOffsetY=(
                float(overlay.horizontal_slab_offset_y) / canvas_height
                if overlay.horizontal_slab_offset_y is not None and canvas_height > 0
                else None
            ),
            verticalSlabOffsetX=(
                float(overlay.vertical_slab_offset_x) / canvas_width
                if overlay.vertical_slab_offset_x is not None and canvas_width > 0
                else None
            ),
            verticalSlabOffsetY=(
                float(overlay.vertical_slab_offset_y) / canvas_height
                if overlay.vertical_slab_offset_y is not None and canvas_height > 0
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
    def _resolve_mpr_slab_offset_canvas(
        plane_pose,
        target_pose,
        thickness_mm: float,
        center_image_x: float,
        center_image_y: float,
        center_canvas_x: float,
        center_canvas_y: float,
        image_to_canvas,
    ) -> tuple[float | None, float | None]:
        active_normal = np.asarray(plane_pose.normal_world, dtype=np.float64)
        target_normal = np.asarray(target_pose.normal_world, dtype=np.float64)
        projected_normal = target_normal - float(np.dot(target_normal, active_normal)) * active_normal
        projected_norm = float(np.linalg.norm(projected_normal))
        if not np.isfinite(projected_norm) or projected_norm <= 1e-6:
            return None, None

        offset_world = projected_normal / projected_norm * (float(thickness_mm) / 2.0 / projected_norm)
        offset_image_x = float(np.dot(offset_world, plane_pose.col_world)) / max(float(plane_pose.pixel_spacing_col_mm), 1e-6)
        offset_image_y = float(np.dot(offset_world, plane_pose.row_world)) / max(float(plane_pose.pixel_spacing_row_mm), 1e-6)
        offset_canvas_x, offset_canvas_y = image_to_canvas(
            center_image_x + offset_image_x,
            center_image_y + offset_image_y,
        )
        return float(offset_canvas_x - center_canvas_x), float(offset_canvas_y - center_canvas_y)

    def _build_mpr_crosshair_overlay(
        self,
        view: ViewRecord,
        volume_shape: tuple[int, int, int],
        plane_shape: tuple[int, int],
        image_transform,
        *,
        pose_context: MprPoseContext | None = None,
    ) -> MprCrosshairOverlay:
        plane_height, plane_width = plane_shape
        canvas_width = view.width or plane_width
        canvas_height = view.height or plane_height
        target_viewport = self._resolve_mpr_viewport(view)
        is_active = view.mpr_active_viewport == target_viewport
        line_alpha = 255
        if pose_context is None:
            try:
                series = compat.series_registry.get(view.series_id)
            except Exception:
                series = None
            pose_context = self._build_mpr_pose_context(view, volume_shape, series=series)
        plane_pose = pose_context.poses[target_viewport]
        horizontal_angle, vertical_angle = self._get_mpr_visible_crosshair_line_angles(
            view.view_group,
            pose_context.poses,
            target_viewport,
        )
        center_image_x, center_image_y = self._project_world_point_to_plane_image(plane_pose, pose_context.cursor.center_world)

        def with_alpha(rgb: tuple[int, int, int], alpha: int) -> tuple[int, int, int, int]:
            return rgb[0], rgb[1], rgb[2], alpha

        axial_color = with_alpha((34, 197, 94), line_alpha)
        coronal_color = with_alpha((59, 130, 246), line_alpha)
        sagittal_color = with_alpha((239, 68, 68), line_alpha)

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
        center_x, center_y = image_to_canvas(center_image_x, center_image_y)
        horizontal_position = None
        vertical_position = None
        if not plane_pose.is_oblique:
            _, horizontal_position = image_to_canvas(0.0, center_image_y)
            vertical_position, _ = image_to_canvas(center_image_x, 0.0)

        def slab_offset_for_line(line: str) -> tuple[float | None, float | None]:
            if not view.mpr_mip.enabled:
                return None, None
            target_line_viewport = self._resolve_mpr_oblique_target_viewport(target_viewport, line)
            viewport_mip = view.mpr_mip.viewports.get(target_line_viewport, MprMipViewportState())
            target_pose = pose_context.poses.get(target_line_viewport)
            if target_pose is None:
                return None, None
            configured_thickness_mm = float(viewport_mip.thickness)
            thickness_mm = (
                max(1e-6, float(spacing_along_world_direction(pose_context.geometry, target_pose.normal_world)))
                if configured_thickness_mm <= 0.0
                else configured_thickness_mm
            )
            return self._resolve_mpr_slab_offset_canvas(
                plane_pose,
                target_pose,
                thickness_mm,
                center_image_x,
                center_image_y,
                center_x,
                center_y,
                image_to_canvas,
            )

        horizontal_slab_offset_x, horizontal_slab_offset_y = slab_offset_for_line("horizontal")
        vertical_slab_offset_x, vertical_slab_offset_y = slab_offset_for_line("vertical")

        if target_viewport == MPR_VIEWPORT_CORONAL:
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
                horizontal_angle_rad=horizontal_angle,
                vertical_angle_rad=vertical_angle,
                horizontal_slab_offset_x=horizontal_slab_offset_x,
                horizontal_slab_offset_y=horizontal_slab_offset_y,
                vertical_slab_offset_x=vertical_slab_offset_x,
                vertical_slab_offset_y=vertical_slab_offset_y,
                center_x=center_x,
                center_y=center_y,
                is_active=is_active,
            )
        if target_viewport == MPR_VIEWPORT_SAGITTAL:
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
                horizontal_angle_rad=horizontal_angle,
                vertical_angle_rad=vertical_angle,
                horizontal_slab_offset_x=horizontal_slab_offset_x,
                horizontal_slab_offset_y=horizontal_slab_offset_y,
                vertical_slab_offset_x=vertical_slab_offset_x,
                vertical_slab_offset_y=vertical_slab_offset_y,
                center_x=center_x,
                center_y=center_y,
                is_active=is_active,
            )
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
            horizontal_angle_rad=horizontal_angle,
            vertical_angle_rad=vertical_angle,
            horizontal_slab_offset_x=horizontal_slab_offset_x,
            horizontal_slab_offset_y=horizontal_slab_offset_y,
            vertical_slab_offset_x=vertical_slab_offset_x,
            vertical_slab_offset_y=vertical_slab_offset_y,
            center_x=center_x,
            center_y=center_y,
            is_active=is_active,
        )

    def _get_mpr_crosshair_line_angles_from_poses(
        self,
        poses: dict[str, PlanePose],
        viewport_key: str,
    ) -> tuple[float, float]:
        active_pose = poses[viewport_key]

        def line_angle(line: str, fallback: float) -> float:
            target_viewport = self._resolve_mpr_oblique_target_viewport(viewport_key, line)
            target_pose = poses[target_viewport]
            line_world = self._normalize_oblique_vector(
                np.cross(active_pose.normal_world, target_pose.normal_world),
                fallback=tuple(active_pose.col_world if line == "horizontal" else active_pose.row_world),
            )
            col_component = float(np.dot(line_world, active_pose.col_world))
            row_component = float(np.dot(line_world, active_pose.row_world))
            magnitude = float(np.hypot(col_component, row_component))
            if not np.isfinite(magnitude) or magnitude <= 1e-8:
                return fallback
            return self._normalize_screen_half_turn_angle(float(np.arctan2(row_component, col_component)))

        return (
            line_angle("horizontal", 0.0),
            line_angle("vertical", float(np.pi / 2.0)),
        )

    @staticmethod
    def _get_reference_instance_and_cache(series: SeriesRecord) -> tuple[InstanceRecord | None, CachedDicom | None]:
        for instance in series.instances:
            if not instance.sop_instance_uid:
                continue
            return instance, compat.dicom_cache.get(instance.sop_instance_uid, instance.path)
        return None, None

    @staticmethod
    def _get_indexed_instance_and_cache(
        series: SeriesRecord,
        index: int,
    ) -> tuple[InstanceRecord | None, CachedDicom | None]:
        if not series.instances:
            return None, None
        clamped_index = max(0, min(int(index), len(series.instances) - 1))
        instance = series.instances[clamped_index]
        if not instance.sop_instance_uid:
            return compat.ViewerService._get_reference_instance_and_cache(series)
        return instance, compat.dicom_cache.get(instance.sop_instance_uid, instance.path)

    @staticmethod
    def _corner_info_tag(value: str | None) -> tuple[str, ...]:
        return (value,) if value else tuple()

    @staticmethod
    def _labeled_corner_info_tag(label: str, value: str | None) -> tuple[str, ...]:
        return (f"{label}: {value}",) if value else tuple()

    @staticmethod
    def _build_corner_info_tags(items: dict[str, tuple[str, ...]]) -> dict[str, tuple[str, ...]]:
        return {
            key: tuple(line for line in lines if line)
            for key, lines in items.items()
            if any(line for line in lines)
        }

    @staticmethod
    def _format_multi_number(value, *, precision: int = 2, separator: str = " x ", suffix: str = "") -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            text = value.strip()
            return text or None
        try:
            values = list(value)
        except TypeError:
            return compat.ViewerService._format_number(value, precision=precision, suffix=suffix)
        parts: list[str] = []
        for item in values:
            formatted = compat.ViewerService._format_number(item, precision=precision)
            if formatted is None:
                formatted = compat.ViewerService._safe_text(item)
            if formatted:
                parts.append(formatted)
        if not parts:
            return None
        return f"{separator.join(parts)}{suffix}"

    @staticmethod
    def _build_matrix_label(rows, columns) -> str | None:
        row_text = compat.ViewerService._safe_text(rows)
        column_text = compat.ViewerService._safe_text(columns)
        if not row_text or not column_text:
            return None
        return f"{row_text} x {column_text}"

    @staticmethod
    def _build_rescale_label(slope, intercept) -> str | None:
        slope_text = compat.ViewerService._format_number(slope, precision=4)
        intercept_text = compat.ViewerService._format_number(intercept, precision=4)
        if not slope_text and not intercept_text:
            return None
        return f"m {slope_text or '-'}  b {intercept_text or '-'}"

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
        series_description = self._first_non_empty(
            self._safe_text(getattr(dataset, "SeriesDescription", None)),
            self._safe_text(series.series_description),
        )
        exam_text = self._first_non_empty(
            study_description,
            self._safe_text(getattr(dataset, "StudyID", None)),
            series_description,
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
        modality = self._first_non_empty(self._safe_text(getattr(dataset, "Modality", None)), self._safe_text(series.modality))
        accession_number = self._first_non_empty(
            self._safe_text(getattr(dataset, "AccessionNumber", None)),
            self._safe_text(series.accession_number),
        )
        study_date = self._first_non_empty(
            self._format_dicom_date(getattr(dataset, "StudyDate", None)),
            self._format_dicom_date(series.study_date),
        )
        study_time = self._format_dicom_time(getattr(dataset, "StudyTime", None))
        study_id = self._safe_text(getattr(dataset, "StudyID", None))
        study_uid = self._first_non_empty(
            self._safe_text(getattr(dataset, "StudyInstanceUID", None)),
            self._safe_text(series.study_instance_uid),
        )
        series_uid = self._first_non_empty(
            self._safe_text(getattr(dataset, "SeriesInstanceUID", None)),
            self._safe_text(series.series_instance_uid),
        )
        sop_uid = self._safe_text(getattr(dataset, "SOPInstanceUID", None))
        body_part = self._safe_text(getattr(dataset, "BodyPartExamined", None))
        protocol_name = self._safe_text(getattr(dataset, "ProtocolName", None))
        patient_birth_date = self._format_dicom_date(getattr(dataset, "PatientBirthDate", None))
        referring_physician = self._safe_text(getattr(dataset, "ReferringPhysicianName", None))
        patient_position = self._safe_text(getattr(dataset, "PatientPosition", None))
        pixel_spacing = self._format_multi_number(getattr(dataset, "PixelSpacing", None), precision=3, suffix="mm")
        matrix = self._build_matrix_label(getattr(dataset, "Rows", None), getattr(dataset, "Columns", None))
        image_position = self._format_multi_number(getattr(dataset, "ImagePositionPatient", None), precision=2, separator=", ")
        image_orientation = self._format_multi_number(getattr(dataset, "ImageOrientationPatient", None), precision=4, separator=", ")
        rescale = self._build_rescale_label(getattr(dataset, "RescaleSlope", None), getattr(dataset, "RescaleIntercept", None))
        convolution_kernel = self._safe_text(getattr(dataset, "ConvolutionKernel", None))
        reconstruction_diameter = self._format_number(getattr(dataset, "ReconstructionDiameter", None), precision=1, suffix="mm")
        ctdi_vol = self._format_number(getattr(dataset, "CTDIvol", None), precision=2, suffix="mGy")
        exposure = self._format_number(getattr(dataset, "Exposure", None), precision=1, suffix="mAs")
        exposure_time = self._format_number(getattr(dataset, "ExposureTime", None), precision=1, suffix="ms")

        vendor_line = self._join_non_empty(" / ", manufacturer, manufacturer_model)
        patient_meta = self._join_non_empty(" ", patient_id, self._join_non_empty(" / ", patient_sex, patient_age))
        technique_parts = [part for part in (kv, ma) if part]
        acquisition_datetime = self._join_non_empty(" ", acquisition_date, acquisition_time)
        tags = self._build_corner_info_tags(
            {
                "manufacturerModel": self._corner_info_tag(vendor_line),
                "stationName": self._corner_info_tag(station_name),
                "institutionName": self._corner_info_tag(institution_name),
                "examDescription": self._corner_info_tag(exam_text),
                "seriesNumber": self._corner_info_tag(f"Se: {series_number}" if series_number else None),
                "patientName": self._corner_info_tag(patient_name),
                "patientSummary": self._corner_info_tag(patient_meta),
                "technique": self._corner_info_tag(" ".join(technique_parts) if technique_parts else None),
                "sliceThickness": self._corner_info_tag(thickness),
                "acquisitionDateTime": self._corner_info_tag(acquisition_datetime),
                "modality": self._labeled_corner_info_tag("Modality", modality),
                "accessionNumber": self._labeled_corner_info_tag("Acc", accession_number),
                "studyDate": self._labeled_corner_info_tag("Study date", study_date),
                "studyTime": self._labeled_corner_info_tag("Study time", study_time),
                "studyId": self._labeled_corner_info_tag("Study ID", study_id),
                "studyInstanceUid": self._labeled_corner_info_tag("Study UID", study_uid),
                "seriesInstanceUid": self._labeled_corner_info_tag("Series UID", series_uid),
                "sopInstanceUid": self._labeled_corner_info_tag("SOP UID", sop_uid),
                "seriesDescription": self._labeled_corner_info_tag("Series", series_description),
                "bodyPartExamined": self._labeled_corner_info_tag("Body", body_part),
                "protocolName": self._labeled_corner_info_tag("Protocol", protocol_name),
                "patientBirthDate": self._labeled_corner_info_tag("Birth", patient_birth_date),
                "referringPhysicianName": self._labeled_corner_info_tag("Referrer", referring_physician),
                "patientPosition": self._labeled_corner_info_tag("Patient pos", patient_position),
                "pixelSpacing": self._labeled_corner_info_tag("Pixel", pixel_spacing),
                "rowsColumns": self._labeled_corner_info_tag("Matrix", matrix),
                "imagePositionPatient": self._labeled_corner_info_tag("IPP", image_position),
                "imageOrientationPatient": self._labeled_corner_info_tag("IOP", image_orientation),
                "rescaleSlopeIntercept": self._labeled_corner_info_tag("Rescale", rescale),
                "convolutionKernel": self._labeled_corner_info_tag("Kernel", convolution_kernel),
                "reconstructionDiameter": self._labeled_corner_info_tag("FOV", reconstruction_diameter),
                "ctdiVol": self._labeled_corner_info_tag("CTDIvol", ctdi_vol),
                "exposure": self._labeled_corner_info_tag("Exposure", exposure),
                "exposureTime": self._labeled_corner_info_tag("Exp time", exposure_time),
            }
        )

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
                acquisition_datetime,
            )
            if line
        )
        return CornerInfoOverlay(
            top_left=top_left,
            top_right=top_right,
            bottom_left=bottom_left,
            bottom_right=tuple(),
            tags=tags,
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
        plane_state: MprObliquePlaneState | None = None,
        plane_pose: PlanePose | None = None,
        cursor: MprCursorState | None = None,
        show_physical_location: bool = True,
        show_image_index: bool = True,
    ) -> CornerInfoOverlay:
        zoom = self._format_number(view.zoom, precision=2, suffix="x")
        physical_location = (
            self._build_physical_location_label(
                view,
                series,
                dataset,
                current_index,
                viewport_label,
                plane_state=plane_state,
                plane_pose=plane_pose,
                cursor=cursor,
            )
            if show_physical_location
            else None
        )
        viewport_location = self._join_non_empty("  ", viewport_label, physical_location)
        image_index = f"Im: {current_index + 1}/{total_slices}" if show_image_index and total_slices > 0 else None
        window_level = self._build_window_label(view.window_width, view.window_center)
        slice_location = self._format_number(getattr(dataset, "SliceLocation", None), precision=2, suffix="mm")
        instance_number = self._safe_text(getattr(dataset, "InstanceNumber", None))
        sop_uid = self._safe_text(getattr(dataset, "SOPInstanceUID", None))
        image_position = self._format_multi_number(getattr(dataset, "ImagePositionPatient", None), precision=2, separator=", ")
        image_orientation = self._format_multi_number(getattr(dataset, "ImageOrientationPatient", None), precision=4, separator=", ")
        pixel_spacing = self._format_multi_number(getattr(dataset, "PixelSpacing", None), precision=3, suffix="mm")
        matrix = self._build_matrix_label(getattr(dataset, "Rows", None), getattr(dataset, "Columns", None))
        tags = self._build_corner_info_tags(
            {
                "viewportLocation": self._corner_info_tag(viewport_location),
                "imageIndex": self._corner_info_tag(image_index),
                "windowLevel": self._corner_info_tag(window_level),
                "zoom": self._corner_info_tag(f"Zoom:{zoom}" if zoom else None),
                "sliceLocation": self._labeled_corner_info_tag("Slice loc", slice_location) if show_image_index else tuple(),
                "instanceNumber": self._labeled_corner_info_tag("Instance", instance_number) if show_image_index else tuple(),
                "sopInstanceUid": self._labeled_corner_info_tag("SOP UID", sop_uid) if show_image_index else tuple(),
                "imagePositionPatient": self._labeled_corner_info_tag("IPP", image_position) if show_image_index else tuple(),
                "imageOrientationPatient": self._labeled_corner_info_tag("IOP", image_orientation) if show_image_index else tuple(),
                "pixelSpacing": self._labeled_corner_info_tag("Pixel", pixel_spacing),
                "rowsColumns": self._labeled_corner_info_tag("Matrix", matrix),
            }
        )
        top_left = tuple(
            line
            for line in (
                viewport_location,
                image_index,
            )
            if line
        )
        top_right = tuple()
        bottom_left = tuple(
            line
            for line in (
                window_level,
            )
            if line
        )
        bottom_right = tuple(
            line
            for line in (
                f"Zoom:{zoom}" if zoom else None,
            )
            if line
        )
        return CornerInfoOverlay(
            top_left=top_left,
            top_right=top_right,
            bottom_left=bottom_left,
            bottom_right=bottom_right,
            tags=tags,
        )

    @staticmethod
    def _serialize_corner_info_overlay(overlay: CornerInfoOverlay) -> CornerInfoPayload:
        return CornerInfoPayload(
            topLeft=list(overlay.top_left),
            topRight=list(overlay.top_right),
            bottomLeft=list(overlay.bottom_left),
            bottomRight=list(overlay.bottom_right),
            tags={key: list(lines) for key, lines in overlay.tags.items() if lines},
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
        *,
        plane_state: MprObliquePlaneState | None = None,
        plane_pose: PlanePose | None = None,
        cursor: MprCursorState | None = None,
    ) -> str | None:
        label = viewport_label.lower()
        if label.startswith("oblique "):
            label = label.removeprefix("oblique ").strip()
        if self._is_mpr_view_type(view.view_type):
            transform = self._get_series_patient_transform(series)
            if plane_pose is not None and cursor is not None and plane_pose.is_oblique:
                return self._format_mpr_plane_pose_physical_location(
                    cursor,
                    plane_pose,
                    transform,
                )
            if cursor is not None:
                try:
                    geometry = self._get_series_volume_geometry(series, self._get_series_volume(series).shape)
                    frame_center = world_to_ijk_point(geometry, cursor.center_world)
                except Exception:
                    frame_center = np.asarray(cursor.center_world, dtype=np.float64)
            else:
                frame_center = np.asarray(
                    [float(view.mpr_axial_index), float(view.mpr_coronal_index), float(view.mpr_sagittal_index)],
                    dtype=np.float64,
                )
            if transform is not None:
                patient_point = transform.clamped_point_to_patient(frame_center)
                return self._format_standard_physical_location(label, patient_point)

        position = self._get_dataset_position(dataset)
        if position is None:
            return None
        return self._format_standard_physical_location(label, position)

    def _format_standard_physical_location(self, label: str, patient_point: np.ndarray) -> str | None:
        if label.startswith("stack") or label.startswith("ax"):
            return self._format_oriented_mm(float(patient_point[2]), positive="I", negative="S")
        if label.startswith("cor"):
            return self._format_oriented_mm(float(patient_point[1]), positive="P", negative="A")
        if label.startswith("sag"):
            return self._format_oriented_mm(float(patient_point[0]), positive="L", negative="R")
        return self._join_non_empty(
            " ",
            self._format_oriented_mm(float(patient_point[0]), positive="L", negative="R"),
            self._format_oriented_mm(float(patient_point[1]), positive="P", negative="A"),
            self._format_oriented_mm(float(patient_point[2]), positive="S", negative="I"),
        )

    def _format_mpr_plane_pose_physical_location(
        self,
        cursor: MprCursorState,
        plane_pose: PlanePose,
        transform: VolumePatientTransform | None,
    ) -> str | None:
        delta_world = np.asarray(cursor.center_world, dtype=np.float64) - np.asarray(cursor.reference_center_world, dtype=np.float64)
        normal_world = np.asarray(plane_pose.normal_world, dtype=np.float64)
        if transform is not None:
            distance_vector = delta_world
            direction_vector = mpr_geometry.normalize_patient_vector(
                normal_world,
                fallback=np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
            )
        else:
            distance_vector = np.asarray([delta_world[2], delta_world[1], delta_world[0]], dtype=np.float64)
            direction_vector = mpr_geometry.fallback_volume_direction_to_patient_vector(normal_world)

        signed_distance = float(np.dot(distance_vector, direction_vector))
        if abs(signed_distance) < 0.005:
            signed_distance = 0.0
        label = self._dominant_orientation_text_for_vector(direction_vector if signed_distance >= 0.0 else -direction_vector)
        if not label:
            return None
        magnitude = self._format_number(abs(signed_distance), precision=2, suffix="mm") or "0mm"
        return f"{label} {magnitude}"

    def _get_mpr_reference_center(
        self,
        view: ViewRecord,
        series: SeriesRecord,
        fallback_center: np.ndarray,
    ) -> np.ndarray:
        group = view.view_group
        if group is not None and group.mpr_reference_center is not None:
            return np.asarray(group.mpr_reference_center, dtype=np.float64)
        if group is not None:
            try:
                reference_center = self._ensure_mpr_reference_center(group, self._get_series_volume(series).shape)
                return np.asarray(reference_center, dtype=np.float64)
            except Exception:
                pass
        return np.asarray(fallback_center, dtype=np.float64)

    def _get_series_patient_transform(self, series: SeriesRecord) -> VolumePatientTransform | None:
        cached_transform = self._series_patient_transform_cache.get(series.series_id, Ellipsis)
        if cached_transform is not Ellipsis:
            return cached_transform

        slice_entries: list[tuple[np.ndarray, np.ndarray | None, np.ndarray | None, Dataset]] = []
        for instance in series.instances:
            if not instance.sop_instance_uid:
                continue
            cached = compat.dicom_cache.get(instance.sop_instance_uid, instance.path)
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

        axis_mapping = get_standardized_axis_mapping(orientation)
        if axis_mapping is None:
            self._series_patient_transform_cache[series.series_id] = None
            return None

        positions = [item[2] for item in slice_entries]
        if any(position is None for position in positions):
            ordered_entries = slice_entries
        else:
            ordered_entries = sorted(
                slice_entries,
                key=lambda item: float(np.dot(item[2], axis_mapping.slice_direction)) if item[2] is not None else 0.0,
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
        slice_spacing = self._estimate_slice_spacing(ordered_positions, axis_mapping.slice_direction, first_dataset)

        raw_axis_vectors = (axis_mapping.slice_direction, axis_mapping.column_direction, axis_mapping.row_direction)
        raw_axis_steps = (slice_spacing, row_spacing, col_spacing)
        raw_lengths = (
            len(ordered_entries),
            int(getattr(first_dataset, "Rows", 0) or 0),
            int(getattr(first_dataset, "Columns", 0) or 0),
        )
        if any(length <= 0 for length in raw_lengths):
            self._series_patient_transform_cache[series.series_id] = None
            return None

        if ordered_entries[0][2] is None:
            self._series_patient_transform_cache[series.series_id] = None
            return None

        origin = np.asarray(ordered_entries[0][2], dtype=np.float64)
        for canonical_axis, raw_axis in enumerate(axis_mapping.transpose_order):
            if axis_mapping.canonical_signs[canonical_axis] < 0:
                origin = origin + raw_axis_vectors[raw_axis] * raw_axis_steps[raw_axis] * float(raw_lengths[raw_axis] - 1)

        axis_vectors = tuple(
            raw_axis_vectors[raw_axis] * raw_axis_steps[raw_axis] * float(axis_mapping.canonical_signs[canonical_axis])
            for canonical_axis, raw_axis in enumerate(axis_mapping.transpose_order)
        )
        shape = tuple(raw_lengths[raw_axis] for raw_axis in axis_mapping.transpose_order)
        result = VolumePatientTransform(origin=origin, axis_vectors=axis_vectors, shape=shape)
        self._series_patient_transform_cache[series.series_id] = result
        return result

    def _get_series_volume_geometry(self, series: SeriesRecord, volume_shape: tuple[int, int, int]) -> VolumeGeometry:
        cached_geometry = self._series_volume_geometry_cache.get(series.series_id)
        normalized_shape = tuple(int(value) for value in volume_shape)
        if cached_geometry is not None and cached_geometry.shape_ijk == normalized_shape:
            return cached_geometry

        transform = self._get_series_patient_transform(series)
        geometry = build_geometry_from_patient_transform(transform) if transform is not None else build_identity_geometry(normalized_shape)
        if geometry.shape_ijk != normalized_shape:
            geometry = build_identity_geometry(normalized_shape)
        self._series_volume_geometry_cache[series.series_id] = geometry
        return geometry

    @staticmethod
    def _build_fallback_mpr_frame(view: ViewRecord) -> MprFrameState:
        return MprFrameState(
            center=(
                float(view.mpr_axial_index),
                float(view.mpr_coronal_index),
                float(view.mpr_sagittal_index),
            ),
            axis_slice=(1.0, 0.0, 0.0),
            axis_row=(0.0, 1.0, 0.0),
            axis_col=(0.0, 0.0, 1.0),
        )

    def _get_mpr_cursor_state(
        self,
        view: ViewRecord,
        geometry: VolumeGeometry,
        volume_shape: tuple[int, int, int],
    ):
        if view.view_group is None:
            frame = self._build_fallback_mpr_frame(view)
            return legacy_frame_to_cursor(frame, geometry, reference_center=frame.center)

        group = view.view_group
        if group.mpr_cursor is not None:
            return self._deserialize_mpr_cursor_record(group.mpr_cursor)

        reference_center = self._ensure_mpr_reference_center(group, volume_shape)
        cursor = create_default_cursor(geometry)
        center_ijk = np.asarray(
            [
                float(max(0, min(group.axial_index, volume_shape[0] - 1))),
                float(max(0, min(group.coronal_index, volume_shape[1] - 1))),
                float(max(0, min(group.sagittal_index, volume_shape[2] - 1))),
            ],
            dtype=np.float64,
        )
        cursor = replace(
            cursor,
            center_world=ijk_to_world_point(geometry, center_ijk),
            reference_center_world=ijk_to_world_point(geometry, reference_center),
        )
        group.mpr_cursor = self._serialize_mpr_cursor_record(cursor)
        return cursor

    @staticmethod
    def _should_use_mpr_display_basis_for_cursor_offsets(group: ViewGroupRecord | None) -> bool:
        return bool(
            group is not None
            and (
                group.crosshair_drag_active
                or group.mpr_use_display_basis_for_cursor_offsets
            )
        )

    @staticmethod
    def _build_reslice_mip_config(mip_state: MprMipState, viewport_key: str) -> ResliceMipConfig:
        viewport_config = mip_state.viewports.get(viewport_key, MprMipViewportState())
        return ResliceMipConfig(
            enabled=bool(mip_state.enabled),
            algorithm=str(mip_state.algorithm or "maximum"),
            thickness=max(0, min(100, int(viewport_config.thickness))),
        )

    @staticmethod
    def _serialize_mpr_cursor_record(cursor: MprCursorState) -> MprCursorRecord:
        orientation = np.asarray(cursor.orientation_world, dtype=np.float64)
        return MprCursorRecord(
            center_world=tuple(float(value) for value in np.asarray(cursor.center_world, dtype=np.float64)),
            reference_center_world=tuple(float(value) for value in np.asarray(cursor.reference_center_world, dtype=np.float64)),
            orientation_world=tuple(
                tuple(float(value) for value in orientation[:, column_index])
                for column_index in range(orientation.shape[1])
            ),
            linked_to_volume_rotation=bool(cursor.linked_to_volume_rotation),
        )

    @staticmethod
    def _deserialize_mpr_cursor_record(record: MprCursorRecord) -> MprCursorState:
        orientation_columns = [
            np.asarray(column, dtype=np.float64)
            for column in record.orientation_world
        ]
        if len(orientation_columns) != 3:
            orientation_columns = [
                np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
                np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
                np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
            ]
        return MprCursorState(
            center_world=np.asarray(record.center_world, dtype=np.float64),
            reference_center_world=np.asarray(record.reference_center_world, dtype=np.float64),
            orientation_world=np.column_stack(orientation_columns),
            linked_to_volume_rotation=bool(record.linked_to_volume_rotation),
        )

    def _build_mpr_pose_context(
        self,
        view: ViewRecord,
        volume_shape: tuple[int, int, int],
        *,
        series: SeriesRecord | None = None,
    ) -> MprPoseContext:
        normalized_shape = tuple(int(value) for value in volume_shape)
        geometry = (
            self._get_series_volume_geometry(series, normalized_shape)
            if series is not None
            else build_identity_geometry(normalized_shape)
        )
        cursor = self._get_mpr_cursor_state(view, geometry, normalized_shape)
        return MprPoseContext(
            geometry=geometry,
            cursor=cursor,
            poses=self._build_mpr_plane_poses(
                cursor,
                geometry,
                normalized_shape,
                normal_overrides=self._get_independent_plane_normal_overrides(view.view_group),
                use_display_basis_for_cursor_offsets=self._should_use_mpr_display_basis_for_cursor_offsets(view.view_group),
            ),
        )

    @staticmethod
    def _project_world_point_to_plane_image(plane_pose: PlanePose, point_world: np.ndarray) -> tuple[float, float]:
        delta_world = np.asarray(point_world, dtype=np.float64) - np.asarray(plane_pose.center_world, dtype=np.float64)
        image_y = (
            float(np.dot(delta_world, plane_pose.row_world)) / max(float(plane_pose.pixel_spacing_row_mm), 1e-6)
            + float(plane_pose.output_shape[0]) / 2.0
        )
        image_x = (
            float(np.dot(delta_world, plane_pose.col_world)) / max(float(plane_pose.pixel_spacing_col_mm), 1e-6)
            + float(plane_pose.output_shape[1]) / 2.0
        )
        return image_x, image_y

    def _sync_group_from_mpr_cursor(
        self,
        group: ViewGroupRecord,
        cursor: MprCursorState,
        geometry: VolumeGeometry,
        volume_shape: tuple[int, int, int],
    ) -> None:
        group.mpr_reference_center = tuple(
            float(value)
            for value in world_to_ijk_point(geometry, cursor.reference_center_world)
        )
        group.mpr_cursor = self._serialize_mpr_cursor_record(cursor)
        center_ijk = world_to_ijk_point(geometry, cursor.center_world)
        group.axial_index = int(max(0, min(int(np.round(center_ijk[0])), volume_shape[0] - 1)))
        group.coronal_index = int(max(0, min(int(np.round(center_ijk[1])), volume_shape[1] - 1)))
        group.sagittal_index = int(max(0, min(int(np.round(center_ijk[2])), volume_shape[2] - 1)))

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

    def _format_projected_physical_location(
        self,
        patient_point: np.ndarray,
        patient_normal: np.ndarray,
        *,
        origin_point: np.ndarray | None = None,
        orientation_vector: np.ndarray | None = None,
    ) -> str | None:
        normal = mpr_geometry.normalize_patient_vector(patient_normal, fallback=np.asarray([0.0, 0.0, 1.0], dtype=np.float64))
        point = np.asarray(patient_point, dtype=np.float64)
        origin = np.zeros(3, dtype=np.float64) if origin_point is None else np.asarray(origin_point, dtype=np.float64)
        orientation_source = normal if orientation_vector is None else mpr_geometry.normalize_patient_vector(
            orientation_vector,
            fallback=normal,
        )
        if float(np.dot(normal, orientation_source)) < 0.0:
            normal = -normal
        distance = float(np.dot(point - origin, normal))
        if abs(distance) < 0.005:
            distance = 0.0
        orientation = self._dominant_orientation_text_for_vector(orientation_source if distance >= 0.0 else -orientation_source)
        if not orientation:
            return None
        magnitude = self._format_number(abs(distance), precision=2, suffix="mm") or "0mm"
        return f"{orientation} {magnitude}"

    @staticmethod
    def _resolve_mpr_directed_line_angle(current_row: np.ndarray, current_col: np.ndarray, line_dir: np.ndarray) -> float | None:
        col_component = float(np.dot(line_dir, current_col))
        row_component = float(np.dot(line_dir, current_row))
        if not np.isfinite(col_component) or not np.isfinite(row_component):
            return None
        magnitude = float(np.hypot(col_component, row_component))
        if magnitude <= 1e-8:
            return None
        angle = float(np.arctan2(row_component, col_component))
        if angle < 0.0:
            angle += float(np.pi * 2.0)
        return angle

    @staticmethod
    def _dominant_orientation_text_for_vector(vector: np.ndarray | None) -> str | None:
        return compat.ViewerService._orientation_text_for_vector(
            vector,
            minimum_magnitude=1e-4,
            max_components=1,
            axis_priority=(1, 0, 2),
        )

    @staticmethod
    def _mpr_oblique_orientation_text_for_vector(vector: np.ndarray | None) -> str | None:
        return compat.ViewerService._orientation_text_for_vector(
            vector,
            minimum_magnitude=0.2,
            max_components=2,
            axis_priority=(1, 0, 2),
        )

    @staticmethod
    def _orientation_text_for_vector(
        vector: np.ndarray | None,
        *,
        minimum_magnitude: float = 0.2,
        max_components: int = 3,
        axis_priority: tuple[int, int, int] = (0, 1, 2),
    ) -> str | None:
        if vector is None:
            return None
        axis_map = (
            (0, "L", "R", axis_priority[0]),
            (1, "P", "A", axis_priority[1]),
            (2, "S", "I", axis_priority[2]),
        )
        components: list[tuple[float, int, str]] = []
        for axis_index, positive_label, negative_label, priority in axis_map:
            component = float(vector[axis_index])
            magnitude = abs(component)
            if magnitude < minimum_magnitude:
                continue
            label = positive_label if component >= 0 else negative_label
            components.append((magnitude, priority, label))
        if not components:
            return None
        components.sort(key=lambda item: (-item[0], item[1]))
        return ''.join(label for _, _, label in components[:max(1, max_components)])

    @staticmethod
    def _rotate_screen_axes(
        x_vector: np.ndarray,
        y_vector: np.ndarray,
        rotation_degrees: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        normalized_rotation = compat.viewport_transformer.normalize_rotation_degrees(rotation_degrees)
        if normalized_rotation == 90:
            return y_vector, -x_vector
        if normalized_rotation == 180:
            return -x_vector, -y_vector
        if normalized_rotation == 270:
            return -y_vector, x_vector
        return x_vector, y_vector

    @staticmethod
    def _build_view_transform_payload(view: ViewRecord) -> ViewTransformPayload:
        return ViewTransformPayload(
            rotationDegrees=compat.viewport_transformer.normalize_rotation_degrees(view.rotation_degrees),
            horFlip=bool(view.hor_flip),
            verFlip=bool(view.ver_flip),
            zoom=float(compat.viewport_transformer.clamp_zoom(view.zoom)),
            offsetX=float(view.offset_x),
            offsetY=float(view.offset_y),
        )

    @staticmethod
    def _build_mpr_frame_payload(cursor: MprCursorState | None, geometry: VolumeGeometry | None) -> MprFrameInfo | None:
        if cursor is None or geometry is None:
            return None
        frame = cursor_to_legacy_frame(cursor, geometry)
        return MprFrameInfo(
            center=tuple(float(value) for value in frame.center),
            axisSlice=tuple(float(value) for value in frame.axis_slice),
            axisRow=tuple(float(value) for value in frame.axis_row),
            axisCol=tuple(float(value) for value in frame.axis_col),
        )

    @staticmethod
    def _vector_payload(vector: tuple[float, float, float] | np.ndarray) -> tuple[float, float, float]:
        return tuple(float(value) for value in np.asarray(vector, dtype=np.float64))

    @staticmethod
    def _matrix3_payload(matrix: object | None) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] | None:
        if matrix is None:
            return None
        values = np.asarray(matrix, dtype=np.float64)
        if values.shape != (3, 3) or not np.all(np.isfinite(values)):
            return None
        return tuple(
            tuple(float(values[row_index, col_index]) for col_index in range(3))
            for row_index in range(3)
        )

    @staticmethod
    def _build_mpr_cursor_payload(cursor: MprCursorState | None) -> MprCursorInfo | None:
        if cursor is None:
            return None
        orientation = np.asarray(cursor.orientation_world, dtype=np.float64)
        return MprCursorInfo(
            centerWorld=compat.ViewerService._vector_payload(cursor.center_world),
            referenceCenterWorld=compat.ViewerService._vector_payload(cursor.reference_center_world),
            orientationWorld=tuple(
                tuple(float(value) for value in orientation[row_index, :3])
                for row_index in range(3)
            ),
            linkedToVolumeRotation=bool(cursor.linked_to_volume_rotation),
        )

    @staticmethod
    def _plane_state_from_pose(plane_pose: PlanePose) -> MprObliquePlaneState:
        return MprObliquePlaneState(
            row=compat.ViewerService._vector_payload(plane_pose.row_world),
            col=compat.ViewerService._vector_payload(plane_pose.col_world),
            normal=compat.ViewerService._vector_payload(plane_pose.normal_world),
            is_oblique=bool(plane_pose.is_oblique),
        )

    def _build_mpr_plane_payload(
        self,
        view: ViewRecord,
        viewport_key: str,
        *,
        plane_pose: PlanePose | None = None,
        geometry: VolumeGeometry | None = None,
        image_transform: Any | None = None,
    ) -> MprPlaneInfo | None:
        if view.view_group is None:
            return None
        plane = self._plane_state_from_pose(plane_pose) if plane_pose is not None else self._default_mpr_oblique_plane(viewport_key)
        center_world = plane_pose.center_world if plane_pose is not None else (0.0, 0.0, 0.0)
        cursor_center_world = plane_pose.cursor_center_world if plane_pose is not None else center_world
        row_world = plane_pose.row_world if plane_pose is not None else plane.row
        col_world = plane_pose.col_world if plane_pose is not None else plane.col
        normal_world = plane_pose.normal_world if plane_pose is not None else plane.normal
        output_shape = plane_pose.output_shape if plane_pose is not None else (0, 0)
        pixel_spacing_normal_mm = (
            spacing_along_world_direction(geometry, normal_world)
            if geometry is not None
            else 1.0
        )
        return MprPlaneInfo(
            viewport=viewport_key,
            centerWorld=self._vector_payload(center_world),
            cursorCenterWorld=self._vector_payload(cursor_center_world),
            rowWorld=self._vector_payload(row_world),
            colWorld=self._vector_payload(col_world),
            normalWorld=self._vector_payload(normal_world),
            pixelSpacingRowMm=float(plane_pose.pixel_spacing_row_mm) if plane_pose is not None else 1.0,
            pixelSpacingColMm=float(plane_pose.pixel_spacing_col_mm) if plane_pose is not None else 1.0,
            pixelSpacingNormalMm=float(pixel_spacing_normal_mm),
            outputShape=(int(output_shape[0]), int(output_shape[1])),
            row=tuple(float(value) for value in plane.row),
            col=tuple(float(value) for value in plane.col),
            normal=tuple(float(value) for value in plane.normal),
            imageToCanvasMatrix=self._matrix3_payload(getattr(image_transform, "matrix", None)),
            isOblique=bool(plane_pose.is_oblique if plane_pose is not None else plane.is_oblique),
        )

    def _build_direction_orientation_overlay(
        self,
        view: ViewRecord,
        row_world: np.ndarray | None,
        col_world: np.ndarray | None,
    ) -> OrientationOverlay | None:
        row_direction = self._normalize_vector(np.asarray(row_world, dtype=np.float64)) if row_world is not None else None
        col_direction = self._normalize_vector(np.asarray(col_world, dtype=np.float64)) if col_world is not None else None
        if row_direction is None or col_direction is None:
            return None

        x_vector = col_direction * (-1.0 if view.hor_flip else 1.0)
        y_vector = row_direction * (-1.0 if view.ver_flip else 1.0)
        x_vector, y_vector = self._rotate_screen_axes(x_vector, y_vector, view.rotation_degrees)
        return OrientationOverlay(
            top=self._orientation_text_for_vector(-y_vector),
            right=self._orientation_text_for_vector(x_vector),
            bottom=self._orientation_text_for_vector(y_vector),
            left=self._orientation_text_for_vector(-x_vector),
        )

    def _build_stack_orientation_overlay(self, view: ViewRecord, dataset: Dataset | None) -> OrientationOverlay | None:
        orientation = self._get_dataset_orientation(dataset)
        if orientation is None:
            return None

        row_direction = self._normalize_vector(orientation[:3])
        column_direction = self._normalize_vector(orientation[3:6])
        if row_direction is None or column_direction is None:
            return None

        return self._build_direction_orientation_overlay(view, column_direction, row_direction)

    def _build_mpr_orientation_overlay(
        self,
        view: ViewRecord,
        viewport_key: str,
        plane_state: MprObliquePlaneState | None = None,
        *,
        plane_pose: PlanePose | None = None,
    ) -> OrientationOverlay:
        resolved_plane = plane_state or self._default_mpr_oblique_plane(viewport_key)
        try:
            series = compat.series_registry.get(view.series_id)
        except Exception:
            series = None
        transform = self._get_series_patient_transform(series) if series is not None else None
        use_model_label_directions = self._should_apply_mpr_model_rotation_to_plane_labels(
            view.view_group,
            plane_pose,
        )
        if plane_pose is not None and transform is not None:
            col_world = (
                self._get_mpr_model_source_direction(view.view_group, plane_pose.col_world)
                if use_model_label_directions
                else plane_pose.col_world
            )
            row_world = (
                self._get_mpr_model_source_direction(view.view_group, plane_pose.row_world)
                if use_model_label_directions
                else plane_pose.row_world
            )
            x_vector = mpr_geometry.normalize_patient_vector(
                col_world,
                fallback=np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
            )
            y_vector = mpr_geometry.normalize_patient_vector(
                row_world,
                fallback=np.asarray([0.0, 0.0, -1.0], dtype=np.float64),
            )
        elif plane_pose is not None:
            col_world = (
                self._get_mpr_model_source_direction(view.view_group, plane_pose.col_world)
                if use_model_label_directions
                else plane_pose.col_world
            )
            row_world = (
                self._get_mpr_model_source_direction(view.view_group, plane_pose.row_world)
                if use_model_label_directions
                else plane_pose.row_world
            )
            x_vector = mpr_geometry.fallback_volume_direction_to_patient_vector(col_world)
            y_vector = mpr_geometry.fallback_volume_direction_to_patient_vector(row_world)
        elif transform is not None:
            x_vector = mpr_geometry.normalize_patient_vector(
                transform.direction_step_to_patient(resolved_plane.col),
                fallback=np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
            )
            y_vector = mpr_geometry.normalize_patient_vector(
                transform.direction_step_to_patient(resolved_plane.row),
                fallback=np.asarray([0.0, 0.0, -1.0], dtype=np.float64),
            )
        else:
            x_vector = mpr_geometry.fallback_volume_direction_to_patient_vector(resolved_plane.col)
            y_vector = mpr_geometry.fallback_volume_direction_to_patient_vector(resolved_plane.row)

        if view.hor_flip:
            x_vector = -x_vector
        if view.ver_flip:
            y_vector = -y_vector
        x_vector, y_vector = self._rotate_screen_axes(x_vector, y_vector, view.rotation_degrees)

        orientation_text = (
            self._mpr_oblique_orientation_text_for_vector
            if use_model_label_directions or ((plane_pose is not None and plane_pose.is_oblique) or resolved_plane.is_oblique)
            else self._dominant_orientation_text_for_vector
        )

        return OrientationOverlay(
            top=orientation_text(-y_vector),
            right=orientation_text(x_vector),
            bottom=orientation_text(y_vector),
            left=orientation_text(-x_vector),
        )

    def _resolve_mpr_orientation_screen_axes(
        self,
        view: ViewRecord,
        normal_vector: np.ndarray,
        plane_state: MprObliquePlaneState | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        series = None
        try:
            series = compat.series_registry.get(view.series_id)
        except Exception:
            series = None
        transform = self._get_series_patient_transform(series) if series is not None else None
        if plane_state is not None and transform is not None:
            return (
                mpr_geometry.volume_direction_to_patient_vector(plane_state.col, transform),
                mpr_geometry.volume_direction_to_patient_vector(plane_state.row, transform),
            )
        return mpr_geometry.resolve_mpr_orientation_screen_axes(normal_vector, transform)

    @staticmethod
    def _build_mpr_viewport_label(viewport_key: str, plane_state: MprObliquePlaneState | None = None) -> str:
        if viewport_key == MPR_VIEWPORT_CORONAL:
            label = "CORONAL"
        elif viewport_key == MPR_VIEWPORT_SAGITTAL:
            label = "SAGITTAL"
        else:
            label = "AXIAL"
        if plane_state is not None and plane_state.is_oblique:
            return f"OBLIQUE {label}"
        return label

    @staticmethod
    def _build_window_label(window_width: float | None, window_center: float | None) -> str | None:
        ww = compat.ViewerService._format_number(window_width, precision=0)
        wl = compat.ViewerService._format_number(window_center, precision=0)
        if ww is None and wl is None:
            return None
        return f"W: {ww or '-'} L: {wl or '-'}"

    @staticmethod
    def _resolve_window_min(window_width: float | None, window_center: float | None) -> float | None:
        if window_width is None or window_center is None:
            return None
        return float(window_center) - float(window_width) / 2.0

    @staticmethod
    def _resolve_window_max(window_width: float | None, window_center: float | None) -> float | None:
        if window_width is None or window_center is None:
            return None
        return float(window_center) + float(window_width) / 2.0

    @staticmethod
    def _build_pet_window_label(
        display: FusionPetDisplayVolume,
        window_width: float | None,
        window_center: float | None,
    ) -> str | None:
        low = compat.ViewerService._resolve_window_min(window_width, window_center)
        high = compat.ViewerService._resolve_window_max(window_width, window_center)
        if low is None or high is None:
            return None
        low_text = f"{float(low):.2f}"
        high_text = f"{float(high):.2f}"
        prefix = "SUV" if display.unit in {FUSION_PET_UNIT_SUV_BW, FUSION_PET_UNIT_SUV_BSA, FUSION_PET_UNIT_SUL} else "PET"
        unit_label = compat.ViewerService._strip_trailing_unit_detail(display.unit_label)
        return f"{prefix}:{low_text}--{high_text}{unit_label}".strip()

    @staticmethod
    def _strip_trailing_unit_detail(value: str | None) -> str:
        text = str(value or "").strip()
        if text.endswith(")") and "(" in text:
            prefix = text.rsplit("(", 1)[0].strip()
            if prefix:
                return prefix
        return text

    def _with_pet_window_corner_info(
        self,
        corner_info: CornerInfoOverlay,
        display: FusionPetDisplayVolume,
        window_width: float | None,
        window_center: float | None,
    ) -> CornerInfoOverlay:
        pet_window = self._build_pet_window_label(display, window_width, window_center)
        if not pet_window:
            return corner_info
        default_window = self._build_window_label(window_width, window_center)
        tags = dict(corner_info.tags)
        tags["windowLevel"] = (pet_window,)
        bottom_left = tuple(
            pet_window
            if (default_window and line == default_window) or str(line).strip().upper().startswith("W:")
            else line
            for line in corner_info.bottom_left
        )
        if pet_window not in bottom_left:
            bottom_left = (pet_window, *bottom_left)
        return replace(corner_info, bottom_left=bottom_left, tags=tags)

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
        text = compat.ViewerService._safe_text(value)
        if not text or len(text) != 8 or not text.isdigit():
            return text
        return f"{text[:4]}.{text[4:6]}.{text[6:8]}"

    @staticmethod
    def _format_dicom_time(value) -> str | None:
        text = compat.ViewerService._safe_text(value)
        if not text:
            return None
        digits = ''.join(ch for ch in text if ch.isdigit())
        if len(digits) < 6:
            return text
        return f"{digits[:2]}:{digits[2:4]}:{digits[4:6]}"

    @staticmethod
    def _window_array(
        pixels: np.ndarray,
        window_width: float | None,
        window_center: float | None,
        *,
        pixel_min: float | None = None,
        pixel_max: float | None = None,
    ) -> np.ndarray:
        if pixels.ndim == 3 and pixels.shape[-1] in (3, 4):
            color_pixels = pixels[..., :3]
            if color_pixels.dtype == np.uint8:
                return color_pixels
            return np.clip(color_pixels, 0, 255).astype(np.uint8)

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
        view.drag_origin_arcball_z = None

    @staticmethod
    def _encode_image(image: Image.Image, image_format: ImageFormat, *, fast_preview: bool = False) -> bytes:
        output = io.BytesIO()
        if image_format == "jpeg":
            # JPEG is only used for transient interaction previews. Settled frames
            # stay PNG so overlays and measurements align with lossless pixels.
            image.convert("RGB").save(output, format="JPEG", quality=FAST_PREVIEW_JPEG_QUALITY)
        elif image_format == "webp":
            if fast_preview:
                image.save(
                    output,
                    format="WEBP",
                    lossless=False,
                    quality=WEBP_PREVIEW_QUALITY,
                    method=WEBP_PREVIEW_METHOD,
                )
            else:
                image.save(output, format="WEBP", lossless=True)
        else:
            # PNG is lossless at every compression level. Keep all viewer PNG
            # frames at a low compression level to reduce encode latency and
            # avoid final-frame tail spikes during interaction.
            image.save(
                output,
                format="PNG",
                compress_level=PNG_COMPRESS_LEVEL,
                optimize=False,
            )
        return output.getvalue()

    @staticmethod
    def _encode_3d_image(image: Image.Image, image_format: ImageFormat, *, fast_preview: bool = False) -> bytes:
        if image_format != "webp":
            return ViewerPresentationMixin._encode_image(image, image_format, fast_preview=fast_preview)

        output = io.BytesIO()
        if fast_preview:
            image.save(
                output,
                format="WEBP",
                lossless=False,
                quality=WEBP_PREVIEW_QUALITY,
                method=WEBP_PREVIEW_METHOD,
            )
        else:
            # A rendered 3D RGB frame has already passed through transfer functions,
            # lighting and ray sampling. High-quality lossy WebP avoids the large
            # lossless encode tail without changing quantitative 2D/MPR transport.
            image.save(
                output,
                format="WEBP",
                lossless=False,
                quality=WEBP_3D_FINAL_QUALITY,
                method=WEBP_3D_FINAL_METHOD,
            )
        return output.getvalue()
