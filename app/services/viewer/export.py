from __future__ import annotations

"""Viewer exports and registration artifacts."""

from app.services.viewer.shared import *  # noqa: F403


class ViewerExportMixin:
    def export_view_by_id(
        self,
        view_id: str,
        export_format: str,
        *,
        overlays: ViewExportOverlaysPayload | None = None,
        workspace_id: str | None = None,
    ) -> ExportedFileResult:
        view = compat.view_registry.get(view_id, workspace_id=workspace_id)
        safe_view_type = str(view.view_type or "view").lower()

        if export_format == "dicom-sr":
            reference_dataset = self._get_export_reference_dataset(view)
            dicom_sr_bytes = build_measurement_sr_dicom_bytes(view, overlays, reference_dataset)
            return ExportedFileResult(
                file_bytes=dicom_sr_bytes,
                file_name=f"{view.view_id}-{safe_view_type}-measurements-sr.dcm",
                media_type="application/dicom",
            )

        if export_format == "dicom-gsps":
            reference_dataset = self._get_export_reference_dataset(view)
            gsps_bytes = build_gsps_dicom_bytes(view, overlays, reference_dataset)
            return ExportedFileResult(
                file_bytes=gsps_bytes,
                file_name=f"{view.view_id}-{safe_view_type}-presentation-state.dcm",
                media_type="application/dicom",
            )

        if export_format == "png":
            rendered = self._render_by_view_type(view, image_format="png", fast_preview=False)
            if overlays and (overlays.annotations or overlays.measurements):
                try:
                    image = Image.open(io.BytesIO(rendered.image_bytes)).convert("RGB")
                    image = self._apply_export_overlays(image, overlays)
                    rendered_bytes = self._encode_image(image, "png", fast_preview=False)
                except Exception as exc:  # pragma: no cover - defensive
                    raise HTTPException(status_code=500, detail="Failed to render export overlays") from exc
            else:
                rendered_bytes = rendered.image_bytes
            return ExportedFileResult(
                file_bytes=rendered_bytes,
                file_name=f"{view.view_id}-{safe_view_type}.png",
                media_type="image/png",
            )
        if export_format != "dicom":
            raise HTTPException(status_code=400, detail="Unsupported export format")

        rendered = self._render_by_view_type(view, image_format="png", fast_preview=False)
        try:
            image = Image.open(io.BytesIO(rendered.image_bytes)).convert("RGB")
        except Exception as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=500, detail="Failed to decode rendered image for DICOM export") from exc

        if overlays and (overlays.annotations or overlays.measurements):
            image = self._apply_export_overlays(image, overlays)

        reference_dataset = self._get_export_reference_dataset(view)
        dicom_bytes = self._build_secondary_capture_dicom_bytes(view, image, reference_dataset)
        return ExportedFileResult(
            file_bytes=dicom_bytes,
            file_name=f"{view.view_id}-{safe_view_type}.dcm",
            media_type="application/dicom",
        )

    def export_fusion_registration(
        self,
        payload: FusionRegistrationExportRequest,
        *,
        workspace_id: str | None = None,
    ) -> FusionRegistrationExportResponse:
        output_directory = self._resolve_fusion_registration_output_directory(payload.output_directory)
        context = self._build_fusion_registration_export_context(
            payload.view_id,
            payload.series_description,
            workspace_id=workspace_id,
        )

        if payload.mode == "br":
            file_path = self._write_fusion_registration_sidecar(
                output_directory,
                group=context.group,
                ct_series=context.ct_series,
                pet_series=context.pet_series,
                pet_display=context.pet_display,
                series_description=context.series_description,
            )
            compat.view_group_registry.save_fusion_registration(context.group)
            return FusionRegistrationExportResponse(
                mode="br",
                directoryPath=str(file_path.parent),
                filePath=str(file_path),
                fileCount=1,
                seriesDescription=context.series_description,
                petUnit=context.pet_display.unit,
                petUnitLabel=context.pet_display.unit_label,
            )

        if payload.mode != "newDicom":
            raise HTTPException(status_code=400, detail="Unsupported fusion registration export mode")

        directory_path, file_count = self._write_fusion_registration_dicom_series(
            output_directory,
            group=context.group,
            ct_series=context.ct_series,
            pet_series=context.pet_series,
            ct_volume=context.ct_volume,
            ct_geometry=context.ct_geometry,
            pet_geometry=context.pet_geometry,
            pet_display=context.pet_display,
            series_description=context.series_description,
        )
        compat.view_group_registry.save_fusion_registration(context.group)
        return FusionRegistrationExportResponse(
            mode="newDicom",
            directoryPath=str(directory_path),
            filePath=None,
            fileCount=file_count,
            seriesDescription=context.series_description,
            petUnit=context.pet_display.unit,
            petUnitLabel=context.pet_display.unit_label,
        )

    def export_fusion_registration_artifact(
        self,
        payload: FusionRegistrationArtifactExportRequest,
        *,
        workspace_id: str | None = None,
    ) -> ExportedFileResult:
        context = self._build_fusion_registration_export_context(
            payload.view_id,
            payload.series_description,
            workspace_id=workspace_id,
        )
        if payload.mode == "br":
            file_name = f"{self._safe_fusion_file_name_part(context.series_description)}.br"
            sidecar_payload = self._build_fusion_registration_sidecar_payload(
                group=context.group,
                ct_series=context.ct_series,
                pet_series=context.pet_series,
                pet_display=context.pet_display,
                series_description=context.series_description,
            )
            compat.view_group_registry.save_fusion_registration(context.group)
            return ExportedFileResult(
                file_bytes=json.dumps(sidecar_payload, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name=file_name,
                media_type="application/json",
                extra_headers={
                    "x-dicomvision-artifact-kind": "br",
                    "x-dicomvision-file-count": "1",
                },
            )

        if payload.mode != "newDicom":
            raise HTTPException(status_code=400, detail="Unsupported fusion registration export mode")

        series_folder = self._safe_fusion_file_name_part(context.series_description)
        datasets = self._build_fusion_registration_dicom_datasets(
            group=context.group,
            ct_series=context.ct_series,
            pet_series=context.pet_series,
            ct_volume=context.ct_volume,
            ct_geometry=context.ct_geometry,
            pet_geometry=context.pet_geometry,
            pet_display=context.pet_display,
            series_description=context.series_description,
        )
        archive = io.BytesIO()
        try:
            with zipfile.ZipFile(archive, mode="w", compression=zipfile.ZIP_DEFLATED) as zip_file:
                for index, dataset in enumerate(datasets, start=1):
                    buffer = io.BytesIO()
                    dcmwrite(buffer, dataset, write_like_original=False)
                    zip_file.writestr(f"{series_folder}/IM{index:06d}.dcm", buffer.getvalue())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to package DICOM export artifact: {exc}") from exc

        compat.view_group_registry.save_fusion_registration(context.group)
        return ExportedFileResult(
            file_bytes=archive.getvalue(),
            file_name=f"{series_folder}.zip",
            media_type="application/zip",
            extra_headers={
                "x-dicomvision-artifact-kind": "zip",
                "x-dicomvision-file-count": str(len(datasets)),
            },
        )

    def _build_fusion_registration_export_context(
        self,
        view_id: str,
        series_description: str | None,
        *,
        workspace_id: str | None = None,
    ) -> FusionRegistrationExportContext:
        view = compat.view_registry.get(view_id, workspace_id=workspace_id)
        if not self._is_fusion_view_type(view.view_type):
            raise HTTPException(status_code=400, detail="viewId does not refer to a PET/CT fusion view")

        group, ct_series, pet_series = self._resolve_fusion_group_series(view)
        resolved_description = self._resolve_fusion_registration_series_description(
            series_description,
            pet_series,
        )
        ct_volume = self._get_series_volume(ct_series)
        pet_volume = self._get_series_volume(pet_series)
        ct_geometry = self._get_series_volume_geometry(ct_series, ct_volume.shape)
        pet_geometry = self._get_series_volume_geometry(pet_series, pet_volume.shape)
        pet_display = self._build_fusion_pet_display_volume(pet_series, pet_volume, group.fusion_pet_unit)
        group.fusion_pet_unit = pet_display.unit
        return FusionRegistrationExportContext(
            group=group,
            ct_series=ct_series,
            pet_series=pet_series,
            ct_volume=ct_volume,
            pet_volume=pet_volume,
            ct_geometry=ct_geometry,
            pet_geometry=pet_geometry,
            pet_display=pet_display,
            series_description=resolved_description,
        )

    @staticmethod
    def _resolve_fusion_registration_output_directory(value: str) -> Path:
        text = str(value or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="outputDirectory is required")
        directory = Path(text).expanduser().resolve()
        if directory.exists() and not directory.is_dir():
            raise HTTPException(status_code=400, detail="outputDirectory must be a directory")
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to create output directory: {exc}") from exc
        return directory

    @classmethod
    def _resolve_fusion_registration_series_description(
        cls,
        value: str | None,
        pet_series: SeriesRecord,
    ) -> str:
        fallback = f"{str(pet_series.series_description or pet_series.series_id or 'PET').strip() or 'PET'}_Reg"
        description = str(value or fallback).strip() or fallback
        return description[:64]

    @staticmethod
    def _safe_fusion_file_name_part(value: object) -> str:
        text = str(value or "").strip()
        sanitized = "".join("-" if char in '\\/:*?"<>|\r\n\t' else char for char in text)
        sanitized = "-".join(part for part in sanitized.split() if part).strip(".-_ ")
        return sanitized or "fusion-registration"

    @classmethod
    def _resolve_unique_path(cls, path: Path) -> Path:
        if not path.exists():
            return path
        stem = path.stem
        suffix = path.suffix
        parent = path.parent
        index = 1
        while True:
            candidate = parent / f"{stem}-{index}{suffix}"
            if not candidate.exists():
                return candidate
            index += 1

    @classmethod
    def _resolve_unique_directory(cls, directory: Path) -> Path:
        if not directory.exists():
            return directory
        parent = directory.parent
        stem = directory.name
        index = 1
        while True:
            candidate = parent / f"{stem}-{index}"
            if not candidate.exists():
                return candidate
            index += 1

    @staticmethod
    def _format_dicom_ds(value: float) -> str:
        text = format(float(value), ".8g")
        return text if len(text) <= 16 else format(float(value), ".6e")

    def _build_fusion_registration_sidecar_payload(
        self,
        *,
        group: ViewGroupRecord,
        ct_series: SeriesRecord,
        pet_series: SeriesRecord,
        pet_display: FusionPetDisplayVolume,
        series_description: str,
    ) -> dict[str, object]:
        registration = group.fusion_registration
        return {
            "format": "DicomVisionFusionRegistration",
            "version": 1,
            "createdAt": datetime.now().isoformat(timespec="seconds"),
            "seriesDescription": series_description,
            "ct": {
                "seriesId": ct_series.series_id,
                "seriesInstanceUid": ct_series.series_instance_uid,
                "seriesDescription": ct_series.series_description,
            },
            "pet": {
                "seriesId": pet_series.series_id,
                "seriesInstanceUid": pet_series.series_instance_uid,
                "seriesDescription": pet_series.series_description,
                "unit": pet_display.unit,
                "unitLabel": pet_display.unit_label,
                "sourceUnits": pet_display.source_units,
                "scale": float(pet_display.scale),
                "window": {
                    "min": self._resolve_window_min(
                        group.fusion_pet_window.window_width,
                        group.fusion_pet_window.window_center,
                    ),
                    "max": self._resolve_window_max(
                        group.fusion_pet_window.window_width,
                        group.fusion_pet_window.window_center,
                    ),
                },
            },
            "registration": {
                "translateRowMm": float(registration.translate_row_mm),
                "translateColMm": float(registration.translate_col_mm),
                "rotationDegrees": float(registration.rotation_degrees),
            },
        }

    def _write_fusion_registration_sidecar(
        self,
        output_directory: Path,
        *,
        group: ViewGroupRecord,
        ct_series: SeriesRecord,
        pet_series: SeriesRecord,
        pet_display: FusionPetDisplayVolume,
        series_description: str,
    ) -> Path:
        file_name = f"{self._safe_fusion_file_name_part(series_description)}.br"
        file_path = self._resolve_unique_path(output_directory / file_name)
        payload = self._build_fusion_registration_sidecar_payload(
            group=group,
            ct_series=ct_series,
            pet_series=pet_series,
            pet_display=pet_display,
            series_description=series_description,
        )
        try:
            file_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to write .br file: {exc}") from exc
        return file_path

    @staticmethod
    def _resolve_pet_dicom_units(display: FusionPetDisplayVolume) -> str:
        if display.unit == FUSION_PET_UNIT_SUV_BW:
            return "GML"
        if display.unit == FUSION_PET_UNIT_SOURCE and display.source_units:
            return display.source_units
        return "CNTS"

    def _resample_fusion_pet_volume_to_ct_grid(
        self,
        *,
        group: ViewGroupRecord,
        ct_volume: np.ndarray,
        ct_geometry: VolumeGeometry,
        pet_geometry: VolumeGeometry,
        pet_display: FusionPetDisplayVolume,
    ) -> list[tuple[PlanePose, np.ndarray]]:
        ct_shape = tuple(int(value) for value in ct_volume.shape)
        slices: list[tuple[PlanePose, np.ndarray]] = []
        for axial_index in range(ct_shape[0]):
            plane = build_ct_axial_plane(ct_geometry, ct_shape, axial_index)
            pet_plane = transform_pet_sampling_plane(plane, group.fusion_registration)
            pet_slice = compat.reslice_plane(
                pet_display.volume,
                pet_geometry,
                pet_plane,
                ResliceMipConfig(enabled=False),
                interpolation_order=1,
            )
            slices.append((plane, np.asarray(pet_slice, dtype=np.float32)))
        return slices

    @staticmethod
    def _resolve_dicom_rescale_for_slices(slices: list[np.ndarray]) -> tuple[float, float]:
        finite_arrays: list[np.ndarray] = []
        for item in slices:
            array = np.asarray(item, dtype=np.float32)
            finite = array[np.isfinite(array)]
            if finite.size:
                finite_arrays.append(finite)
        if not finite_arrays:
            return (1.0, 0.0)
        finite_values = np.concatenate(finite_arrays)
        low = float(np.min(finite_values))
        high = float(np.max(finite_values))
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            return (1.0, low if np.isfinite(low) else 0.0)
        return ((high - low) / 65535.0, low)

    @staticmethod
    def _encode_dicom_uint16_pixels(pixels: np.ndarray, *, slope: float, intercept: float) -> np.ndarray:
        source = np.asarray(pixels, dtype=np.float32)
        source = np.where(np.isfinite(source), source, intercept)
        if abs(float(slope)) <= 1e-12:
            encoded = np.zeros(source.shape, dtype=np.uint16)
        else:
            encoded = np.clip(np.rint((source - float(intercept)) / float(slope)), 0, 65535).astype(np.uint16)
        return np.ascontiguousarray(encoded)

    def _apply_fusion_registration_private_tags(
        self,
        dataset: Dataset,
        *,
        group: ViewGroupRecord,
        pet_display: FusionPetDisplayVolume,
    ) -> None:
        registration = group.fusion_registration
        dataset.add_new((0x0011, 0x0010), "LO", "DICOMVISION_FUSION")
        dataset.add_new((0x0011, 0x1001), "LO", pet_display.unit)
        dataset.add_new((0x0011, 0x1002), "LO", pet_display.unit_label)
        dataset.add_new((0x0011, 0x1003), "DS", self._format_dicom_ds(registration.translate_row_mm))
        dataset.add_new((0x0011, 0x1004), "DS", self._format_dicom_ds(registration.translate_col_mm))
        dataset.add_new((0x0011, 0x1005), "DS", self._format_dicom_ds(registration.rotation_degrees))
        window_min = self._resolve_window_min(group.fusion_pet_window.window_width, group.fusion_pet_window.window_center)
        window_max = self._resolve_window_max(group.fusion_pet_window.window_width, group.fusion_pet_window.window_center)
        if window_min is not None:
            dataset.add_new((0x0011, 0x1006), "DS", self._format_dicom_ds(window_min))
        if window_max is not None:
            dataset.add_new((0x0011, 0x1007), "DS", self._format_dicom_ds(window_max))

    @staticmethod
    def _resolve_derived_series_number(dataset: Dataset) -> int:
        try:
            return int(float(getattr(dataset, "SeriesNumber", 0) or 0)) + 1000
        except (TypeError, ValueError):
            return 1000

    def _build_fusion_registration_dicom_dataset(
        self,
        *,
        reference_dataset: Dataset | None,
        plane: PlanePose,
        pixels: np.ndarray,
        group: ViewGroupRecord,
        pet_display: FusionPetDisplayVolume,
        series_description: str,
        series_instance_uid: str,
        instance_number: int,
        rescale_slope: float,
        rescale_intercept: float,
    ) -> Dataset:
        dataset = deepcopy(reference_dataset) if reference_dataset is not None else Dataset()
        now = datetime.now()
        sop_instance_uid = generate_uid()
        sop_class_uid = str(getattr(dataset, "SOPClassUID", "") or SecondaryCaptureImageStorage)

        file_meta = getattr(dataset, "file_meta", None)
        if file_meta is None:
            file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = sop_class_uid
        file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        file_meta.ImplementationClassUID = PYDICOM_IMPLEMENTATION_UID
        dataset.file_meta = file_meta
        dataset.is_little_endian = True
        dataset.is_implicit_VR = False

        encoded_pixels = self._encode_dicom_uint16_pixels(
            pixels,
            slope=rescale_slope,
            intercept=rescale_intercept,
        )
        rows, columns = encoded_pixels.shape
        top_left_world = (
            np.asarray(plane.center_world, dtype=np.float64)
            - np.asarray(plane.col_world, dtype=np.float64) * plane.pixel_spacing_col_mm * ((float(columns) - 1.0) / 2.0)
            - np.asarray(plane.row_world, dtype=np.float64) * plane.pixel_spacing_row_mm * ((float(rows) - 1.0) / 2.0)
        )

        for keyword in (
            "NumberOfFrames",
            "SharedFunctionalGroupsSequence",
            "PerFrameFunctionalGroupsSequence",
            "FloatPixelData",
            "DoubleFloatPixelData",
        ):
            if hasattr(dataset, keyword):
                delattr(dataset, keyword)

        dataset.SOPClassUID = sop_class_uid
        dataset.SOPInstanceUID = sop_instance_uid
        dataset.SeriesInstanceUID = series_instance_uid
        dataset.Modality = str(getattr(dataset, "Modality", "") or "PT")
        dataset.SeriesDescription = series_description
        dataset.SeriesNumber = self._resolve_derived_series_number(dataset)
        dataset.InstanceNumber = instance_number
        dataset.ImageType = ["DERIVED", "SECONDARY", "REGISTRATION"]
        dataset.DerivationDescription = (
            "DicomVision PET/CT registration export; "
            f"unit={pet_display.unit}; "
            f"translateRowMm={self._format_dicom_ds(group.fusion_registration.translate_row_mm)}; "
            f"translateColMm={self._format_dicom_ds(group.fusion_registration.translate_col_mm)}; "
            f"rotationDegrees={self._format_dicom_ds(group.fusion_registration.rotation_degrees)}"
        )
        dataset.ContentDate = now.strftime("%Y%m%d")
        dataset.ContentTime = now.strftime("%H%M%S")
        dataset.InstanceCreationDate = dataset.ContentDate
        dataset.InstanceCreationTime = dataset.ContentTime
        dataset.Rows = int(rows)
        dataset.Columns = int(columns)
        dataset.SamplesPerPixel = 1
        dataset.PhotometricInterpretation = "MONOCHROME2"
        dataset.BitsAllocated = 16
        dataset.BitsStored = 16
        dataset.HighBit = 15
        dataset.PixelRepresentation = 0
        dataset.RescaleSlope = self._format_dicom_ds(rescale_slope)
        dataset.RescaleIntercept = self._format_dicom_ds(rescale_intercept)
        dataset.RescaleType = pet_display.unit
        dataset.Units = self._resolve_pet_dicom_units(pet_display)
        dataset.WindowWidth = self._format_dicom_ds(group.fusion_pet_window.window_width or 1.0)
        dataset.WindowCenter = self._format_dicom_ds(group.fusion_pet_window.window_center or 0.5)
        dataset.PixelSpacing = [
            self._format_dicom_ds(plane.pixel_spacing_row_mm),
            self._format_dicom_ds(plane.pixel_spacing_col_mm),
        ]
        dataset.ImageOrientationPatient = [
            self._format_dicom_ds(float(value))
            for value in (*np.asarray(plane.row_world, dtype=np.float64), *np.asarray(plane.col_world, dtype=np.float64))
        ]
        dataset.ImagePositionPatient = [self._format_dicom_ds(float(value)) for value in top_left_world]
        dataset.SliceLocation = self._format_dicom_ds(float(np.dot(plane.normal_world, plane.center_world)))
        dataset.PixelData = encoded_pixels.tobytes()
        self._apply_fusion_registration_private_tags(dataset, group=group, pet_display=pet_display)
        return dataset

    def _build_fusion_registration_dicom_datasets(
        self,
        *,
        group: ViewGroupRecord,
        ct_series: SeriesRecord,
        pet_series: SeriesRecord,
        ct_volume: np.ndarray,
        ct_geometry: VolumeGeometry,
        pet_geometry: VolumeGeometry,
        pet_display: FusionPetDisplayVolume,
        series_description: str,
    ) -> list[Dataset]:
        _, reference_cached = self._get_reference_instance_and_cache(pet_series)
        series_instance_uid = generate_uid()
        resampled_slices = self._resample_fusion_pet_volume_to_ct_grid(
            group=group,
            ct_volume=ct_volume,
            ct_geometry=ct_geometry,
            pet_geometry=pet_geometry,
            pet_display=pet_display,
        )
        rescale_slope, rescale_intercept = self._resolve_dicom_rescale_for_slices([pixels for _, pixels in resampled_slices])

        datasets: list[Dataset] = []
        for index, (plane, pixels) in enumerate(resampled_slices, start=1):
            dataset = self._build_fusion_registration_dicom_dataset(
                reference_dataset=reference_cached.dataset if reference_cached is not None else None,
                plane=plane,
                pixels=pixels,
                group=group,
                pet_display=pet_display,
                series_description=series_description,
                series_instance_uid=series_instance_uid,
                instance_number=index,
                rescale_slope=rescale_slope,
                rescale_intercept=rescale_intercept,
            )
            datasets.append(dataset)
        return datasets

    def _write_fusion_registration_dicom_series(
        self,
        output_directory: Path,
        *,
        group: ViewGroupRecord,
        ct_series: SeriesRecord,
        pet_series: SeriesRecord,
        ct_volume: np.ndarray,
        ct_geometry: VolumeGeometry,
        pet_geometry: VolumeGeometry,
        pet_display: FusionPetDisplayVolume,
        series_description: str,
    ) -> tuple[Path, int]:
        series_folder = self._safe_fusion_file_name_part(series_description)
        directory_path = self._resolve_unique_directory(output_directory / series_folder)
        try:
            directory_path.mkdir(parents=True, exist_ok=False)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to create DICOM output directory: {exc}") from exc

        datasets = self._build_fusion_registration_dicom_datasets(
            group=group,
            ct_series=ct_series,
            pet_series=pet_series,
            ct_volume=ct_volume,
            ct_geometry=ct_geometry,
            pet_geometry=pet_geometry,
            pet_display=pet_display,
            series_description=series_description,
        )
        for index, dataset in enumerate(datasets, start=1):
            file_path = directory_path / f"IM{index:06d}.dcm"
            try:
                dcmwrite(str(file_path), dataset, write_like_original=False)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Failed to write DICOM file: {exc}") from exc

        return (directory_path, len(datasets))

    def _apply_export_overlays(self, image: Image.Image, overlays: ViewExportOverlaysPayload) -> Image.Image:
        canvas = image.convert("RGBA")
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()
        width, height = canvas.size

        for measurement in overlays.measurements:
            points = tuple((point.x * width, point.y * height) for point in measurement.points)
            self._draw_export_measurement(draw, font, measurement.tool_type, points, measurement.label_lines, width, height)

        for annotation in overlays.annotations:
            points = tuple((point.x * width, point.y * height) for point in annotation.points)
            self._draw_export_annotation(draw, font, points, annotation.text, annotation.color, annotation.size, width, height)

        return canvas.convert("RGB")

    def _draw_export_measurement(
        self,
        draw: ImageDraw.ImageDraw,
        font: ImageFont.ImageFont,
        tool_type: str,
        points: tuple[tuple[float, float], ...],
        label_lines: list[str],
        width: int,
        height: int,
    ) -> None:
        if not points:
            return

        if tool_type in {"line", "alignment-horizontal", "alignment-vertical"} and len(points) >= 2:
            self._draw_export_polyline(draw, points[:2])
        elif tool_type == "rect" and len(points) >= 2:
            left, right = sorted((points[0][0], points[1][0]))
            top, bottom = sorted((points[0][1], points[1][1]))
            draw.rectangle((left, top, right, bottom), outline=(3, 15, 24, 235), width=5)
            draw.rectangle((left, top, right, bottom), outline=(85, 231, 255, 255), width=2)
        elif tool_type == "ellipse" and len(points) >= 2:
            left, right = sorted((points[0][0], points[1][0]))
            top, bottom = sorted((points[0][1], points[1][1]))
            draw.ellipse((left, top, right, bottom), outline=(3, 15, 24, 235), width=5)
            draw.ellipse((left, top, right, bottom), outline=(85, 231, 255, 255), width=2)
        elif tool_type == "angle" and len(points) >= 2:
            self._draw_export_polyline(draw, points[:2])
            if len(points) >= 3:
                self._draw_export_polyline(draw, points[1:3])
        elif tool_type == "curve" and len(points) >= 2:
            self._draw_export_polyline(draw, build_smooth_path_points(points))
        elif tool_type == "freeform" and len(points) >= 3:
            self._draw_export_polyline(draw, build_smooth_path_points(points, close_path=True))
        else:
            return

        if label_lines:
            anchor = points[-1] if tool_type == "curve" else points[1] if len(points) >= 2 else points[0]
            self._draw_export_label(draw, font, label_lines, anchor[0] + 12, anchor[1] - 32, width, height)

    @staticmethod
    def _draw_export_polyline(draw: ImageDraw.ImageDraw, points: tuple[tuple[float, float], ...]) -> None:
        draw.line(points, fill=(3, 15, 24, 235), width=5, joint="curve")
        draw.line(points, fill=(85, 231, 255, 255), width=2, joint="curve")

    def _draw_export_annotation(
        self,
        draw: ImageDraw.ImageDraw,
        font: ImageFont.ImageFont,
        points: tuple[tuple[float, float], ...],
        text: str,
        color: str,
        size: str,
        width: int,
        height: int,
    ) -> None:
        if len(points) < 2:
            return

        stroke = self._parse_export_color(color)
        stroke_width = 3 if size == "lg" else 2
        draw.line(points[:2], fill=stroke, width=stroke_width)
        self._draw_export_arrow_head(draw, points[0], points[1], stroke, stroke_width * 3)

        visible_text = text.strip()
        if visible_text:
            self._draw_export_label(draw, font, [visible_text], points[0][0] + 12, points[0][1] - 30, width, height, text_fill=stroke)

    @staticmethod
    def _draw_export_arrow_head(
        draw: ImageDraw.ImageDraw,
        start: tuple[float, float],
        end: tuple[float, float],
        fill: tuple[int, int, int, int],
        size: int,
    ) -> None:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = float(np.hypot(dx, dy))
        if length < 1e-6:
            return

        ux = dx / length
        uy = dy / length
        back_x = end[0] - ux * size * 2.8
        back_y = end[1] - uy * size * 2.8
        perp_x = -uy * size
        perp_y = ux * size
        draw.polygon(
            (
                end,
                (back_x + perp_x, back_y + perp_y),
                (back_x - perp_x, back_y - perp_y),
            ),
            fill=fill,
        )

    @staticmethod
    def _draw_export_label(
        draw: ImageDraw.ImageDraw,
        font: ImageFont.ImageFont,
        lines: list[str],
        x: float,
        y: float,
        width: int,
        height: int,
        *,
        text_fill: tuple[int, int, int, int] = (235, 245, 255, 255),
    ) -> None:
        visible_lines = [line.strip() for line in lines if line.strip()]
        if not visible_lines:
            return

        padding_x = 8
        padding_y = 6
        line_gap = 3
        line_sizes = [draw.textbbox((0, 0), line, font=font) for line in visible_lines]
        text_width = max((bbox[2] - bbox[0]) for bbox in line_sizes)
        text_height = sum((bbox[3] - bbox[1]) for bbox in line_sizes) + max(0, len(visible_lines) - 1) * line_gap
        left = max(6, min(width - text_width - padding_x * 2 - 6, int(round(x))))
        top = max(6, min(height - text_height - padding_y * 2 - 6, int(round(y))))
        right = left + text_width + padding_x * 2
        bottom = top + text_height + padding_y * 2

        draw.rounded_rectangle((left, top, right, bottom), radius=7, fill=(7, 16, 28, 232), outline=(108, 201, 255, 188), width=1)
        cursor_y = top + padding_y
        for index, line in enumerate(visible_lines):
            bbox = line_sizes[index]
            draw.text((left + padding_x, cursor_y), line, fill=text_fill, font=font)
            cursor_y += (bbox[3] - bbox[1]) + line_gap

    @staticmethod
    def _parse_export_color(value: str) -> tuple[int, int, int, int]:
        hex_value = value.strip().lstrip("#")
        if len(hex_value) == 3:
            hex_value = "".join(char * 2 for char in hex_value)
        if len(hex_value) != 6:
            return (255, 209, 102, 255)
        try:
            red = int(hex_value[0:2], 16)
            green = int(hex_value[2:4], 16)
            blue = int(hex_value[4:6], 16)
        except ValueError:
            return (255, 209, 102, 255)
        return (red, green, blue, 255)
