from __future__ import annotations

import base64
import io
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
from PIL import Image
from pydicom.dataset import Dataset
from pydicom.multival import MultiValue

from app.core.logging import get_logger
from app.models.viewer import InstanceRecord, SeriesRecord
from app.schemas.dicom import FourDPhaseItem, FourDPhasesResponse, SeriesSummary
from app.services.dicom_cache import dicom_cache
from app.services.dicom_geometry import build_standardized_volume, get_dataset_orientation, get_dataset_position


logger = get_logger(__name__)

MPR_AXIAL = "mpr-ax"
MPR_CORONAL = "mpr-cor"
MPR_SAGITTAL = "mpr-sag"
MAX_PREVIEW_SLICES = 96
PREVIEW_SIZE = (320, 320)

_PERCENT_PHASE_RE = re.compile(
    r"(?<!\d)(100(?:\.0+)?|[0-9]{1,2}(?:\.\d+)?)[\s_-]*(?:%|pct\b|percent\b)",
    re.IGNORECASE,
)
_NAMED_PHASE_RE = re.compile(r"\b(?:phase|ph)[\s_:#-]*(\d{1,3}(?:\.\d+)?)\b", re.IGNORECASE)
_SHORT_PHASE_RE = re.compile(r"\bp[\s_-]*(\d{1,3})(?![a-zA-Z0-9])", re.IGNORECASE)
_FIRST_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


@dataclass(frozen=True)
class PhaseValue:
    sort_value: float
    label_value: str
    source: str


@dataclass(frozen=True)
class PhaseCandidate:
    series: SeriesRecord
    phase: PhaseValue


class FourDService:
    def get_four_d_phases(
        self,
        series_id: str,
        series_records: list[SeriesRecord],
    ) -> FourDPhasesResponse:
        target_series = next((series for series in series_records if series.series_id == series_id), None)
        if target_series is None:
            return FourDPhasesResponse(seriesId=series_id, isFourDSeries=False, fourDPhaseCount=0, fourDPhases=[])

        phase_entries = self._resolve_multi_series_phase_entries(target_series, series_records)
        if not phase_entries:
            phase_entries = self._resolve_single_series_phase_entries(target_series)

        if len(phase_entries) < 2:
            return FourDPhasesResponse(seriesId=series_id, isFourDSeries=False, fourDPhaseCount=0, fourDPhases=[])

        phase_items = self._build_phase_items(phase_entries)
        return FourDPhasesResponse(
            seriesId=series_id,
            isFourDSeries=True,
            fourDPhaseCount=len(phase_items),
            fourDPhases=phase_items,
        )

    def apply_four_d_metadata(
        self,
        summaries: list[SeriesSummary],
        series_records: list[SeriesRecord],
    ) -> None:
        if not summaries or not series_records:
            return

        summaries_by_id = {summary.series_id: summary for summary in summaries}
        self._apply_multi_series_groups(summaries_by_id, series_records)
        self._apply_single_series_groups(summaries_by_id, series_records)

    def _apply_multi_series_groups(
        self,
        summaries_by_id: dict[str, SeriesSummary],
        series_records: list[SeriesRecord],
    ) -> None:
        candidates_by_key: dict[tuple[Any, ...], list[PhaseCandidate]] = defaultdict(list)
        for series in series_records:
            phase = self._detect_series_phase(series)
            if phase is None:
                continue
            group_key = self._build_multi_series_group_key(series)
            candidates_by_key[group_key].append(PhaseCandidate(series=series, phase=phase))

        for candidates in candidates_by_key.values():
            if len(candidates) < 2:
                continue
            if len({candidate.phase.sort_value for candidate in candidates}) < 2:
                continue

            ordered_candidates = sorted(candidates, key=lambda item: (item.phase.sort_value, item.series.series_id))
            phase_items = self._build_phase_items(self._phase_entries_from_candidates(ordered_candidates))
            if len(phase_items) < 2:
                continue

            for candidate in ordered_candidates:
                summary = summaries_by_id.get(candidate.series.series_id)
                if summary is None:
                    continue
                summary.is_four_d_series = True
                summary.four_d_phase_count = len(phase_items)
                summary.four_d_phases = phase_items

    def _apply_single_series_groups(
        self,
        summaries_by_id: dict[str, SeriesSummary],
        series_records: list[SeriesRecord],
    ) -> None:
        for series in series_records:
            summary = summaries_by_id.get(series.series_id)
            if summary is None or summary.is_four_d_series:
                continue

            grouped_instances = self._group_instances_by_dataset_phase(series)
            if len(grouped_instances) < 2:
                continue

            phase_entries = self._phase_entries_from_grouped_instances(series, grouped_instances)
            phase_items = self._build_phase_items(phase_entries)
            if len(phase_items) < 2:
                continue

            summary.is_four_d_series = True
            summary.four_d_phase_count = len(phase_items)
            summary.four_d_phases = phase_items

    def _resolve_multi_series_phase_entries(
        self,
        target_series: SeriesRecord,
        series_records: list[SeriesRecord],
    ) -> list[tuple[PhaseValue, str | None, list[InstanceRecord]]]:
        target_phase = self._detect_series_phase(target_series)
        if target_phase is None:
            return []

        target_group_key = self._build_multi_series_group_key(target_series)
        candidates: list[PhaseCandidate] = []
        for series in series_records:
            if self._build_multi_series_group_key(series) != target_group_key:
                continue
            phase = self._detect_series_phase(series)
            if phase is None:
                continue
            candidates.append(PhaseCandidate(series=series, phase=phase))

        if len(candidates) < 2 or len({candidate.phase.sort_value for candidate in candidates}) < 2:
            return []

        ordered_candidates = sorted(candidates, key=lambda item: (item.phase.sort_value, item.series.series_id))
        return self._phase_entries_from_candidates(ordered_candidates)

    def _resolve_single_series_phase_entries(
        self,
        series: SeriesRecord,
    ) -> list[tuple[PhaseValue, str | None, list[InstanceRecord]]]:
        grouped_instances = self._group_instances_by_dataset_phase(series)
        if len(grouped_instances) < 2:
            return []
        return self._phase_entries_from_grouped_instances(series, grouped_instances)

    @staticmethod
    def _phase_entries_from_candidates(
        candidates: list[PhaseCandidate],
    ) -> list[tuple[PhaseValue, str | None, list[InstanceRecord]]]:
        return [(candidate.phase, candidate.series.series_id, candidate.series.instances) for candidate in candidates]

    @staticmethod
    def _phase_entries_from_grouped_instances(
        series: SeriesRecord,
        grouped_instances: dict[float, tuple[PhaseValue, list[InstanceRecord]]],
    ) -> list[tuple[PhaseValue, str | None, list[InstanceRecord]]]:
        return [
            (phase, series.series_id, instances)
            for phase, instances in sorted(grouped_instances.values(), key=lambda item: item[0].sort_value)
        ]

    def _build_phase_items(
        self,
        phase_entries: list[tuple[PhaseValue, str | None, list[InstanceRecord]]],
    ) -> list[FourDPhaseItem]:
        phase_items: list[FourDPhaseItem] = []
        for phase_index, (phase, series_id, instances) in enumerate(phase_entries):
            viewport_images, status = self._build_phase_viewport_images(instances)
            phase_items.append(
                FourDPhaseItem(
                    phaseIndex=phase_index,
                    label=self._format_phase_label(phase),
                    seriesId=series_id,
                    imageSrc=viewport_images.get(MPR_AXIAL, ""),
                    viewportImages=viewport_images,
                    status=status,
                )
            )
        return phase_items

    def _group_instances_by_dataset_phase(
        self,
        series: SeriesRecord,
    ) -> dict[float, tuple[PhaseValue, list[InstanceRecord]]]:
        grouped: dict[float, tuple[PhaseValue, list[InstanceRecord]]] = {}
        for instance in series.instances:
            header = self._read_header(instance.path)
            if header is None:
                continue
            phase = self._extract_phase_from_dataset(header)
            if phase is None:
                continue

            existing = grouped.get(phase.sort_value)
            if existing is None:
                grouped[phase.sort_value] = (phase, [instance])
            else:
                existing[1].append(instance)
        return grouped

    def _detect_series_phase(self, series: SeriesRecord) -> PhaseValue | None:
        for text in (series.series_description, self._relative_parent_text(series)):
            phase = self._extract_phase_from_text(text)
            if phase is not None:
                return phase

        first_instance = series.instances[0] if series.instances else None
        if first_instance is None:
            return None
        first_header = self._read_header(first_instance.path)
        if first_header is None:
            return None
        return self._extract_phase_from_dataset(first_header)

    def _build_multi_series_group_key(self, series: SeriesRecord) -> tuple[Any, ...]:
        first = series.instances[0] if series.instances else None
        description_key = self._normalize_phase_group_text(series.series_description)
        folder_key = self._normalize_phase_group_text(self._relative_parent_text(series))
        group_label_key = description_key or folder_key
        return (
            series.study_instance_uid or series.folder_path,
            series.modality or "",
            len(series.instances),
            first.rows if first is not None else None,
            first.columns if first is not None else None,
            group_label_key,
        )

    @staticmethod
    def _extract_phase_from_text(value: str | None) -> PhaseValue | None:
        text = str(value or "")
        if not text.strip():
            return None

        for pattern, source, suffix in (
            (_PERCENT_PHASE_RE, "text-percent", "%"),
            (_NAMED_PHASE_RE, "text-phase", ""),
            (_SHORT_PHASE_RE, "text-short-phase", ""),
        ):
            match = pattern.search(text)
            if not match:
                continue
            numeric_value = FourDService._coerce_number(match.group(1))
            if numeric_value is None:
                continue
            label_value = FourDService._format_numeric_label_value(numeric_value, suffix=suffix)
            return PhaseValue(sort_value=numeric_value, label_value=label_value, source=source)
        return None

    @staticmethod
    def _extract_phase_from_dataset(dataset: Dataset) -> PhaseValue | None:
        for keyword, suffix in (
            ("NominalPercentageOfRespiratoryPhase", "%"),
            ("ActualRespiratoryPhase", "%"),
            ("RespiratoryPhase", "%"),
            ("TemporalPositionIdentifier", ""),
            ("TriggerTime", "ms"),
            ("NominalRespiratoryTriggerDelayTime", "ms"),
            ("RespiratoryTriggerDelayThreshold", "ms"),
        ):
            value = getattr(dataset, keyword, None)
            numeric_value = FourDService._coerce_number(value)
            if numeric_value is None:
                continue
            return PhaseValue(
                sort_value=numeric_value,
                label_value=FourDService._format_numeric_label_value(numeric_value, suffix=suffix),
                source=keyword,
            )
        return None

    @staticmethod
    def _coerce_number(value: object) -> float | None:
        if value is None:
            return None
        if isinstance(value, MultiValue):
            if not value:
                return None
            value = value[0]
        elif isinstance(value, (list, tuple)):
            if not value:
                return None
            value = value[0]

        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            match = _FIRST_NUMBER_RE.search(str(value))
            if not match:
                return None
            try:
                numeric_value = float(match.group(0))
            except ValueError:
                return None

        if not np.isfinite(numeric_value):
            return None
        return numeric_value

    @staticmethod
    def _format_numeric_label_value(value: float, *, suffix: str) -> str:
        if abs(value - round(value)) < 1e-6:
            text = str(int(round(value)))
        else:
            text = f"{value:.2f}".rstrip("0").rstrip(".")

        if suffix == "%":
            return f"{text}%"
        if suffix == "ms":
            return f"{text}ms"
        return text

    @staticmethod
    def _format_phase_label(phase: PhaseValue) -> str:
        label_value = phase.label_value
        if label_value.endswith("%") or label_value.endswith("ms"):
            return f"Phase {label_value}"

        numeric_value = FourDService._coerce_number(label_value)
        if numeric_value is not None and abs(numeric_value - round(numeric_value)) < 1e-6:
            return f"Phase {int(round(numeric_value)):02d}"
        return f"Phase {label_value}"

    @staticmethod
    def _normalize_phase_group_text(value: str | None) -> str:
        text = str(value or "").lower()
        if not text.strip():
            return ""
        text = _PERCENT_PHASE_RE.sub(" ", text)
        text = _NAMED_PHASE_RE.sub(" ", text)
        text = _SHORT_PHASE_RE.sub(" ", text)
        text = re.sub(r"[^a-z0-9]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _relative_parent_text(series: SeriesRecord) -> str:
        first_instance = series.instances[0] if series.instances else None
        if first_instance is None:
            return ""

        parent = first_instance.path.parent
        try:
            relative = parent.relative_to(Path(series.folder_path))
            return " ".join(part for part in relative.parts if part not in ("", "."))
        except ValueError:
            return parent.name

    @staticmethod
    def _read_header(path: Path) -> Dataset | None:
        try:
            return pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        except Exception as exc:
            logger.debug("failed to read DICOM header for 4D phase detection path=%s error=%s", path, exc)
            return None

    def _build_phase_viewport_images(self, instances: list[InstanceRecord]) -> tuple[dict[str, str], str]:
        if not instances:
            return {}, "pending"

        try:
            volume = self._build_preview_volume(instances)
            volume_min, volume_max = self._resolve_window_range(volume)
            depth, height, width = volume.shape
            axial = volume[depth // 2, :, :]
            coronal = volume[:, height // 2, :]
            sagittal = volume[:, :, width // 2]
            viewport_images = {
                MPR_AXIAL: self._encode_plane_data_url(axial, volume_min, volume_max),
                MPR_CORONAL: self._encode_plane_data_url(coronal, volume_min, volume_max),
                MPR_SAGITTAL: self._encode_plane_data_url(sagittal, volume_min, volume_max),
            }
            return viewport_images, "ready"
        except Exception as exc:
            logger.warning("failed to build 4D phase preview images error=%s", exc)
            return {}, "error"

    def _build_preview_volume(self, instances: list[InstanceRecord]) -> np.ndarray:
        selected_instances = self._select_preview_instances(instances)
        slice_entries: list[tuple[np.ndarray, np.ndarray | None, np.ndarray | None]] = []
        first_shape: tuple[int, int] | None = None

        for instance in selected_instances:
            cache_key = instance.sop_instance_uid or instance.path.resolve().as_posix()
            cached = dicom_cache.get(cache_key, instance.path)
            pixels = np.asarray(cached.source_pixels, dtype=np.float32)
            if pixels.ndim != 2:
                raise ValueError("4D preview expects 2D source slices")
            if first_shape is None:
                first_shape = pixels.shape
            elif pixels.shape != first_shape:
                raise ValueError("4D preview requires consistent slice dimensions")

            dataset = cached.dataset
            slice_entries.append(
                (
                    pixels,
                    get_dataset_orientation(dataset),
                    get_dataset_position(dataset),
                )
            )

        if not slice_entries:
            raise ValueError("4D preview has no readable slices")
        return build_standardized_volume(slice_entries, logger=logger)

    @staticmethod
    def _select_preview_instances(instances: list[InstanceRecord]) -> list[InstanceRecord]:
        ordered_instances = sorted(instances, key=lambda item: item.instance_number)
        if len(ordered_instances) <= MAX_PREVIEW_SLICES:
            return ordered_instances

        indexes = np.linspace(0, len(ordered_instances) - 1, MAX_PREVIEW_SLICES)
        unique_indexes = sorted({int(round(index)) for index in indexes})
        return [ordered_instances[index] for index in unique_indexes]

    @staticmethod
    def _resolve_window_range(volume: np.ndarray) -> tuple[float, float]:
        finite_values = np.asarray(volume[np.isfinite(volume)], dtype=np.float32)
        if finite_values.size == 0:
            return 0.0, 1.0

        low = float(np.percentile(finite_values, 1.0))
        high = float(np.percentile(finite_values, 99.0))
        if high <= low:
            low = float(np.min(finite_values))
            high = float(np.max(finite_values))
        if high <= low:
            high = low + 1.0
        return low, high

    @staticmethod
    def _encode_plane_data_url(plane: np.ndarray, low: float, high: float) -> str:
        pixels = np.asarray(plane, dtype=np.float32)
        clipped = np.clip(pixels, low, high)
        normalized = ((clipped - low) * (255.0 / max(high - low, 1e-6))).astype(np.uint8)
        image = Image.fromarray(normalized)
        image = image.resize(PREVIEW_SIZE, Image.Resampling.LANCZOS)

        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"


four_d_service = FourDService()
