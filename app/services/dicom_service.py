import base64
import io
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image
from fastapi import HTTPException
from pydicom.dataset import FileDataset
from pydicom.multival import MultiValue

from app.schemas.dicom import DicomRenderRequest, DicomRenderResponse


class DicomService:
    supported_suffixes = {".dcm", ".dicom", ""}

    def render_from_request(self, payload: DicomRenderRequest) -> DicomRenderResponse:
        file_path = self._resolve_file_path(payload.dicom_dir, payload.file_name, payload.index)
        dataset = pydicom.dcmread(str(file_path))
        image = self._render_dataset_to_image(
            dataset=dataset,
            image_format=payload.image_format,
            window_center=payload.window_center,
            window_width=payload.window_width,
            invert=payload.invert,
        )

        width, height = image.size
        content_type = f"image/{payload.image_format}"
        image_base64 = self._image_to_base64(image, payload.image_format)

        return DicomRenderResponse(
            file_path=str(file_path),
            image_format=payload.image_format,
            image_base64=image_base64,
            content_type=content_type,
            width=width,
            height=height,
            patient_id=getattr(dataset, "PatientID", None),
            study_instance_uid=getattr(dataset, "StudyInstanceUID", None),
            series_instance_uid=getattr(dataset, "SeriesInstanceUID", None),
            sop_instance_uid=getattr(dataset, "SOPInstanceUID", None),
        )

    def _resolve_file_path(self, dicom_dir: str, file_name: str | None, index: int) -> Path:
        directory = Path(dicom_dir).expanduser().resolve()
        if not directory.exists() or not directory.is_dir():
            raise HTTPException(status_code=404, detail="DICOM directory not found")

        if file_name:
            file_path = directory / file_name
            if not file_path.exists():
                raise HTTPException(status_code=404, detail="DICOM file not found")
            return file_path

        files = sorted(
            path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in self.supported_suffixes
        )
        if not files:
            raise HTTPException(status_code=404, detail="No DICOM files found in directory")
        if index >= len(files):
            raise HTTPException(status_code=400, detail=f"Requested index {index} exceeds file count {len(files)}")
        return files[index]

    def _render_dataset_to_image(
        self,
        dataset: FileDataset,
        image_format: str,
        window_center: float | None,
        window_width: float | None,
        invert: bool,
    ) -> Image.Image:
        if "PixelData" not in dataset:
            raise HTTPException(status_code=400, detail="DICOM file does not contain pixel data")

        try:
            pixel_array = dataset.pixel_array.astype(np.float32)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to decode pixel data: {exc}") from exc

        if pixel_array.ndim == 3:
            pixel_array = pixel_array[0]

        pixel_array = self._apply_modality_lut(dataset, pixel_array)
        window_center = window_center if window_center is not None else self._get_first_number(getattr(dataset, "WindowCenter", None))
        window_width = window_width if window_width is not None else self._get_first_number(getattr(dataset, "WindowWidth", None))

        if window_center is not None and window_width is not None:
            lower = window_center - window_width / 2
            upper = window_center + window_width / 2
            pixel_array = np.clip(pixel_array, lower, upper)
        else:
            pixel_min = float(np.min(pixel_array))
            pixel_max = float(np.max(pixel_array))
            if pixel_max == pixel_min:
                pixel_array = np.zeros_like(pixel_array)
            else:
                pixel_array = (pixel_array - pixel_min) / (pixel_max - pixel_min)
                pixel_array = pixel_array * 255.0

        if window_center is not None and window_width is not None:
            pixel_array = (pixel_array - np.min(pixel_array)) / max(np.max(pixel_array) - np.min(pixel_array), 1e-6)
            pixel_array = pixel_array * 255.0

        if getattr(dataset, "PhotometricInterpretation", "") == "MONOCHROME1":
            invert = not invert
        if invert:
            pixel_array = 255.0 - pixel_array

        image = Image.fromarray(pixel_array.astype(np.uint8), mode="L")
        if image_format == "jpeg":
            return image.convert("L")
        return image

    @staticmethod
    def _apply_modality_lut(dataset: FileDataset, pixel_array: np.ndarray) -> np.ndarray:
        slope = float(getattr(dataset, "RescaleSlope", 1.0))
        intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
        return pixel_array * slope + intercept

    @staticmethod
    def _get_first_number(value: float | MultiValue | None) -> float | None:
        if value is None:
            return None
        if isinstance(value, MultiValue):
            return float(value[0])
        return float(value)

    @staticmethod
    def _image_to_base64(image: Image.Image, image_format: str) -> str:
        output = io.BytesIO()
        save_format = "JPEG" if image_format == "jpeg" else "PNG"
        image.save(output, format=save_format)
        return base64.b64encode(output.getvalue()).decode("utf-8")


dicom_service = DicomService()
