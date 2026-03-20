from pydantic import BaseModel, Field


class DicomRenderRequest(BaseModel):
    dicom_dir: str = Field(..., description="Directory containing DICOM files")
    file_name: str | None = Field(default=None, description="Specific DICOM file name")
    index: int = Field(default=0, ge=0, description="File index when file_name is not set")
    image_format: str = Field(default="png", pattern="^(png|jpeg)$")
    window_center: float | None = Field(default=None)
    window_width: float | None = Field(default=None, gt=0)
    invert: bool = Field(default=False)


class DicomRenderResponse(BaseModel):
    file_path: str
    image_format: str
    image_base64: str
    content_type: str
    width: int
    height: int
    patient_id: str | None = None
    study_instance_uid: str | None = None
    series_instance_uid: str | None = None
    sop_instance_uid: str | None = None
