from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi.testclient import TestClient
from pydicom.dataset import Dataset

from app.api.routes import pacs as pacs_route
from app.main import fastapi_app
from app.schemas.pacs import (
    PacsDimseProfile,
    PacsDimseSeriesQueryRequest,
    PacsDimseStudyQueryRequest,
    PacsDicomwebTestResponse,
)
from app.services.pacs_dimse_service import PacsDimseService


def _status(code: int) -> Dataset:
    dataset = Dataset()
    dataset.Status = code
    return dataset


def _study_record(study_uid: str, patient_id: str) -> Dataset:
    dataset = Dataset()
    dataset.StudyInstanceUID = study_uid
    dataset.PatientName = "Patient^Demo"
    dataset.PatientID = patient_id
    dataset.StudyDate = "20260522"
    dataset.ModalitiesInStudy = ["CT", "MR"]
    dataset.NumberOfStudyRelatedSeries = 2
    dataset.NumberOfStudyRelatedInstances = 42
    return dataset


def _series_record(series_uid: str) -> Dataset:
    dataset = Dataset()
    dataset.StudyInstanceUID = "1.2.3"
    dataset.SeriesInstanceUID = series_uid
    dataset.SeriesNumber = 7
    dataset.Modality = "CT"
    dataset.SeriesDescription = "Portal Venous"
    dataset.BodyPartExamined = "ABDOMEN"
    dataset.NumberOfSeriesRelatedInstances = 99
    return dataset


def _profile() -> PacsDimseProfile:
    return PacsDimseProfile(
        id="dimse-local",
        name="DIMSE Local",
        host="127.0.0.1",
        port=104,
        calledAeTitle="ORTHANC",
        clientAeTitle="DICOMVISION",
    )


@dataclass
class FakeAssociation:
    responses: list[tuple[Dataset, Dataset | None]]
    echo_status: Dataset | None = None
    is_established: bool = True
    released: bool = False
    query_dataset: Dataset | None = None
    query_context: Any = None

    def send_c_echo(self) -> Dataset:
        return self.echo_status or _status(0x0000)

    def send_c_find(self, dataset: Dataset, context: Any) -> list[tuple[Dataset, Dataset | None]]:
        self.query_dataset = dataset
        self.query_context = context
        return self.responses

    def release(self) -> None:
        self.released = True


class FakeAE:
    def __init__(self, association: FakeAssociation) -> None:
        self.association = association
        self.contexts: list[Any] = []
        self.associate_args: tuple[str, int, str] | None = None

    def add_requested_context(self, context: Any) -> None:
        self.contexts.append(context)

    def associate(self, host: str, port: int, *, ae_title: str) -> FakeAssociation:
        self.associate_args = (host, port, ae_title)
        return self.association


def test_dimse_echo_uses_ae_titles_and_host() -> None:
    association = FakeAssociation(responses=[])
    fake_ae = FakeAE(association)
    service = PacsDimseService(ae_factory=lambda ae_title: fake_ae)

    result = service.test_connection(_profile())

    assert result.ok is True
    assert result.status_code == 0
    assert fake_ae.associate_args == ("127.0.0.1", 104, "ORTHANC")
    assert association.released is True


def test_dimse_study_query_maps_filters_and_results() -> None:
    association = FakeAssociation(
        responses=[
            (_status(0xFF00), _study_record("1.2.3", "P001")),
            (_status(0xFF00), _study_record("1.2.4", "P002")),
            (_status(0x0000), None),
        ]
    )
    service = PacsDimseService(ae_factory=lambda ae_title: FakeAE(association))

    response = service.query_studies(
        PacsDimseStudyQueryRequest(
            profile=_profile(),
            studyInstanceUid="1.2.3",
            patientName="Patient*",
            modality="CT",
            studyDateFrom="2026-05-01",
            studyDateTo="2026-05-22",
            limit=1,
            offset=1,
        )
    )

    query_dataset = association.query_dataset
    assert query_dataset is not None
    assert query_dataset.QueryRetrieveLevel == "STUDY"
    assert query_dataset.StudyInstanceUID == "1.2.3"
    assert query_dataset.PatientName == "Patient*"
    assert query_dataset.ModalitiesInStudy == "CT"
    assert query_dataset.StudyDate == "20260501-20260522"
    assert response.items[0].study_instance_uid == "1.2.4"
    assert response.items[0].modalities_in_study == ["CT", "MR"]
    assert response.items[0].number_of_study_related_instances == 42


def test_dimse_series_query_maps_filters_and_results() -> None:
    association = FakeAssociation(
        responses=[
            (_status(0xFF00), _series_record("4.5.6")),
            (_status(0x0000), None),
        ]
    )
    service = PacsDimseService(ae_factory=lambda ae_title: FakeAE(association))

    response = service.query_series(
        PacsDimseSeriesQueryRequest(
            profile=_profile(),
            studyInstanceUid="1.2.3",
            seriesInstanceUid="4.5.6",
            seriesDescription="Portal",
            bodyPartExamined="ABDOMEN",
            limit=10,
        )
    )

    query_dataset = association.query_dataset
    assert query_dataset is not None
    assert query_dataset.QueryRetrieveLevel == "SERIES"
    assert query_dataset.StudyInstanceUID == "1.2.3"
    assert query_dataset.SeriesInstanceUID == "4.5.6"
    assert query_dataset.SeriesDescription == "Portal"
    assert query_dataset.BodyPartExamined == "ABDOMEN"
    assert response.items[0].series_instance_uid == "4.5.6"
    assert response.items[0].number_of_series_related_instances == 99


def test_dimse_test_connection_endpoint_uses_service(monkeypatch) -> None:
    def fake_test_connection(profile: PacsDimseProfile) -> PacsDicomwebTestResponse:
        assert profile.host == "127.0.0.1"
        assert profile.called_ae_title == "ORTHANC"
        return PacsDicomwebTestResponse(ok=True, statusCode=0, message="ok")

    monkeypatch.setattr(pacs_route.pacs_dimse_service, "test_connection", fake_test_connection)

    client = TestClient(fastapi_app)
    response = client.post(
        "/api/v1/pacs/dimse/test",
        json={
            "profile": {
                "id": "p1",
                "name": "DIMSE",
                "host": "127.0.0.1",
                "port": 104,
                "calledAeTitle": "ORTHANC",
                "clientAeTitle": "DICOMVISION",
            }
        },
    )

    assert response.status_code == 200
    assert response.json() == {"ok": True, "statusCode": 0, "message": "ok"}
