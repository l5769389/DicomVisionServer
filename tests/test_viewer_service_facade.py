from types import SimpleNamespace

import app.services.viewer_service as viewer_service_module
from app.services.viewer.interaction import ViewerInteractionMixin
from app.services.viewer.mpr import ViewerMprMixin
from app.services.viewer.volume import ViewerVolumeMixin
from app.services.viewer_service import ViewerService, viewer_service


def test_viewer_service_keeps_the_legacy_facade_and_singleton() -> None:
    assert isinstance(viewer_service, ViewerService)
    assert issubclass(ViewerService, ViewerInteractionMixin)
    assert issubclass(ViewerService, ViewerMprMixin)
    assert issubclass(ViewerService, ViewerVolumeMixin)
    assert callable(viewer_service._render_mpr_view)
    assert callable(viewer_service._render_3d_view)
    assert callable(viewer_service._handle_measurement)


def test_domain_mixins_resolve_replaced_facade_dependencies(monkeypatch) -> None:
    replacement = SimpleNamespace(get=lambda *_args, **_kwargs: None)

    monkeypatch.setattr(viewer_service_module, "series_registry", replacement)

    assert viewer_service_module.compat.series_registry is replacement


def test_viewer_service_facade_remains_small() -> None:
    facade_lines = viewer_service_module.__file__
    assert facade_lines is not None
    with open(facade_lines, encoding="utf-8") as source_file:
        assert sum(1 for _line in source_file) < 300
