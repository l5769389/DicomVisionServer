from types import SimpleNamespace

from PIL import Image

from app.services import webp_3d_encoder


def test_auto_method_benchmarks_actual_frame_sample_once(monkeypatch) -> None:
    calls: list[tuple[tuple[int, int], int]] = []
    timings = iter((0.000, 0.010, 0.020, 0.026, 0.030, 0.050))

    monkeypatch.setattr(
        webp_3d_encoder,
        "get_settings",
        lambda: SimpleNamespace(normalized_three_d_final_webp_method="auto"),
    )
    monkeypatch.setattr(webp_3d_encoder, "perf_counter", lambda: next(timings))

    def fake_encode(image: Image.Image, *, method: int) -> bytes:
        calls.append((image.size, method))
        return bytes({0: 90, 1: 70, 2: 50}[method])

    monkeypatch.setattr(webp_3d_encoder, "_encode_lossless_webp", fake_encode)
    webp_3d_encoder.reset_3d_final_webp_method_selection()
    image = Image.new("RGB", (1024, 512), "black")

    assert webp_3d_encoder.resolve_3d_final_webp_method(image) == 1
    assert webp_3d_encoder.resolve_3d_final_webp_method(image) == 1
    assert calls == [((256, 128), 0), ((256, 128), 1), ((256, 128), 2)]


def test_configured_method_skips_auto_calibration(monkeypatch) -> None:
    monkeypatch.setattr(
        webp_3d_encoder,
        "get_settings",
        lambda: SimpleNamespace(normalized_three_d_final_webp_method=1),
    )
    monkeypatch.setattr(
        webp_3d_encoder,
        "_select_fastest_method",
        lambda image: (_ for _ in ()).throw(AssertionError("auto calibration should not run")),
    )

    assert webp_3d_encoder.resolve_3d_final_webp_method(Image.new("RGB", (8, 8))) == 1


def test_lossless_encoder_uses_selected_method(monkeypatch) -> None:
    monkeypatch.setattr(webp_3d_encoder, "resolve_3d_final_webp_method", lambda image: 1)
    monkeypatch.setattr(
        webp_3d_encoder,
        "_encode_lossless_webp",
        lambda image, *, method: f"method-{method}".encode(),
    )

    assert webp_3d_encoder.encode_lossless_3d_webp(Image.new("RGB", (8, 8))) == b"method-1"
