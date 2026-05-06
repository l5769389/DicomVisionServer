from app.sockets.runtime import RenderRequest, ViewSocketHub


def test_merge_render_request_keeps_full_quality_when_preview_arrives_later() -> None:
    merged = ViewSocketHub._merge_render_request(
        RenderRequest(image_format="png", fast_preview=False, target_sids=("sid-1",)),
        RenderRequest(image_format="jpeg", fast_preview=True, target_sids=("sid-2",)),
    )

    assert merged.image_format == "png"
    assert merged.fast_preview is False
    assert merged.target_sids == ("sid-1", "sid-2")


def test_merge_render_request_promotes_pending_preview_to_full_quality() -> None:
    merged = ViewSocketHub._merge_render_request(
        RenderRequest(image_format="jpeg", fast_preview=True, target_sids=("sid-1",)),
        RenderRequest(image_format="png", fast_preview=False, target_sids=("sid-1", "sid-2")),
    )

    assert merged.image_format == "png"
    assert merged.fast_preview is False
    assert merged.target_sids == ("sid-1", "sid-2")


def test_merge_render_request_keeps_broadcast_target_when_either_request_broadcasts() -> None:
    merged = ViewSocketHub._merge_render_request(
        RenderRequest(image_format="jpeg", fast_preview=True, target_sids=None),
        RenderRequest(image_format="png", fast_preview=False, target_sids=("sid-1",)),
    )

    assert merged.image_format == "png"
    assert merged.fast_preview is False
    assert merged.target_sids is None
