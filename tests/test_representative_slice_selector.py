import numpy as np

from app.services.representative_slice_selector import (
    build_representative_sample_indexes,
    score_representative_pixels,
)


def test_representative_sample_indexes_are_bounded_and_include_edges() -> None:
    indexes = build_representative_sample_indexes(200, sample_limit=10)

    assert len(indexes) == 10
    assert indexes[0] == 0
    assert indexes[-1] == 199
    assert indexes == sorted(set(indexes))


def test_representative_score_prefers_content_over_blank_background() -> None:
    blank = np.full((16, 16), -1000.0, dtype=np.float32)
    content = blank.copy()
    content[4:12, 4:12] = 180.0

    assert score_representative_pixels(content) > score_representative_pixels(blank)
