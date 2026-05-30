"""Tests for the survey module: sample detection and scan-grid planning.

All inputs are synthetic NumPy arrays / tiny PNGs -- no microscope, no libvips.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import PIL.Image
import pytest

from yosegi.survey import BBox, ScanPlan, SurveyError, detect_sample_bbox, plan_tile_grid


def _blob_image(
    h: int, w: int, y0: int, y1: int, x0: int, x1: int, *, bg: int = 240, fg: int = 40
) -> np.ndarray:
    """Background ``bg``, with a rectangular dark blob at ``[y0:y1, x0:x1]``."""
    img = np.full((h, w), bg, dtype=np.uint8)
    img[y0:y1, x0:x1] = fg
    return img


def test_detects_centred_blob_within_close_radius() -> None:
    img = _blob_image(200, 300, y0=50, y1=150, x0=80, x1=220)
    bbox = detect_sample_bbox(img, close_radius=5)
    # The morphological close dilates the mask by ~close_radius, so the bbox
    # can grow by up to that many pixels on each side -- never shrink past the blob.
    assert bbox.x0 <= 80 and bbox.x1 >= 220
    assert bbox.y0 <= 50 and bbox.y1 >= 150
    assert bbox.x0 >= 80 - 10 and bbox.x1 <= 220 + 10
    assert bbox.y0 >= 50 - 10 and bbox.y1 <= 150 + 10


def test_rejects_blank_slide_with_guidance() -> None:
    blank = np.full((200, 200), 250, dtype=np.uint8)
    with pytest.raises(SurveyError, match="no sample detected"):
        detect_sample_bbox(blank)


def test_rejects_flat_image() -> None:
    flat = np.full((100, 100), 128, dtype=np.uint8)
    with pytest.raises(SurveyError, match="no sample detected"):
        detect_sample_bbox(flat)


def test_ignores_tiny_speck_below_area_threshold() -> None:
    # A 2x2 speck in a 400x400 image is 4/160000 = 0.0025% -- well below 0.5% default.
    img = np.full((400, 400), 250, dtype=np.uint8)
    img[10:12, 10:12] = 0
    with pytest.raises(SurveyError):
        detect_sample_bbox(img)


def test_two_blobs_yield_union_bbox() -> None:
    img = np.full((300, 400), 240, dtype=np.uint8)
    img[40:90, 50:120] = 40   # top-left blob
    img[200:260, 280:360] = 40  # bottom-right blob
    bbox = detect_sample_bbox(img)
    # union must enclose both blobs
    assert bbox.x0 <= 50 and bbox.x1 >= 360
    assert bbox.y0 <= 40 and bbox.y1 >= 260


def test_accepts_path_pil_and_array(tmp_path: Path) -> None:
    img = _blob_image(200, 200, y0=60, y1=140, x0=60, x1=140)
    path = tmp_path / "overview.png"
    PIL.Image.fromarray(img).save(path)

    from_arr = detect_sample_bbox(img)
    from_pil = detect_sample_bbox(PIL.Image.fromarray(img))
    from_path = detect_sample_bbox(path)
    # JPEG would smear edges; PNG round-trips losslessly so all three must agree.
    assert from_arr == from_pil == from_path


def test_accepts_rgb_array() -> None:
    gray = _blob_image(200, 200, y0=50, y1=150, x0=50, x1=150)
    rgb = np.stack([gray, gray, gray], axis=-1)  # H x W x 3
    assert detect_sample_bbox(rgb) == detect_sample_bbox(gray)


def test_unreadable_path_raises(tmp_path: Path) -> None:
    with pytest.raises(SurveyError, match="could not read"):
        detect_sample_bbox(tmp_path / "does_not_exist.png")


def test_bbox_properties() -> None:
    b = BBox(x0=10, y0=20, x1=110, y1=80)
    assert b.width == 100 and b.height == 60 and not b.is_empty
    assert BBox(0, 0, 0, 10).is_empty


# --- plan_tile_grid ---------------------------------------------------------


_IDENTITY_CSM = [[1.0, 0.0], [0.0, 1.0]]


def test_plan_grid_identity_csm_snake_order() -> None:
    plan = plan_tile_grid(
        BBox(0, 0, 600, 400),
        overview_origin_stage=(0, 0),
        overview_csm=_IDENTITY_CSM,
        tile_size_px=(200, 200),
        overlap=0.0,
    )
    assert isinstance(plan, ScanPlan)
    assert (plan.rows, plan.cols) == (2, 3)
    assert plan.step_x == 200 and plan.step_y == 200
    assert plan.positions == [
        (0, 0), (200, 0), (400, 0),
        (400, 200), (200, 200), (0, 200),
    ]


def test_plan_grid_respects_overlap() -> None:
    plan = plan_tile_grid(
        BBox(0, 0, 1000, 800),
        overview_origin_stage=(5000, 6000),
        overview_csm=_IDENTITY_CSM,
        tile_size_px=(200, 200),
        overlap=0.2,
    )
    # 200 * (1 - 0.2) = 160 step. 1000/160 = 6.25 -> 7 cols. 800/160 = 5 -> 5 rows.
    assert plan.step_x == 160 and plan.step_y == 160
    assert (plan.rows, plan.cols) == (5, 7)
    assert plan.positions[0] == (5000, 6000)
    assert plan.bbox_stage == BBox(5000, 6000, 6000, 6800)


def test_plan_grid_handles_rotated_csm() -> None:
    """OpenFlexure-style CSM: ~90deg rotation + ~4.4 steps/px scaling.

    The pixel bbox is small enough to fit in one tile-stage-extent, so we expect
    a 1x1 grid anchored at the projected bbox corner.
    """
    csm = [[0.01, -4.4], [-4.37, 0.0]]
    plan = plan_tile_grid(
        BBox(0, 0, 832, 624),
        overview_origin_stage=(0, 0),
        overview_csm=csm,
        tile_size_px=(832, 624),
        overlap=0.2,
    )
    assert (plan.rows, plan.cols) == (1, 1)
    assert len(plan.positions) == 1


def test_plan_grid_rejects_empty_bbox() -> None:
    with pytest.raises(SurveyError, match="empty bounding box"):
        plan_tile_grid(
            BBox(10, 10, 10, 50),
            overview_origin_stage=(0, 0), overview_csm=_IDENTITY_CSM,
            tile_size_px=(100, 100),
        )


def test_plan_grid_rejects_bad_tile_size() -> None:
    with pytest.raises(SurveyError, match="tile_size_px"):
        plan_tile_grid(
            BBox(0, 0, 100, 100),
            overview_origin_stage=(0, 0), overview_csm=_IDENTITY_CSM,
            tile_size_px=(0, 100),
        )


def test_plan_grid_rejects_bad_overlap() -> None:
    with pytest.raises(SurveyError, match="overlap"):
        plan_tile_grid(
            BBox(0, 0, 100, 100),
            overview_origin_stage=(0, 0), overview_csm=_IDENTITY_CSM,
            tile_size_px=(50, 50), overlap=1.0,
        )


def test_plan_grid_rejects_wrong_shape_csm() -> None:
    with pytest.raises(SurveyError, match="2x2"):
        plan_tile_grid(
            BBox(0, 0, 100, 100),
            overview_origin_stage=(0, 0), overview_csm=[[1.0, 0.0, 0.0]],
            tile_size_px=(50, 50),
        )
