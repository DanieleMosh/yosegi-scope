"""Tests for the survey module: sample detection and scan-grid planning.

All inputs are synthetic NumPy arrays / tiny PNGs -- no microscope, no libvips.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import PIL.Image
import pytest

from yosegi.survey import (
    BBox,
    Region,
    ScanPlan,
    SurveyError,
    TissueMask,
    detect_sample_bbox,
    detect_sample_regions,
    plan_survey,
    plan_tile_grid,
)


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


def test_two_blobs_yield_largest_bbox() -> None:
    """Two disconnected blobs: the bbox snaps to the larger one.

    Old behaviour was to return the union of all surviving components, but on
    real overviews that lets a single scattered speck of camera noise near an
    empty corner inflate the bbox to cover the whole image. The current policy
    is to return the bbox of the *largest* component, which keeps detection
    robust to noise; a multi-region slide can still be re-surveyed with a
    smaller ``min_area_frac`` if both blobs are important.
    """
    img = np.full((300, 400), 240, dtype=np.uint8)
    img[40:90, 50:120] = 40        # smaller blob (50x70 = 3500 px)
    img[200:260, 280:380] = 40     # larger blob (60x100 = 6000 px)
    bbox = detect_sample_bbox(img)
    # bbox must enclose the larger blob (with a few pixels of morphological-close slop)
    # and must NOT extend into the smaller blob's region.
    assert 270 <= bbox.x0 <= 290 and 370 <= bbox.x1 <= 390
    assert 190 <= bbox.y0 <= 210 and 250 <= bbox.y1 <= 270


def test_accepts_path_pil_and_array(tmp_path: Path) -> None:
    img = _blob_image(200, 200, y0=60, y1=140, x0=60, x1=140)
    path = tmp_path / "overview.png"
    PIL.Image.fromarray(img).save(path)

    from_arr = detect_sample_bbox(img)
    from_pil = detect_sample_bbox(PIL.Image.fromarray(img))
    from_path = detect_sample_bbox(path)
    # JPEG would smear edges; PNG round-trips losslessly so all three must agree.
    assert from_arr == from_pil == from_path


def test_ignores_canvas_black_gaps_in_overview() -> None:
    """Regression: a gappy stitched overview (RGB 0,0,0 between tiles) must
    not be classified as dark tissue.

    On the real scope, an overview with step > tile-size produces a canvas with
    large black gutters. A previous version's Otsu pass treated those gutters
    as the darkest region (= tissue) and inflated the bbox to cover everything.
    Here we paste a small tissue blob into a mostly-black canvas and assert the
    bbox snaps to the blob, not the canvas.
    """
    canvas = np.zeros((400, 400, 3), dtype=np.uint8)  # mostly canvas-black
    # One textured "tissue" tile at (200, 200) -- bright/varied, not pure black.
    rng = np.random.default_rng(seed=42)
    tile = rng.integers(60, 180, size=(60, 60, 3), dtype=np.uint8)
    canvas[200:260, 200:260] = tile
    bbox = detect_sample_bbox(canvas, min_area_frac=0.0001)
    # Bbox should snap to the tile within a few pixels of morphological close.
    assert 195 <= bbox.x0 <= 205 and 195 <= bbox.y0 <= 205
    assert 255 <= bbox.x1 <= 268 and 255 <= bbox.y1 <= 268


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
    # 200 * (1 - 0.2) = 160 step.
    # cols: ceil((1000 - 200) / 160) + 1 = ceil(800/160) + 1 = 6  (covers [0, 1000])
    # rows: ceil((800 - 200) / 160) + 1 = ceil(600/160) + 1 = 5
    assert plan.step_x == 160 and plan.step_y == 160
    assert (plan.rows, plan.cols) == (5, 6)
    assert plan.positions[0] == (5000, 6000)
    # Last col anchor at c=5: 5000 + 5*160 = 5800; tile right edge: 5800+200 = 6000.
    assert plan.positions[1] == (5160, 6000)
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


def test_plan_grid_rotated_csm_produces_image_space_overlap() -> None:
    """Regression: a rotated CSM must produce stage steps that overlap *in image space*.

    With a 90-deg rotated CSM the image's x-axis maps mostly to stage-y, so the
    per-tile stage motion between camera-adjacent tiles is mostly along stage-y,
    not stage-x. A previous version stepped along stage axes directly and
    produced zero-overlap (disconnected) tiles on the real scope.
    """
    import numpy as np

    csm = [[0.01, -4.4], [-4.37, 0.0]]
    tw, th = 832, 624
    # Bbox big enough that we'll plan a real 3x3 grid (3 tiles needed each way).
    plan = plan_tile_grid(
        BBox(0, 0, 3 * tw, 3 * th),
        overview_origin_stage=(0, 0),
        overview_csm=csm,
        tile_size_px=(tw, th),
        overlap=0.2,
    )
    assert plan.rows >= 2 and plan.cols >= 2
    # Project the actual planned stage positions back into image space using the
    # inverse CSM. Adjacent (camera-) col tiles should differ by ~tile_w * (1-overlap)
    # in image-x and ~0 in image-y (NOT the other way round).
    csm_arr = np.asarray(csm, dtype=float)
    csm_inv = np.linalg.inv(csm_arr)
    p0 = np.array(plan.positions[0])  # row 0, col 0
    p1 = np.array(plan.positions[1])  # row 0, col 1 (snake order: same row, next col)
    delta_px = csm_inv @ (p1 - p0)
    # Expected step in image-x is ~tw * (1 - 0.2) = 665.6 pixels
    expected_step = tw * (1.0 - 0.2)
    assert abs(delta_px[0]) > 0.5 * expected_step, (
        f"camera-col step has too little image-x motion: {delta_px[0]:.1f} px "
        f"(expected ~{expected_step:.0f})"
    )
    assert abs(delta_px[1]) < 0.2 * th, (
        f"camera-col step should not move much in image-y: got {delta_px[1]:.1f} px"
    )


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


# --- detect_sample_regions --------------------------------------------------


def test_detect_regions_finds_multiple_blobs() -> None:
    """Two separated tissue blobs must yield two Regions, not one union bbox."""
    img = np.full((300, 400), 240, dtype=np.uint8)
    img[40:90, 50:120] = 40      # blob A (top-left)
    img[200:260, 280:380] = 40   # blob B (bottom-right)
    tissue = detect_sample_regions(img, min_area_frac=0.002)
    assert isinstance(tissue, TissueMask)
    assert len(tissue.regions) == 2
    # Regions are largest-first; the union bbox spans both blobs.
    assert all(isinstance(r, Region) for r in tissue.regions)
    assert tissue.bbox.x0 <= 50 and tissue.bbox.x1 >= 380
    assert tissue.bbox.y0 <= 40 and tissue.bbox.y1 >= 260
    # The gap between the blobs is background (mask False there).
    assert not tissue.mask[150, 200]


def test_detect_regions_max_regions_caps_count() -> None:
    img = np.full((300, 400), 240, dtype=np.uint8)
    img[40:90, 50:120] = 40      # smaller
    img[200:260, 280:390] = 40   # larger
    tissue = detect_sample_regions(img, min_area_frac=0.002, max_regions=1)
    assert len(tissue.regions) == 1
    # The single kept region is the larger blob (bottom-right).
    kept = tissue.regions[0].bbox
    assert kept.x0 >= 270 and kept.y0 >= 190


def test_detect_regions_saturation_cue_finds_colored_tissue() -> None:
    """A coloured blob on a bright grey background: the saturation cue must catch
    it even though its luminance is close to the background (intensity Otsu alone
    would struggle)."""
    # Background: bright, near-grey (low saturation).
    img = np.full((200, 200, 3), 220, dtype=np.uint8)
    # Blob: strongly coloured (high saturation) but similar brightness.
    img[70:130, 70:130] = (200, 60, 60)  # saturated red
    tissue = detect_sample_regions(img, min_area_frac=0.002)
    assert len(tissue.regions) >= 1
    b = tissue.regions[0].bbox
    # Region should snap to the coloured blob (a few px of morphological slop).
    assert 60 <= b.x0 <= 75 and 125 <= b.x1 <= 140
    assert 60 <= b.y0 <= 75 and 125 <= b.y1 <= 140


def test_detect_regions_rejects_blank_slide() -> None:
    blank = np.full((200, 200), 250, dtype=np.uint8)
    with pytest.raises(SurveyError, match="no sample detected"):
        detect_sample_regions(blank)


def test_detect_sample_bbox_still_returns_largest_region() -> None:
    """The backward-compat wrapper returns the largest region's bbox."""
    img = np.full((300, 400), 240, dtype=np.uint8)
    img[40:90, 50:120] = 40      # smaller
    img[200:260, 280:390] = 40   # larger
    bbox = detect_sample_bbox(img, min_area_frac=0.002)
    # Largest blob is at [200:260, 280:390]; morphological close (r=5) may grow
    # the bbox by a few px on each side (and clamp to the 400px image edge).
    assert 270 <= bbox.x0 <= 290 and 385 <= bbox.x1 <= 400
    assert 190 <= bbox.y0 <= 210 and 255 <= bbox.y1 <= 270


# --- plan_survey (tissue-gated) ---------------------------------------------


def _two_blob_tissue() -> TissueMask:
    """A 300x400 mask with two well-separated tissue blocks."""
    mask = np.zeros((300, 400), dtype=bool)
    mask[0:100, 0:100] = True      # region 1
    mask[0:100, 300:400] = True    # region 2 (far right)
    regions = [
        Region(bbox=BBox(0, 0, 100, 100), area=10000, centroid=(50.0, 50.0)),
        Region(bbox=BBox(300, 0, 400, 100), area=10000, centroid=(50.0, 350.0)),
    ]
    return TissueMask(mask=mask, regions=regions, bbox=BBox(0, 0, 400, 100))


def test_plan_survey_skips_empty_tiles() -> None:
    """Tiles whose centre lands in the gap between two regions are skipped, so a
    tissue-gated plan has fewer tiles than a full-bbox grid over the same span."""
    tissue = _two_blob_tissue()
    gated = plan_survey(
        tissue,
        overview_origin_stage=(0, 0),
        overview_csm=_IDENTITY_CSM,
        tile_size_px=(50, 50),
        overlap=0.0,
    )
    dense = plan_tile_grid(
        tissue.bbox,
        overview_origin_stage=(0, 0),
        overview_csm=_IDENTITY_CSM,
        tile_size_px=(50, 50),
        overlap=0.0,
    )
    # Same raster extent...
    assert (gated.rows, gated.cols) == (dense.rows, dense.cols)
    # ...but the gated plan drops the empty middle columns.
    assert gated.tile_count < dense.tile_count
    assert gated.tile_count == len(gated.positions) == len(gated.rowcols)
    # Every kept tile's centre stage position is under one of the two regions
    # (identity CSM: stage == pixel). Tile centre offset is +25 px from origin.
    for (sx, _sy) in gated.positions:
        assert sx < 125 or sx >= 275  # left region OR right region, not the gap


def test_plan_survey_rowcols_are_dense_and_snake_ordered() -> None:
    tissue = _two_blob_tissue()
    plan = plan_survey(
        tissue,
        overview_origin_stage=(0, 0),
        overview_csm=_IDENTITY_CSM,
        tile_size_px=(50, 50),
        overlap=0.0,
    )
    # Dense renumbering: rows start at 0 and are contiguous; each row's cols
    # start at 0 with no gaps.
    rows_seen = sorted({r for r, _ in plan.rowcols})
    assert rows_seen == list(range(len(rows_seen)))
    from collections import defaultdict

    cols_by_row: dict[int, list[int]] = defaultdict(list)
    for r, c in plan.rowcols:
        cols_by_row[r].append(c)
    for cols in cols_by_row.values():
        assert cols == list(range(len(cols))) or cols == list(range(len(cols) - 1, -1, -1))


def test_plan_survey_falls_back_when_tissue_smaller_than_a_tile() -> None:
    """If every tile centre misses a tiny tissue speck, plan the whole bbox rather
    than return an empty scan."""
    mask = np.zeros((60, 60), dtype=bool)
    mask[0:3, 0:3] = True  # speck in the corner; no 40px-tile centre lands on it
    tissue = TissueMask(
        mask=mask,
        regions=[Region(bbox=BBox(0, 0, 3, 3), area=9, centroid=(1.0, 1.0))],
        bbox=BBox(0, 0, 3, 3),
    )
    plan = plan_survey(
        tissue,
        overview_origin_stage=(0, 0),
        overview_csm=_IDENTITY_CSM,
        tile_size_px=(40, 40),
        overlap=0.0,
    )
    assert plan.tile_count >= 1


def test_plan_survey_rejects_no_regions() -> None:
    empty = TissueMask(mask=np.zeros((10, 10), dtype=bool), regions=[], bbox=BBox(0, 0, 0, 0))
    with pytest.raises(SurveyError, match="no detected tissue"):
        plan_survey(
            empty,
            overview_origin_stage=(0, 0),
            overview_csm=_IDENTITY_CSM,
            tile_size_px=(10, 10),
        )
