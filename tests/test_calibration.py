"""Calibration: validation, mapping accuracy, persistence, wizard."""

import json
import math

import pytest

from helpers import make_hand
from nocursor.calibration import (
    CalibrationError,
    CalibrationStore,
    CalibrationWizard,
    WizardState,
    default_calibration_path,
    region_from_corners,
    validate_region,
)
from nocursor.config import Config
from nocursor.mapping import CursorMapper, ViewRegion
from nocursor.processor import GestureProcessor

# Corners as seen in the mirrored preview: top-left, top-right, bottom-right, bottom-left.
GOOD = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.7), (0.2, 0.7)]


class TestRegionFromCorners:
    def test_valid_corners_give_the_expected_region(self):
        r = region_from_corners(GOOD)
        assert (r.left, r.top, r.right, r.bottom) == pytest.approx((0.2, 0.2, 0.8, 0.7))

    def test_slightly_skewed_corners_are_averaged(self):
        r = region_from_corners([(0.20, 0.20), (0.80, 0.22), (0.82, 0.70), (0.22, 0.72)])
        assert (r.left, r.right) == pytest.approx((0.21, 0.81))
        assert (r.top, r.bottom) == pytest.approx((0.21, 0.71))

    @pytest.mark.parametrize("points", [
        [], GOOD[:3], GOOD + [(0.5, 0.5)],
    ])
    def test_wrong_number_of_points_rejected(self, points):
        with pytest.raises(CalibrationError, match="exactly 4"):
            region_from_corners(points)

    def test_duplicate_corners_rejected(self):
        with pytest.raises(CalibrationError, match="too close"):
            region_from_corners([(0.2, 0.2), (0.2, 0.2), (0.8, 0.7), (0.2, 0.7)])

    def test_nearly_identical_corners_rejected(self):
        with pytest.raises(CalibrationError, match="too close"):
            region_from_corners([(0.5, 0.5), (0.51, 0.5), (0.51, 0.51), (0.5, 0.51)])

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -0.1, 1.1])
    def test_invalid_coordinates_rejected(self, bad):
        with pytest.raises(CalibrationError, match="outside|number"):
            region_from_corners([(bad, 0.2), (0.8, 0.2), (0.8, 0.7), (0.2, 0.7)])

    def test_degenerate_width_rejected(self):
        with pytest.raises(CalibrationError):
            region_from_corners([(0.40, 0.2), (0.45, 0.2), (0.45, 0.7), (0.40, 0.7)])

    def test_degenerate_height_rejected(self):
        with pytest.raises(CalibrationError):
            region_from_corners([(0.2, 0.40), (0.8, 0.40), (0.8, 0.45), (0.2, 0.45)])

    def test_region_too_short_but_with_distinct_corners_is_rejected_as_too_small(self):
        with pytest.raises(CalibrationError, match="too small"):
            region_from_corners([(0.2, 0.40), (0.8, 0.40), (0.8, 0.55), (0.2, 0.55)])

    def test_corners_in_the_wrong_order_rejected(self):
        # Captured right-to-left: looks like a mirrored region.
        with pytest.raises(CalibrationError, match="wrong order"):
            region_from_corners([GOOD[1], GOOD[0], GOOD[3], GOOD[2]])

    def test_non_rectangular_corners_rejected(self):
        with pytest.raises(CalibrationError, match="rectangular"):
            region_from_corners([(0.2, 0.2), (0.8, 0.2), (0.8, 0.7), (0.5, 0.7)])

    def test_validate_region_enforces_minimum_size(self):
        with pytest.raises(CalibrationError, match="too small"):
            validate_region(ViewRegion(0.4, 0.4, 0.5, 0.9))
        assert validate_region(ViewRegion(0.1, 0.1, 0.9, 0.9)).width == pytest.approx(0.8)

    @pytest.mark.parametrize("args", [
        (0.5, 0.1, 0.4, 0.9), (0.1, 0.5, 0.9, 0.4), (-0.1, 0.1, 0.9, 0.9), (0.1, 0.1, 1.5, 0.9),
        (math.nan, 0.1, 0.9, 0.9),
    ])
    def test_view_region_itself_rejects_nonsense(self, args):
        with pytest.raises(ValueError):
            ViewRegion(*args)


class TestCalibratedMapping:
    SCREEN = (1920, 1080)

    def mapper(self, region=None, **kw):
        region = region or region_from_corners(GOOD)
        return CursorMapper(*self.SCREEN, bounds=region, edge_inset=0, **kw)

    def test_region_corners_map_to_screen_corners(self):
        m = self.mapper()
        # The region is in mirrored view space, so raw camera x = 1 - view x.
        assert m.map(1 - 0.2, 0.2) == pytest.approx((0, 0))
        assert m.map(1 - 0.8, 0.2) == pytest.approx((1919, 0))
        assert m.map(1 - 0.8, 0.7) == pytest.approx((1919, 1079))
        assert m.map(1 - 0.2, 0.7) == pytest.approx((0, 1079))

    def test_centre_of_region_maps_to_centre_of_screen(self):
        x, y = self.mapper().map(1 - 0.5, 0.45)
        assert (x, y) == pytest.approx((959.5, 539.5))

    def test_mapping_is_linear_inside_the_region(self):
        m = self.mapper()
        x1, _ = m.map(1 - 0.35, 0.45)
        x2, _ = m.map(1 - 0.65, 0.45)
        assert x2 - x1 == pytest.approx(0.5 * 1919)  # 0.3 of a 0.6-wide region

    def test_left_and_right_are_not_reversed(self):
        """A fingertip on the left of the preview must move the pointer left on screen."""
        m = self.mapper()
        left_of_preview = m.map(1 - 0.3, 0.45)[0]  # appears left in the mirrored preview
        right_of_preview = m.map(1 - 0.7, 0.45)[0]
        assert left_of_preview < right_of_preview

    def test_matches_the_default_mapping_when_the_region_is_the_margin_region(self):
        default = CursorMapper(*self.SCREEN, margin=0.15)
        calibrated = CursorMapper(*self.SCREEN, bounds=ViewRegion.from_margin(0.15))
        for x, y in [(0.1, 0.1), (0.3, 0.6), (0.5, 0.5), (0.9, 0.95)]:
            assert calibrated.map(x, y) == pytest.approx(default.map(x, y))

    def test_points_outside_the_region_clamp_to_the_screen_edge(self):
        m = self.mapper()
        assert m.map(0.0, 0.0) == pytest.approx((1919, 0))  # far camera-left = view-right
        assert m.map(1.0, 1.0) == pytest.approx((0, 1079))
        for x in (-5.0, 5.0):
            for y in (-5.0, 5.0):
                sx, sy = m.map(x, y)
                assert 0 <= sx <= 1919 and 0 <= sy <= 1079

    def test_calibrated_pointer_never_lands_on_a_corner_with_the_default_inset(self):
        m = CursorMapper(*self.SCREEN, bounds=region_from_corners(GOOD))
        for x, y in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]:
            sx, sy = m.map(x, y)
            assert 1 <= sx <= 1918 and 1 <= sy <= 1078

    def test_processor_uses_the_calibrated_region_and_can_revert(self):
        cfg = Config(smoothing=0.0, stable_frames=1, active_margin=0.0)
        proc = GestureProcessor(cfg, self.SCREEN, region_from_corners(GOOD))
        hand = make_hand(index_xy=(1 - 0.2, 0.2))  # top-left of the calibrated region
        first = proc.process(hand, 0.0).cursor
        assert first is not None and first[0] <= 2 and first[1] <= 2
        assert proc.region is not None
        proc.apply_calibration(None)
        assert proc.region is None
        proc.process(None, 0.1)
        reverted = proc.process(hand, 0.2).cursor
        assert reverted[0] > 300  # default mapping puts this point well inside the screen


class TestStore:
    def store(self, tmp_path):
        return CalibrationStore(tmp_path / "calibration.local.json")

    def test_save_then_load_round_trips(self, tmp_path):
        store, region = self.store(tmp_path), region_from_corners(GOOD)
        store.save(region, camera_index=1, aspect_ratio=640 / 480)
        loaded = store.load(camera_index=1, aspect_ratio=640 / 480)
        assert loaded.message is None
        assert loaded.region == region

    def test_file_is_versioned_and_contains_no_image_data(self, tmp_path):
        store = self.store(tmp_path)
        store.save(region_from_corners(GOOD), 0, 4 / 3)
        data = json.loads(store.path.read_text())
        assert data["version"] == 1
        assert set(data) == {"version", "coordinate_space", "region", "camera"}
        assert set(data["camera"]) == {"index", "aspect_ratio"}

    def test_missing_file_is_silent(self, tmp_path):
        result = self.store(tmp_path).load(0, 4 / 3)
        assert result.region is None and result.message is None

    @pytest.mark.parametrize("content", [
        "", "not json", "[]", "null", "{}", '{"version": 1}',
        '{"version": 2, "coordinate_space": "mirrored_view", "region": {}, "camera": {}}',
        '{"version": true, "coordinate_space": "mirrored_view", "region": {}, "camera": {}}',
        b"\xff\xfe\x00garbage".decode("latin-1"),
    ])
    def test_corrupt_or_unsupported_data_falls_back_safely(self, tmp_path, content):
        store = self.store(tmp_path)
        store.path.write_text(content, encoding="utf-8")
        result = store.load(0, 4 / 3)
        assert result.region is None
        assert result.message and "default mapping" in result.message

    def good_payload(self, **overrides):
        data = {
            "version": 1, "coordinate_space": "mirrored_view",
            "region": {"left": 0.2, "top": 0.2, "right": 0.8, "bottom": 0.7},
            "camera": {"index": 0, "aspect_ratio": 4 / 3},
        }
        data.update(overrides)
        return data

    @pytest.mark.parametrize("override", [
        {"region": {"left": 0.8, "top": 0.2, "right": 0.2, "bottom": 0.7}},
        {"region": {"left": 0.4, "top": 0.4, "right": 0.5, "bottom": 0.5}},  # too small
        {"region": {"left": "0.2", "top": 0.2, "right": 0.8, "bottom": 0.7}},
        {"region": {"left": True, "top": 0.2, "right": 0.8, "bottom": 0.7}},
        {"region": {"left": 0.2, "top": 0.2, "right": 0.8}},
        {"region": "nope"},
        {"camera": {"index": -1, "aspect_ratio": 1.3}},
        {"camera": {"index": 0, "aspect_ratio": 0}},
        {"camera": {"index": 0, "aspect_ratio": "wide"}},
        {"coordinate_space": "raw_camera"},
    ])
    def test_invalid_contents_are_rejected(self, tmp_path, override):
        store = self.store(tmp_path)
        store.path.write_text(json.dumps(self.good_payload(**override)))
        assert store.load(0, 4 / 3).region is None

    def test_nan_in_the_file_is_rejected(self, tmp_path):
        store = self.store(tmp_path)
        text = json.dumps(self.good_payload()).replace("0.2", "NaN", 1)
        store.path.write_text(text)
        assert store.load(0, 4 / 3).region is None

    def test_good_payload_loads(self, tmp_path):
        store = self.store(tmp_path)
        store.path.write_text(json.dumps(self.good_payload()))
        assert store.load(0, 4 / 3).region is not None

    def test_calibration_for_a_different_camera_is_ignored_with_a_message(self, tmp_path):
        store = self.store(tmp_path)
        store.save(region_from_corners(GOOD), 0, 4 / 3)
        result = store.load(camera_index=2, aspect_ratio=4 / 3)
        assert result.region is None and "camera 0" in result.message

    def test_calibration_for_a_different_frame_shape_is_ignored(self, tmp_path):
        store = self.store(tmp_path)
        store.save(region_from_corners(GOOD), 0, 4 / 3)
        result = store.load(camera_index=0, aspect_ratio=16 / 9)
        assert result.region is None and "frame shape" in result.message

    def test_oversized_file_is_rejected(self, tmp_path):
        store = self.store(tmp_path)
        store.path.write_text(" " * 20000)
        assert store.load(0, 4 / 3).region is None

    def test_save_rejects_invalid_data_and_writes_nothing(self, tmp_path):
        store = self.store(tmp_path)
        with pytest.raises(CalibrationError):
            store.save(ViewRegion(0.4, 0.4, 0.5, 0.5), 0, 4 / 3)
        with pytest.raises(CalibrationError):
            store.save(region_from_corners(GOOD), -1, 4 / 3)
        with pytest.raises(CalibrationError):
            store.save(region_from_corners(GOOD), 0, math.nan)
        assert not store.path.exists()

    def test_failed_save_does_not_corrupt_an_existing_file(self, tmp_path):
        store = self.store(tmp_path)
        store.save(region_from_corners(GOOD), 0, 4 / 3)
        before = store.path.read_text()
        with pytest.raises(CalibrationError):
            store.save(ViewRegion(0.4, 0.4, 0.5, 0.5), 0, 4 / 3)
        assert store.path.read_text() == before
        assert not list(tmp_path.glob("*.tmp"))

    def test_delete_resets(self, tmp_path):
        store = self.store(tmp_path)
        store.save(region_from_corners(GOOD), 0, 4 / 3)
        assert store.delete() is True
        assert store.delete() is False
        assert store.load(0, 4 / 3).region is None

    def test_save_creates_missing_directories(self, tmp_path):
        store = CalibrationStore(tmp_path / "sub" / "dir" / "c.json")
        store.save(region_from_corners(GOOD), 0, 4 / 3)
        assert store.path.is_file()

    def test_default_path_is_a_local_runtime_file(self):
        assert default_calibration_path().name == "calibration.local.json"


class TestWizard:
    def wizard(self):
        return CalibrationWizard(stable_frames=5, tolerance=0.02)

    def hold(self, wizard, point, frames=5):
        for _ in range(frames):
            wizard.feed(point)

    def capture_all(self, wizard, points=GOOD):
        wizard.start()
        for p in points:
            self.hold(wizard, p)
            assert wizard.confirm()

    def test_happy_path_produces_the_region(self):
        w = self.wizard()
        self.capture_all(w)
        assert w.state is WizardState.SUCCEEDED
        assert w.region == region_from_corners(GOOD)

    def test_nothing_is_captured_without_confirmation(self):
        w = self.wizard()
        w.start()
        self.hold(w, GOOD[0], 50)
        assert w.view().captured == ()

    def test_confirm_requires_a_steady_reliable_fingertip(self):
        w = self.wizard()
        w.start()
        assert not w.confirm()  # nothing seen yet
        self.hold(w, GOOD[0], 3)
        assert not w.confirm()  # not enough frames
        for x in (0.2, 0.3, 0.2, 0.3, 0.2):  # moving around
            w.feed((x, 0.2))
        assert not w.confirm() and "movement" in w.notice

    def test_losing_tracking_resets_the_steady_window(self):
        w = self.wizard()
        w.start()
        self.hold(w, GOOD[0], 4)
        w.feed(None)
        self.hold(w, GOOD[0], 1)
        assert not w.confirm()

    def test_capture_uses_the_median_of_the_steady_window(self):
        w = self.wizard()
        w.start()
        for x in (0.20, 0.205, 0.21, 0.20, 0.205):
            w.feed((x, 0.2))
        assert w.confirm()
        assert w.view().captured[0][0] == pytest.approx(0.205)

    def test_duplicate_corner_is_rejected_at_capture_time(self):
        w = self.wizard()
        w.start()
        self.hold(w, GOOD[0])
        assert w.confirm()
        self.hold(w, GOOD[0])
        assert not w.confirm() and "too close" in w.notice
        assert len(w.view().captured) == 1

    def test_bad_geometry_fails_cleanly(self):
        w = self.wizard()
        self.capture_all(w, [GOOD[1], GOOD[0], GOOD[3], GOOD[2]])  # wrong order
        assert w.state is WizardState.FAILED
        assert w.region is None and "rejected" in w.notice

    def test_cancel_discards_everything(self):
        w = self.wizard()
        w.start()
        self.hold(w, GOOD[0])
        w.confirm()
        w.cancel()
        assert w.state is WizardState.CANCELLED and not w.active
        assert w.region is None and w.view().captured == ()
        w.feed(GOOD[1])  # ignored once cancelled
        assert not w.confirm()

    def test_acknowledge_returns_to_idle_and_it_can_restart(self):
        w = self.wizard()
        self.capture_all(w)
        w.acknowledge()
        assert w.state is WizardState.IDLE
        w.start()
        assert w.active and w.view().captured == () and w.region is None

    def test_cancel_when_not_active_changes_nothing(self):
        w = self.wizard()
        w.cancel()
        assert w.state is WizardState.IDLE

    def test_view_describes_the_current_step(self):
        w = self.wizard()
        w.start()
        assert "1/4" in w.view().instruction and "TOP-LEFT" in w.view().instruction
        self.hold(w, GOOD[0])
        w.confirm()
        assert "2/4" in w.view().instruction and "TOP-RIGHT" in w.view().instruction

    def test_stability_indicator(self):
        w = self.wizard()
        w.start()
        assert w.stability == 0.0
        self.hold(w, GOOD[0], 5)
        assert w.stability == 1.0
        w.feed((0.9, 0.9))
        assert w.stability < 1.0

    @pytest.mark.parametrize("kwargs", [{"stable_frames": 0}, {"tolerance": 0}])
    def test_invalid_parameters_rejected(self, kwargs):
        with pytest.raises(ValueError):
            CalibrationWizard(**kwargs)
