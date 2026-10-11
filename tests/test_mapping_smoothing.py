import pytest

from nocursor.mapping import CursorMapper
from nocursor.smoothing import ExponentialSmoother


class TestCursorMapper:
    def make(self, **kw):
        return CursorMapper(1920, 1080, margin=0.1, edge_inset=1, **kw)

    def test_center_maps_to_screen_center(self):
        x, y = self.make().map(0.5, 0.5)
        assert x == pytest.approx(959.5)
        assert y == pytest.approx(539.5)

    def test_x_is_mirrored(self):
        mapper = self.make()
        left_in_camera, _ = mapper.map(0.2, 0.5)
        right_in_camera, _ = mapper.map(0.8, 0.5)
        assert left_in_camera > right_in_camera  # camera-left = user's right

    def test_no_mirror_option(self):
        mapper = self.make(mirror=False)
        assert mapper.map(0.2, 0.5)[0] < mapper.map(0.8, 0.5)[0]

    @pytest.mark.parametrize("x", [-5.0, 0.0, 1.0, 5.0])
    @pytest.mark.parametrize("y", [-5.0, 0.0, 1.0, 5.0])
    def test_never_leaves_inset_bounds_so_corners_are_unreachable(self, x, y):
        sx, sy = self.make().map(x, y)
        assert 1.0 <= sx <= 1918.0
        assert 1.0 <= sy <= 1078.0

    def test_extremes_hit_inset_bounds(self):
        mapper = self.make()
        assert mapper.map(1.0, 0.0) == (1.0, 1.0)  # mirrored: camera right edge -> screen left
        assert mapper.map(0.0, 1.0) == (1918.0, 1078.0)

    def test_active_region_edges_reach_screen_edges(self):
        mapper = self.make()
        x, y = mapper.map(0.9, 0.1)  # mirrored x=0.1 -> left edge of active region
        assert (x, y) == (1.0, 1.0)

    def test_screen_size_not_hardcoded(self):
        small = CursorMapper(800, 600, margin=0.0)
        assert small.map(0.0, 1.0)[0] <= 799
        assert small.map(0.0, 1.0)[1] <= 599

    @pytest.mark.parametrize("kwargs", [
        {"screen_width": 0, "screen_height": 100},
        {"screen_width": 100, "screen_height": -1},
        {"screen_width": 100, "screen_height": 100, "margin": 0.5},
        {"screen_width": 100, "screen_height": 100, "margin": -0.1},
        {"screen_width": 4, "screen_height": 100, "edge_inset": 2},
    ])
    def test_invalid_arguments_rejected(self, kwargs):
        with pytest.raises(ValueError):
            CursorMapper(**kwargs)


class TestExponentialSmoother:
    def test_first_sample_passes_through(self):
        assert ExponentialSmoother(0.8).update((10.0, 20.0)) == (10.0, 20.0)

    def test_zero_smoothing_is_identity(self):
        s = ExponentialSmoother(0.0)
        s.update((0.0, 0.0))
        assert s.update((100.0, 50.0)) == (100.0, 50.0)

    def test_smoothing_blends_toward_new_value(self):
        s = ExponentialSmoother(0.5)
        s.update((0.0, 0.0))
        assert s.update((100.0, 40.0)) == (50.0, 20.0)

    def test_converges_to_constant_input(self):
        s = ExponentialSmoother(0.7)
        s.update((0.0, 0.0))
        for _ in range(100):
            x, y = s.update((10.0, 10.0))
        assert x == pytest.approx(10.0, abs=1e-6)

    def test_more_smoothing_means_slower_response(self):
        light, heavy = ExponentialSmoother(0.2), ExponentialSmoother(0.8)
        for s in (light, heavy):
            s.update((0.0, 0.0))
        assert light.update((100.0, 0.0))[0] > heavy.update((100.0, 0.0))[0]

    def test_reset_forgets_history(self):
        s = ExponentialSmoother(0.9)
        s.update((0.0, 0.0))
        s.reset()
        assert s.update((500.0, 500.0)) == (500.0, 500.0)

    @pytest.mark.parametrize("bad", [-0.1, 1.0, 2.0])
    def test_invalid_smoothing_rejected(self, bad):
        with pytest.raises(ValueError):
            ExponentialSmoother(bad)
