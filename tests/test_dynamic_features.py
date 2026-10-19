"""Feature extraction: determinism, invariances, and direction preservation.

All sequences are SYNTHETIC test fixtures (see dynamic_helpers); these tests verify the maths,
not recognition accuracy on real hands.
"""

import numpy as np
import pytest

from dynamic_helpers import circle_path, make_sequence, still_path, swipe_path
from nocursor.dynamic.features import (
    FEATURE_DIM,
    FEATURE_NAMES,
    FeatureConfig,
    FeatureError,
    extract_features,
    longest_tracking_gap_s,
    motion_summary,
    to_view_coordinates,
)
from nocursor.dynamic.schema import SequenceRecord

TRAJ_X, TRAJ_Y, VEL_X = 0, 1, 2


def features(seq, config=None):
    return extract_features(seq, config)


class TestShapeAndDeterminism:
    def test_shape_and_names_agree(self):
        f = features(make_sequence(swipe_path(1)))
        assert f.shape == (FeatureConfig().steps, FEATURE_DIM) == (32, 46)
        assert len(FEATURE_NAMES) == FEATURE_DIM == len(set(FEATURE_NAMES))
        assert np.all(np.isfinite(f))

    def test_same_input_gives_bit_identical_output(self):
        seq = make_sequence(circle_path())
        assert np.array_equal(features(seq), features(seq))

    def test_input_is_not_modified(self):
        seq = make_sequence(swipe_path(-1))
        before = seq.landmarks.copy(), seq.t_ms.copy(), seq.usable.copy()
        features(seq)
        assert np.array_equal(before[0], seq.landmarks)
        assert np.array_equal(before[1], seq.t_ms)
        assert np.array_equal(before[2], seq.usable)

    def test_schema_hash_is_stable_and_tracks_the_configuration(self):
        assert FeatureConfig().schema_hash() == FeatureConfig().schema_hash()
        assert FeatureConfig(steps=40).schema_hash() != FeatureConfig().schema_hash()
        assert FeatureConfig(max_gap_s=0.3).schema_hash() != FeatureConfig().schema_hash()

    @pytest.mark.parametrize("kwargs", [
        {"steps": 3}, {"window_s": 0.0}, {"duration_tolerance_s": 5.0}, {"max_gap_s": 0.0},
        {"min_usable_fraction": 0.0}, {"min_usable_fraction": 1.5}, {"min_hand_size": 0.0},
    ])
    def test_invalid_configuration_is_rejected(self, kwargs):
        with pytest.raises(ValueError):
            FeatureConfig(**kwargs)


class TestDirectionIsPreserved:
    """Normalisation must never throw away which way the hand went."""

    def test_swipe_right_moves_x_positive_and_left_negative(self):
        right = motion_summary(features(make_sequence(swipe_path(+1))))
        left = motion_summary(features(make_sequence(swipe_path(-1))))
        assert right.peak_dx > 1.0
        assert left.peak_dx < -1.0
        assert right.peak_dx == pytest.approx(-left.peak_dx)

    def test_left_and_right_swipe_features_are_mirror_images_in_the_trajectory(self):
        right = features(make_sequence(swipe_path(+1)))
        left = features(make_sequence(swipe_path(-1)))
        assert np.allclose(right[:, TRAJ_X], -left[:, TRAJ_X])
        assert np.allclose(right[:, VEL_X], -left[:, VEL_X])
        assert not np.allclose(right, left)

    def test_raw_camera_x_decreasing_is_a_swipe_to_the_users_right(self):
        """MediaPipe's x is unmirrored: the user's right arrives as a *falling* raw x."""
        seq = make_sequence(swipe_path(+1))
        raw_wrist_x = seq.landmarks[:, 0, 0]
        assert raw_wrist_x[-1] < raw_wrist_x[0]
        assert features(seq)[-1, TRAJ_X] > 0

    def test_clockwise_circle_has_positive_area_and_counter_clockwise_negative(self):
        clockwise = motion_summary(features(make_sequence(circle_path(True))))
        counter = motion_summary(features(make_sequence(circle_path(False))))
        assert clockwise.signed_area > 1.0
        assert counter.signed_area < -1.0
        assert clockwise.signed_area == pytest.approx(-counter.signed_area)

    def test_clockwise_circle_goes_right_while_at_the_top(self):
        """Clockwise as drawn on the preview: from the top the path moves right, then down."""
        f = features(make_sequence(circle_path(True)))
        quarter = f.shape[0] // 4
        assert f[quarter, TRAJ_X] > 0  # a quarter of the way round: to the right of the start
        assert f[quarter, TRAJ_Y] > 0  # and lower than the top (y points down)

    def test_a_still_hand_has_no_motion(self):
        motion = motion_summary(features(make_sequence(still_path())))
        assert motion.path_length == pytest.approx(0.0, abs=1e-9)
        assert motion.signed_area == pytest.approx(0.0, abs=1e-9)

    def test_speed_is_kept_not_normalised_away(self):
        fast = features(make_sequence(swipe_path(+1), duration=1.6))
        slow_path = lambda t: swipe_path(+1)(min(1.0, t * 0.5))  # noqa: E731 - half the sweep
        slow = features(make_sequence(slow_path))
        assert np.abs(fast[:, VEL_X]).max() > 1.5 * np.abs(slow[:, VEL_X]).max()


class TestInvariances:
    def test_view_conversion_mirrors_and_applies_the_aspect_ratio(self):
        points = np.zeros((21, 3))
        points[5] = (0.25, 0.5, 0.0)
        view = to_view_coordinates(points, aspect_ratio=2.0)
        assert view[5, 0] == pytest.approx((1 - 0.25) * 2.0)
        assert view[5, 1] == pytest.approx(0.5)
        assert view.shape == (21, 2)

    def test_z_is_ignored(self):
        seq = make_sequence(swipe_path(+1))
        deeper = SequenceRecord(
            sequence_id=seq.sequence_id, session_id=seq.session_id, label=seq.label,
            t_ms=seq.t_ms, landmarks=seq.landmarks + np.array([0.0, 0.0, 0.3]),
            confidence=seq.confidence, usable=seq.usable, aspect_ratio=seq.aspect_ratio,
        )  # fmt: skip
        assert np.array_equal(features(seq), features(deeper))

    def test_hand_size_does_not_change_the_features(self):
        """The same gesture, relative to the hand, near and far from the camera."""
        near = features(make_sequence(swipe_path(+1, amplitude=0.2 * 1.5), hand_size=0.3))
        far = features(make_sequence(swipe_path(+1, amplitude=0.2 * 0.75), hand_size=0.15))
        assert np.allclose(near[:, :4], far[:, :4], atol=1e-9)  # trajectory and velocity
        # Hand shape: the test hand's thumb chain (landmarks 1-3) has a fixed offset that does
        # not scale with hand size, a quirk of the fixture, so compare the other landmarks.
        shape_near, shape_far = near[:, 4:].reshape(32, 21, 2), far[:, 4:].reshape(32, 21, 2)
        keep = [i for i in range(21) if i not in (1, 2, 3)]
        assert np.allclose(shape_near[:, keep], shape_far[:, keep], atol=1e-9)

    @pytest.mark.parametrize("aspect", [1.0, 16 / 9])
    def test_frame_dimensions_do_not_change_the_features(self, aspect):
        """The same physical motion seen by 4:3, square and 16:9 frames."""
        reference = features(make_sequence(circle_path(), aspect=4 / 3))
        other = features(make_sequence(circle_path(), aspect=aspect))
        assert np.allclose(reference, other, atol=1e-6)

    def test_absolute_position_does_not_change_the_trajectory(self):
        centred = make_sequence(swipe_path(+1))
        shifted = SequenceRecord(
            sequence_id="sess-0002", session_id="sess", label="swipe_right",
            t_ms=centred.t_ms, landmarks=centred.landmarks + np.array([-0.05, 0.03, 0.0]),
            confidence=centred.confidence, usable=centred.usable,
            aspect_ratio=centred.aspect_ratio,
        )  # fmt: skip
        assert np.allclose(features(centred), features(shifted), atol=1e-9)

    @pytest.mark.parametrize("fps", [15.0, 20.0, 25.0, 60.0])
    def test_camera_frame_rate_does_not_change_linear_motion(self, fps):
        """Constant-velocity motion resamples to the same features at any frame rate.

        The rates are those for which 1.6 s is a whole number of frames, so the fixture spans
        the full window. Timestamps are stored as whole milliseconds, which is the remaining
        (small) difference, hence the 0.02 tolerance (in hand sizes / hand sizes per second).
        """
        ramp = lambda t: (0.4 * t - 0.2, 0.0)  # noqa: E731 - constant velocity to the right
        base = features(make_sequence(ramp, fps=30.0))
        other = features(make_sequence(ramp, fps=fps))
        assert np.allclose(base, other, atol=0.02)
        assert other[-1, TRAJ_X] == pytest.approx(base[-1, TRAJ_X], abs=0.01)

    def test_irregular_frame_timing_is_handled(self):
        seq = make_sequence(swipe_path(+1), fps=30.0)
        jittered = seq.t_ms + np.where(np.arange(seq.n_frames) % 2, 4.0, 0.0)
        jittered[0] = 0.0
        shaky = SequenceRecord(
            sequence_id="sess-0003", session_id="sess", label="swipe_right", t_ms=jittered,
            landmarks=seq.landmarks, confidence=seq.confidence, usable=seq.usable,
            aspect_ratio=seq.aspect_ratio,
        )  # fmt: skip
        assert np.allclose(features(seq)[:, TRAJ_X], features(shaky)[:, TRAJ_X], atol=0.15)


class TestDropoutsAndLimits:
    def test_a_short_dropout_is_bridged(self):
        count = make_sequence(swipe_path(+1)).n_frames
        usable = np.ones(count, dtype=bool)
        usable[20:23] = False  # three frames at 30 fps = 0.1 s
        gappy = features(make_sequence(swipe_path(+1), usable=usable))
        whole = features(make_sequence(swipe_path(+1)))
        assert np.allclose(gappy[:, TRAJ_X], whole[:, TRAJ_X], atol=0.3)

    def test_a_long_dropout_is_an_error_not_a_guess(self):
        count = make_sequence(swipe_path(+1)).n_frames
        usable = np.ones(count, dtype=bool)
        usable[15:22] = False  # seven frames: a 0.27 s gap between trusted frames
        with pytest.raises(FeatureError, match="tracking dropped"):
            features(make_sequence(swipe_path(+1), usable=usable))

    def test_too_few_trusted_frames(self):
        count = make_sequence(swipe_path(+1)).n_frames
        usable = np.zeros(count, dtype=bool)
        usable[::2] = True  # every other frame: 50%, with only tiny gaps
        with pytest.raises(FeatureError, match="trusted in only"):
            features(make_sequence(swipe_path(+1), usable=usable))

    def test_a_sequence_shorter_than_the_window(self):
        with pytest.raises(FeatureError, match="window"):
            features(make_sequence(swipe_path(+1), duration=1.0))

    def test_a_tiny_hand_is_refused(self):
        with pytest.raises(FeatureError, match="too small"):
            features(make_sequence(still_path(), hand_size=0.01))

    def test_a_longer_sequence_is_cut_to_the_window(self):
        long = make_sequence(swipe_path(+1), duration=2.4)
        assert features(long).shape == (32, FEATURE_DIM)

    def test_longest_tracking_gap_counts_the_edges(self):
        t = np.arange(10) * 0.1
        usable = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1], dtype=bool)
        assert longest_tracking_gap_s(t, usable) == pytest.approx(0.3)
        assert longest_tracking_gap_s(t, np.zeros(10, dtype=bool)) == float("inf")


class TestFailureCategories:
    @pytest.mark.parametrize("message, category", [
        ("tracking dropped for 0.27 s (limit 0.20 s)", "tracking dropped for too long"),
        ("hand trusted in only 76.2% of frames (need 80.0%)", "too few trusted frames"),
        ("hand trusted in only 31.0% of frames (need 80.0%)", "too few trusted frames"),
        ("sequence covers 1.00 s but the window is 1.60 s", "shorter than the window"),
        ("hand too small to normalise (size 0.010 frame heights)", "hand too small"),
        ("something unexpected", "other"),
    ])
    def test_messages_group_by_cause_not_by_the_numbers_in_them(self, message, category):
        from nocursor.dynamic.features import failure_category

        assert failure_category(message) == category

    def test_every_real_failure_message_has_a_named_category(self):
        from nocursor.dynamic.features import failure_category

        count = make_sequence(swipe_path(+1)).n_frames
        usable = np.zeros(count, dtype=bool)
        usable[::2] = True
        for kwargs, expected in (
            ({"usable": usable}, "too few trusted frames"),
            ({"duration": 1.0}, "shorter than the window"),
            ({"hand_size": 0.01}, "hand too small"),
        ):
            with pytest.raises(FeatureError) as info:
                features(make_sequence(swipe_path(+1), **kwargs))
            assert failure_category(str(info.value)) == expected
