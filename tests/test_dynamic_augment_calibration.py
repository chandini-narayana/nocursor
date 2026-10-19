"""Training-only augmentation and rejection-threshold calibration (NumPy only, SYNTHETIC data)."""

import numpy as np
import pytest

from dynamic_helpers import circle_path, make_sequence, still_path, swipe_path
from nocursor.dynamic.augment import AugmentConfig, augment_resampled
from nocursor.dynamic.baseline import trajectory_stats
from nocursor.dynamic.features import (
    FeatureConfig,
    extract_features,
    features_from_resampled,
    motion_summary,
    resample_view,
)
from nocursor.dynamic.train import TrainConfig, calibrate_threshold

CONFIG = FeatureConfig()


def resampled(path, **kwargs):
    return resample_view(make_sequence(path, **kwargs), CONFIG)


class TestFeaturesFromResampled:
    def test_it_is_exactly_what_extract_features_computes(self):
        seq = make_sequence(circle_path())
        assert np.array_equal(extract_features(seq, CONFIG),
                              features_from_resampled(resample_view(seq, CONFIG), CONFIG))

    def test_wrong_shapes_are_refused(self):
        from nocursor.dynamic.features import FeatureError

        with pytest.raises(FeatureError):
            features_from_resampled(np.zeros((5, 21, 2)), CONFIG)


class TestAugmentation:
    def test_the_input_is_never_modified(self):
        pts = resampled(swipe_path(1))
        before = pts.copy()
        augment_resampled(pts, np.random.default_rng(0), AugmentConfig())
        assert np.array_equal(pts, before)

    def test_it_is_deterministic_for_a_seed_and_varies_between_seeds(self):
        pts = resampled(swipe_path(1))
        a = augment_resampled(pts, np.random.default_rng(3), AugmentConfig())
        b = augment_resampled(pts, np.random.default_rng(3), AugmentConfig())
        c = augment_resampled(pts, np.random.default_rng(4), AugmentConfig())
        assert np.array_equal(a, b) and not np.array_equal(a, c)

    def test_it_really_perturbs_the_data(self):
        pts = resampled(swipe_path(1))
        out = augment_resampled(pts, np.random.default_rng(0), AugmentConfig())
        assert out.shape == pts.shape and not np.allclose(out, pts)

    @pytest.mark.parametrize("seed", range(40))
    def test_a_swipe_never_changes_direction(self, seed):
        rng = np.random.default_rng(seed)
        for direction in (+1, -1):
            out = augment_resampled(resampled(swipe_path(direction)), rng, AugmentConfig())
            net = motion_summary(features_from_resampled(out, CONFIG)).peak_dx
            assert np.sign(net) == direction and abs(net) > 1.0

    @pytest.mark.parametrize("seed", range(40))
    def test_a_clockwise_circle_stays_clockwise_and_keeps_a_sensible_size(self, seed):
        pts = resampled(circle_path(True))
        original = motion_summary(features_from_resampled(pts, CONFIG)).signed_area
        out = augment_resampled(pts, np.random.default_rng(seed), AugmentConfig())
        area = motion_summary(features_from_resampled(out, CONFIG)).signed_area
        assert original > 0
        # amplitude scaling (0.8 to 1.25) scales the area by 0.64 to 1.56, so the sign never flips
        assert 0.5 * original < area < 2.0 * original

    @pytest.mark.parametrize("seed", range(40))
    def test_a_counter_clockwise_circle_stays_counter_clockwise(self, seed):
        pts = resampled(circle_path(False))
        original = motion_summary(features_from_resampled(pts, CONFIG)).signed_area
        out = augment_resampled(pts, np.random.default_rng(seed), AugmentConfig())
        area = motion_summary(features_from_resampled(out, CONFIG)).signed_area
        assert original < 0 and 2.0 * original < area < 0.5 * original

    def test_there_is_no_horizontal_flip_in_the_configuration(self):
        assert not any("flip" in name or "mirror" in name for name in AugmentConfig().to_dict())

    def test_the_hand_shape_survives_amplitude_changes(self):
        """Scaling the path must not distort the hand itself (features are hand-size relative)."""
        config = AugmentConfig(speed_range=(1.0, 1.0), amplitude_range=(1.2, 1.2),
                               max_rotation_deg=0.0, jitter_sigma=0.0)  # fmt: skip
        pts = resampled(swipe_path(1))
        out = augment_resampled(pts, np.random.default_rng(0), config)
        before = features_from_resampled(pts, CONFIG)
        after = features_from_resampled(out, CONFIG)
        assert np.allclose(before[:, 4:], after[:, 4:], atol=1e-9)  # shape columns unchanged
        assert np.allclose(after[:, 0:2], 1.2 * before[:, 0:2], atol=1e-9)  # path 20% larger

    def test_a_faster_speed_changes_velocity_not_displacement_direction(self):
        config = AugmentConfig(speed_range=(1.15, 1.15), amplitude_range=(1.0, 1.0),
                               max_rotation_deg=0.0, jitter_sigma=0.0)  # fmt: skip
        pts = resampled(swipe_path(1))
        out = augment_resampled(pts, np.random.default_rng(0), config)
        assert trajectory_stats(features_from_resampled(out, CONFIG)).peak_speed_x > \
            trajectory_stats(features_from_resampled(pts, CONFIG)).peak_speed_x

    def test_a_still_hand_stays_essentially_still(self):
        out = augment_resampled(resampled(still_path()), np.random.default_rng(1), AugmentConfig())
        assert trajectory_stats(features_from_resampled(out, CONFIG)).net_dx == pytest.approx(
            0.0, abs=0.2)  # fmt: skip


class TestTrainConfig:
    def test_defaults_are_valid_and_json_friendly(self):
        import json

        json.dumps(TrainConfig().to_dict())

    @pytest.mark.parametrize("kwargs", [{"hidden_sizes": ()}, {"seeds": ()}, {"dropout": 1.0},
                                        {"patience": 0}, {"batch_size": 0}, {"max_epochs": 0},
                                        {"learning_rate": 0.0}])  # fmt: skip
    def test_invalid_settings(self, kwargs):
        with pytest.raises(ValueError):
            TrainConfig(**kwargs)


class TestThresholdCalibration:
    GRID = (0.5, 0.6, 0.7, 0.8, 0.9, 0.999)

    @staticmethod
    def probs(rows):
        arr = np.array(rows, dtype=float)
        return arr / arr.sum(axis=1, keepdims=True)

    def test_the_smallest_threshold_with_no_validation_false_activation_is_chosen(self):
        # labels: 0 = swipe_left, 3 = no_gesture. Negatives: one confidently wrong at 0.75.
        probabilities = self.probs([[0.75, 0.05, 0.1, 0.1],   # negative read as swipe_left, 0.75
                                    [0.05, 0.05, 0.05, 0.85],  # negative read as no_gesture
                                    [0.9, 0.03, 0.03, 0.04],   # true swipe_left, 0.9
                                    [0.6, 0.2, 0.1, 0.1]])     # true swipe_left, 0.6
        labels = np.array([3, 3, 0, 0])
        result = calibrate_threshold(probabilities, labels, self.GRID)
        assert result["threshold"] == 0.8  # first grid value above 0.75
        assert result["validation_false_activations_at_threshold"] == 0
        assert result["validation_gestures_lost_to_rejection"] == 1  # the 0.6 swipe
        assert result["warning"] is None

    def test_a_perfectly_separated_validation_set_gives_the_lowest_threshold(self):
        probabilities = self.probs([[0.9, 0.04, 0.03, 0.03], [0.02, 0.02, 0.02, 0.94]])
        result = calibrate_threshold(probabilities, np.array([0, 3]), self.GRID)
        assert result["threshold"] == 0.5 and result["validation_gestures_lost_to_rejection"] == 0

    def test_when_no_threshold_helps_the_largest_is_used_and_a_warning_recorded(self):
        probabilities = self.probs([[0.9995, 0.0002, 0.0002, 0.0001]])
        result = calibrate_threshold(probabilities, np.array([3]), self.GRID)
        assert result["threshold"] == 0.999
        assert result["validation_false_activations_at_threshold"] == 1
        assert "no threshold" in result["warning"]

    def test_only_negatives_that_would_trigger_matter(self):
        """A negative read as no_gesture, however confident, never raises the threshold."""
        probabilities = self.probs([[0.01, 0.01, 0.01, 0.97], [0.8, 0.1, 0.05, 0.05]])
        result = calibrate_threshold(probabilities, np.array([3, 0]), self.GRID)
        assert result["threshold"] == 0.5

    def test_the_method_and_counts_are_recorded(self):
        result = calibrate_threshold(self.probs([[0.9, 0.04, 0.03, 0.03], [0.1, 0.1, 0.1, 0.7]]),
                                     np.array([0, 3]), self.GRID)  # fmt: skip
        assert "smallest value" in result["method"] and result["validation_negatives"] == 1
        assert 0.0 <= result["validation_macro_f1_at_threshold"] <= 1.0
