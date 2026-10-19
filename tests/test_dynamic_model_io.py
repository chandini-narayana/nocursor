"""NumPy-only model artifacts: save/load, the forward pass, rejection, and corrupt artifacts.

The weights here are random or hand-set test fixtures (SYNTHETIC): they check the loading and
inference machinery, and say nothing about how a trained model performs.
"""

import json

import numpy as np
import pytest

from dynamic_helpers import circle_path, make_sequence, swipe_path
from ml_fixtures import model_metadata, random_arrays
from nocursor.dynamic.classifier import REJECTED, predict_sequence
from nocursor.dynamic.features import FEATURE_DIM, FeatureConfig, extract_features
from nocursor.dynamic.model_io import (
    ARTIFACT_SCHEMA_VERSION,
    META_FILE,
    MODEL_FILE,
    TENSOR_NAMES,
    ModelArtifactError,
    load_classifier,
    save_artifact,
    weights_checksum,
)
from nocursor.dynamic.schema import CLASS_NAMES


@pytest.fixture
def model_dir(tmp_path):
    directory = tmp_path / "model"
    save_artifact(directory, random_arrays(), model_metadata())
    return directory


def rewrite_meta(directory, edit):
    path = directory / META_FILE
    meta = json.loads(path.read_text(encoding="utf-8"))
    edit(meta)
    path.write_text(json.dumps(meta), encoding="utf-8")


def rewrite_arrays(directory, edit, *, fix_checksum=False):
    with np.load(directory / MODEL_FILE, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    edit(arrays)
    np.savez(directory / MODEL_FILE, **arrays)
    if fix_checksum:
        rewrite_meta(directory, lambda m: m.update(weights_sha256=weights_checksum(arrays)))


class TestRoundTrip:
    def test_a_saved_model_loads_and_predicts(self, model_dir):
        model = load_classifier(model_dir)
        features = extract_features(make_sequence(swipe_path(1)))
        prediction = model.predict_features(features)
        assert prediction.valid and prediction.label in {*CLASS_NAMES, REJECTED}
        assert prediction.probabilities is None or sum(prediction.probabilities) == pytest.approx(1)

    def test_prediction_is_deterministic(self, model_dir):
        model = load_classifier(model_dir)
        features = extract_features(make_sequence(circle_path()))
        assert model.predict_features(features) == model.predict_features(features.copy())
        assert np.array_equal(model.logits(features), load_classifier(model_dir).logits(features))

    def test_the_metadata_is_complete(self, model_dir):
        meta = json.loads((model_dir / META_FILE).read_text(encoding="utf-8"))
        assert meta["artifact_schema_version"] == ARTIFACT_SCHEMA_VERSION
        assert meta["class_names"] == list(CLASS_NAMES)
        assert meta["feature_schema_hash"] == FeatureConfig().schema_hash()
        assert len(meta["weights_sha256"]) == 64 and meta["rejection"]["threshold"] == 0.6

    def test_the_arrays_are_float32_and_load_without_pickle(self, model_dir):
        with np.load(model_dir / MODEL_FILE, allow_pickle=False) as data:
            assert sorted(data.files) == sorted(TENSOR_NAMES)
            assert all(data[name].dtype == np.float32 for name in data.files)

    def test_the_checksum_ignores_nothing_it_should_cover(self):
        arrays = {k: v.astype(np.float32) for k, v in random_arrays().items()}
        base = weights_checksum(arrays)
        arrays["fc.bias"][0] += 1e-6
        assert weights_checksum(arrays) != base


class TestForwardPass:
    def zero_gru(self, bias):
        arrays = {k: np.zeros_like(v) for k, v in random_arrays().items()}
        arrays["norm.std"] = np.ones(FEATURE_DIM)
        arrays["fc.bias"] = np.array(bias, dtype=float)
        return arrays

    def make(self, tmp_path, arrays, threshold=0.6):
        save_artifact(tmp_path / "m", arrays, model_metadata(threshold=threshold))
        return load_classifier(tmp_path / "m")

    def test_with_zero_weights_the_logits_are_the_output_bias(self, tmp_path):
        """r = z = 0.5 and the candidate is tanh(0) = 0, so the hidden state stays zero."""
        model = self.make(tmp_path, self.zero_gru([0.5, -1.0, 2.0, 0.25]))
        features = extract_features(make_sequence(swipe_path(1)))
        assert np.allclose(model.logits(features), [0.5, -1.0, 2.0, 0.25], atol=1e-6)

    def test_the_output_depends_on_the_input(self, model_dir):
        model = load_classifier(model_dir)
        a = model.logits(extract_features(make_sequence(swipe_path(+1))))
        b = model.logits(extract_features(make_sequence(swipe_path(-1))))
        assert not np.allclose(a, b)

    def test_probabilities_are_a_softmax_of_the_logits(self, tmp_path):
        model = self.make(tmp_path, self.zero_gru([0.0, 0.0, 3.0, 0.0]), threshold=0.1)
        prediction = model.predict_features(extract_features(make_sequence(swipe_path(1))))
        expected = np.exp([0, 0, 3, 0]) / np.exp([0, 0, 3, 0]).sum()
        assert np.allclose(prediction.probabilities, expected)
        assert prediction.label == "circle_clockwise"


class TestRejection:
    @staticmethod
    def biased(tmp_path, bias, threshold):
        arrays = {k: np.zeros_like(v) for k, v in random_arrays().items()}
        arrays["norm.std"] = np.ones(FEATURE_DIM)
        arrays["fc.bias"] = np.array(bias, dtype=float)
        save_artifact(tmp_path / "m", arrays, model_metadata(threshold=threshold))
        return load_classifier(tmp_path / "m")

    FEATURES = extract_features(make_sequence(swipe_path(1)))

    def test_a_confident_gesture_is_accepted(self, tmp_path):
        model = self.biased(tmp_path, [0, 6, 0, 0], threshold=0.9)
        assert model.predict_features(self.FEATURES).label == "swipe_right"

    def test_a_low_confidence_gesture_is_rejected(self, tmp_path):
        model = self.biased(tmp_path, [1.0, 0, 0, 0], threshold=0.9)  # p of swipe_left is ~0.48
        prediction = model.predict_features(self.FEATURES)
        assert prediction.label == REJECTED and prediction.valid
        assert prediction.note == "below the rejection threshold"
        assert prediction.effective_label == "no_gesture" and not prediction.activates

    def test_the_threshold_is_the_boundary(self, tmp_path):
        bias = [2.0, 0, 0, 0]
        p = np.exp(2.0) / (np.exp(2.0) + 3)
        assert self.biased(tmp_path, bias, threshold=p - 0.01).predict_features(
            self.FEATURES).label == "swipe_left"
        assert self.biased(tmp_path / "x", bias, threshold=p + 0.01).predict_features(
            self.FEATURES).label == REJECTED

    def test_no_gesture_is_never_a_rejection(self, tmp_path):
        model = self.biased(tmp_path, [0, 0, 0, 1.0], threshold=0.99)  # confident only in name
        assert model.predict_features(self.FEATURES).label == "no_gesture"

    def test_a_threshold_of_one_rejects_every_gesture(self, tmp_path):
        model = self.biased(tmp_path, [0, 12, 0, 0], threshold=1.0)
        assert model.predict_features(self.FEATURES).label == REJECTED

    def test_unusable_input_is_rejected_and_marked_invalid(self, model_dir):
        model = load_classifier(model_dir)
        bad_shape = model.predict_features(np.zeros((5, FEATURE_DIM)))
        not_finite = model.predict_features(np.full((32, FEATURE_DIM), np.nan))
        for prediction in (bad_shape, not_finite):
            assert prediction.label == REJECTED and not prediction.valid
            assert prediction.note == "invalid input features"

    def test_non_finite_model_output_is_rejected_and_marked_invalid(self, model_dir, monkeypatch):
        model = load_classifier(model_dir)
        monkeypatch.setattr(model, "logits", lambda features: np.array([np.nan, 0, 0, 0]))
        prediction = model.predict_features(self.FEATURES)
        assert prediction.label == REJECTED and not prediction.valid
        assert prediction.note == "invalid model output"
        monkeypatch.setattr(model, "logits", lambda features: np.array([np.inf, 0, 0, 0]))
        assert not model.predict_features(self.FEATURES).valid
        monkeypatch.setattr(model, "logits", lambda features: np.zeros(3))
        assert not model.predict_features(self.FEATURES).valid

    def test_a_sequence_that_cannot_become_features_gives_no_prediction(self, model_dir):
        model = load_classifier(model_dir)
        short = make_sequence(swipe_path(1), duration=0.9)
        assert predict_sequence(model, short, FeatureConfig()) is None


class TestCorruptOrIncompatibleArtifacts:
    def test_missing_files(self, model_dir):
        (model_dir / META_FILE).unlink()
        with pytest.raises(ModelArtifactError, match="missing"):
            load_classifier(model_dir)

    def test_missing_weights(self, model_dir):
        (model_dir / MODEL_FILE).unlink()
        with pytest.raises(ModelArtifactError, match="missing"):
            load_classifier(model_dir)

    def test_a_directory_that_is_not_a_model(self, tmp_path):
        with pytest.raises(ModelArtifactError):
            load_classifier(tmp_path)

    def test_invalid_json(self, model_dir):
        (model_dir / META_FILE).write_text("{nope", encoding="utf-8")
        with pytest.raises(ModelArtifactError, match="not valid JSON"):
            load_classifier(model_dir)

    def test_json_that_is_not_an_object(self, model_dir):
        (model_dir / META_FILE).write_text("[1, 2]", encoding="utf-8")
        with pytest.raises(ModelArtifactError, match="JSON object"):
            load_classifier(model_dir)

    def test_a_weights_file_that_is_not_an_archive(self, model_dir):
        (model_dir / MODEL_FILE).write_bytes(b"this is not a zip file")
        with pytest.raises(ModelArtifactError, match="cannot be loaded safely"):
            load_classifier(model_dir)

    def test_a_truncated_weights_file(self, model_dir):
        data = (model_dir / MODEL_FILE).read_bytes()
        (model_dir / MODEL_FILE).write_bytes(data[: len(data) // 2])
        with pytest.raises(ModelArtifactError, match="cannot be loaded safely"):
            load_classifier(model_dir)

    def test_a_pickled_object_array_is_refused_not_executed(self, model_dir):
        class Boom:
            def __reduce__(self):  # would run on unpickling
                return (exec, ("raise SystemExit('executed')",))

        arrays = {name: np.zeros(1) for name in TENSOR_NAMES}
        arrays["fc.bias"] = np.array([Boom()], dtype=object)
        np.savez(model_dir / MODEL_FILE, **arrays)
        with pytest.raises(ModelArtifactError, match="cannot be loaded safely"):
            load_classifier(model_dir)

    def test_an_edited_weight_fails_the_checksum(self, model_dir):
        rewrite_arrays(model_dir, lambda a: a["fc.bias"].__setitem__(0, a["fc.bias"][0] + 0.5))
        with pytest.raises(ModelArtifactError, match="checksum mismatch"):
            load_classifier(model_dir)

    def test_a_wrong_class_list(self, model_dir):
        rewrite_meta(model_dir, lambda m: m.update(class_names=["a", "b", "c", "d"]))
        with pytest.raises(ModelArtifactError, match="class names"):
            load_classifier(model_dir)

    def test_a_reordered_class_list(self, model_dir):
        rewrite_meta(model_dir, lambda m: m["class_names"].reverse())
        with pytest.raises(ModelArtifactError, match="class names"):
            load_classifier(model_dir)

    def test_a_different_feature_definition(self, model_dir):
        rewrite_meta(model_dir, lambda m: m["feature_config"].update(steps=40))
        with pytest.raises(ModelArtifactError, match="feature hash mismatch"):
            load_classifier(model_dir)

    def test_a_different_feature_schema_version(self, model_dir):
        rewrite_meta(model_dir, lambda m: m.update(feature_schema_version=7))
        with pytest.raises(ModelArtifactError, match="feature schema version"):
            load_classifier(model_dir)

    def test_an_unknown_artifact_version(self, model_dir):
        rewrite_meta(model_dir, lambda m: m.update(artifact_schema_version=99))
        with pytest.raises(ModelArtifactError, match="artifact_schema_version"):
            load_classifier(model_dir)

    def test_a_missing_metadata_field(self, model_dir):
        rewrite_meta(model_dir, lambda m: m.pop("weights_sha256"))
        with pytest.raises(ModelArtifactError, match="missing weights_sha256"):
            load_classifier(model_dir)

    @pytest.mark.parametrize("threshold", [0, -0.2, 1.5, "high", None])
    def test_an_invalid_rejection_threshold(self, model_dir, threshold):
        rewrite_meta(model_dir, lambda m: m["rejection"].update(threshold=threshold))
        with pytest.raises(ModelArtifactError, match="threshold"):
            load_classifier(model_dir)

    def test_an_unsupported_architecture(self, model_dir):
        rewrite_meta(model_dir, lambda m: m["architecture"].update(num_layers=2))
        with pytest.raises(ModelArtifactError, match="architecture"):
            load_classifier(model_dir)

    def test_a_missing_tensor(self, model_dir):
        rewrite_arrays(model_dir, lambda a: a.pop("fc.bias"))
        with pytest.raises(ModelArtifactError, match="tensors"):
            load_classifier(model_dir)

    def test_an_extra_tensor(self, model_dir):
        rewrite_arrays(model_dir, lambda a: a.update(surprise=np.zeros(2, dtype=np.float32)))
        with pytest.raises(ModelArtifactError, match="tensors"):
            load_classifier(model_dir)

    def test_a_tensor_of_the_wrong_shape(self, model_dir):
        rewrite_arrays(model_dir, lambda a: a.update({"fc.weight": a["fc.weight"][:, :3]}),
                       fix_checksum=True)  # fmt: skip
        with pytest.raises(ModelArtifactError, match="shape"):
            load_classifier(model_dir)

    def test_a_tensor_of_the_wrong_dtype(self, model_dir):
        rewrite_arrays(model_dir, lambda a: a.update({"fc.bias": a["fc.bias"].astype(np.float64)}),
                       fix_checksum=True)  # fmt: skip
        with pytest.raises(ModelArtifactError, match="float32"):
            load_classifier(model_dir)

    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    def test_non_finite_weights_are_refused_even_with_a_matching_checksum(self, model_dir, bad):
        rewrite_arrays(model_dir, lambda a: a["gru.bias_hh"].__setitem__(0, bad), fix_checksum=True)
        with pytest.raises(ModelArtifactError, match="NaN or infinity"):
            load_classifier(model_dir)

    def test_a_non_positive_standard_deviation(self, model_dir):
        rewrite_arrays(model_dir, lambda a: a["norm.std"].__setitem__(0, 0.0), fix_checksum=True)
        with pytest.raises(ModelArtifactError, match="norm.std"):
            load_classifier(model_dir)

    def test_saving_refuses_what_loading_would_refuse(self, tmp_path):
        meta = model_metadata()
        meta["class_names"] = ["x"]
        with pytest.raises(ModelArtifactError):
            save_artifact(tmp_path / "bad", random_arrays(), meta)
        assert not (tmp_path / "bad" / MODEL_FILE).exists()

    def test_saving_refuses_non_finite_weights(self, tmp_path):
        arrays = random_arrays()
        arrays["fc.weight"][0, 0] = np.nan
        with pytest.raises(ModelArtifactError, match="NaN or infinity"):
            save_artifact(tmp_path / "bad", arrays, model_metadata())
