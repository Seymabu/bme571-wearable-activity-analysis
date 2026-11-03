"""Tests of study assumptions, not model-score snapshots."""

import unittest
import numpy as np
import pandas as pd

from src.analysis import (
    bootstrap_subject_scores, compact_feature_indices, make_classifier,
    paired_subject_bootstrap, subject_scores, validate_subject_split,
)


class ValidationDesignTests(unittest.TestCase):
    def test_overlapping_people_are_rejected_even_if_rows_differ(self):
        with self.assertRaisesRegex(ValueError, "Participant leakage"):
            validate_subject_split(np.array([1, 1, 2]), np.array([2, 3, 3]))
        validate_subject_split(np.array([1, 1, 2]), np.array([3, 4, 4]))

    def test_scaler_uses_training_rows_only(self):
        train = np.array([[0., 2.], [1., 4.], [2., 1.], [3., 3.], [4., 2.], [5., 4.]])
        model = make_classifier(1.0).fit(train, np.array([0, 0, 0, 1, 1, 1]))
        before = model.named_steps["scale"].mean_.copy()
        model.predict(np.array([[1e6, -1e6]]))
        np.testing.assert_allclose(before, train.mean(axis=0))
        np.testing.assert_array_equal(model.named_steps["scale"].mean_, before)

    def test_people_are_equally_weighted_despite_different_window_counts(self):
        # Subject1 has six correct windows; subject2 has sixty incorrect windows.
        truth = np.tile(np.arange(1, 7), 11)
        predicted = np.concatenate([np.arange(1, 7), np.tile([2, 3, 4, 5, 6, 1], 10)])
        people = np.concatenate([np.ones(6), np.full(60, 2)])
        scores = subject_scores(truth, predicted, people)
        self.assertAlmostEqual(scores.macro_f1.mean(), 0.5)
        self.assertNotAlmostEqual((truth == predicted).mean(), 0.5)

    def test_paired_bootstrap_preserves_identical_models(self):
        score = np.array([.4, .6, .8, .9])
        result = paired_subject_bootstrap(score, score, n_bootstrap=300)
        self.assertEqual(result, {"mean_difference": 0., "ci_low": 0., "ci_high": 0.})

    def test_paired_bootstrap_preserves_constant_within_person_difference(self):
        reference = np.array([.2, .3, .55, .7, .8])
        result = paired_subject_bootstrap(reference + .1, reference, n_bootstrap=300)
        for value in result.values():
            self.assertAlmostEqual(value, .1)
        first = bootstrap_subject_scores(np.column_stack([reference, reference + .1]), seed=42, n_bootstrap=30)
        second = bootstrap_subject_scores(np.column_stack([reference, reference + .1]), seed=42, n_bootstrap=30)
        np.testing.assert_array_equal(first["replicates"], second["replicates"])

    def test_compact_features_exclude_gyro_frequency_and_mixed_angles(self):
        compact = [f"{signal}-{stat}()-{axis}" for signal in ["tBodyAcc", "tGravityAcc", "tBodyAccJerk"] for stat in ["mean", "std"] for axis in "XYZ"]
        extras = ["tBodyGyro-mean()-X", "fBodyAcc-mean()-X", "angle(tBodyAccMean,gravity)"]
        features = pd.DataFrame({"original_name": extras + compact})
        indices = compact_feature_indices(features)
        np.testing.assert_array_equal(indices, np.arange(3, 21))
        with self.assertRaisesRegex(ValueError, "exactly once"):
            compact_feature_indices(pd.concat([features, features.iloc[[3]]], ignore_index=True))


if __name__ == "__main__":
    unittest.main()
