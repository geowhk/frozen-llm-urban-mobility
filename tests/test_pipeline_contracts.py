"""Synthetic tests require no manuscript, mobility data, GPU, or network."""

import unittest
import tempfile
from pathlib import Path
import numpy as np
from urban_mobility.polynomial import transform_fit, transform, fit
from urban_mobility.diagnostics import total_variation, metrics
from urban_mobility.config import MODELS, NEW
from urban_mobility.cli import stage_inputs


class PipelineContracts(unittest.TestCase):
    def test_polynomial_has_27_features_and_uses_only_training_stats(self):
        rng = np.random.default_rng(42)
        training, validation = rng.normal(size=(50, 6)), rng.normal(size=(8, 6)) + 30
        expanded, state = transform_fit(training)
        self.assertEqual(expanded.shape, (50, 27))
        means = [state[0].mean_.copy(), state[2].mean_.copy()]
        self.assertEqual(transform(validation, state).shape, (8, 27))
        np.testing.assert_array_equal(means[0], state[0].mean_)
        np.testing.assert_array_equal(means[1], state[2].mean_)
        np.testing.assert_allclose(state[0].mean_, training.mean(axis=0))

    def test_ridge_centers_target_without_target_sd_scaling(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(50, 6))
        y = rng.normal(size=50) * 4 + 20
        a = fit(x, y, 1.0)
        b = fit(x, y + 10, 1.0)
        np.testing.assert_allclose(a.coef_, b.coef_, atol=1e-12)
        self.assertEqual(a.solver, "svd")
        self.assertFalse(a.fit_intercept)

    def test_metrics_and_tv(self):
        y, p = np.array([1.0, 3.0]), np.array([2.0, 6.0])
        self.assertEqual(total_variation(y, p), 0.0)
        self.assertTrue(np.isnan(total_variation(y, np.zeros(2))))
        self.assertAlmostEqual(metrics(y, p)["mae"], 2.0)
        self.assertEqual(len(MODELS), 7)
        self.assertIn(NEW, MODELS)

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as root:
            source, target = Path(root) / "source", Path(root) / "target"
            source.mkdir()
            target.mkdir()
            with self.assertRaises(FileExistsError):
                stage_inputs(source, target)


if __name__ == "__main__":
    unittest.main()
