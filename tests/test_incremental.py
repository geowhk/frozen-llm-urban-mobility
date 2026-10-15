"""Data-free checks for the gravity-plus-representation extension."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location(
    'fit_incremental', Path(__file__).resolve().parents[1]/'scripts/fit_incremental.py')
fit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fit)


class IncrementalTests(unittest.TestCase):
    def test_partial_penalty_solver(self):
        with tempfile.TemporaryDirectory() as folder:
            fit.OUT = Path(folder)
            (fit.OUT/'logs').mkdir()
            fit.CHECKS.clear(); fit.SOLVER.clear()
            fit.synthetic()
            self.assertTrue(all(r['status'] == 'pass' for r in fit.SOLVER))
            self.assertTrue(fit.CHECKS['large_alpha_approaches_base']['passed'])

    def test_training_only_polynomial_scaling(self):
        rng = np.random.default_rng(42)
        train = rng.normal(size=(50, 6))
        heldout = rng.normal(size=(10, 6))
        h, e = fit.preprocess(train, heldout, 'G_POLY2')
        h2, _ = fit.preprocess(train, heldout + 100, 'G_POLY2')
        self.assertEqual(h.shape, (50, 27))
        self.assertEqual(e.shape, (10, 27))
        np.testing.assert_array_equal(h, h2)
        np.testing.assert_allclose(h.mean(axis=0), 0, atol=1e-14)


if __name__ == '__main__':
    unittest.main()
