from __future__ import annotations
import unittest
import numpy as np
import pandas as pd
from paper1.core import (
    choose_alpha,
    fit_corrected_gravity,
    fit_corrected_marginals,
    fit_standardizer,
    inner_validation_pairs,
    make_folds,
    predict_corrected_gravity,
    predict_corrected_marginal_product,
    validate_folds,
)


class Paper1CoreChecks(unittest.TestCase):

    def setUp(self) -> None:
        pairs = [(f"o{i}", f"d{j}") for i in range(5) for j in range(5) if i != j]
        rows = []
        for i, (orig, dest) in enumerate(pairs):
            for hour in range(24):
                rows.append(
                    {
                        "orig": orig,
                        "dest": dest,
                        "hour": hour,
                        "pair_id": f"{orig}|{dest}",
                        "dist_km": i + 1.0,
                        "y_gt": i + hour + 1.0,
                    }
                )
        self.data = pd.DataFrame(rows)

    def test_fold_assignment_is_balanced_deterministic_and_grouped(self) -> None:
        first = make_folds(self.data["pair_id"])
        second = make_folds(reversed(self.data["pair_id"].tolist()))
        pd.testing.assert_frame_equal(first, second)
        summary = validate_folds(self.data, first)
        self.assertEqual(summary["n_pairs"], 20)
        self.assertEqual(set(summary["fold_pair_counts"].values()), {4})

    def test_inner_validation_is_one_eighth_and_disjoint(self) -> None:
        pairs = [f"p{i}" for i in range(480)]
        validation = inner_validation_pairs(pairs, outer_fold=0)
        self.assertEqual(len(validation), 60)
        self.assertTrue(validation.issubset(set(pairs)))

    def test_standardizer_uses_only_values_given_to_fit(self) -> None:
        train = np.array([[0.0, 2.0], [2.0, 4.0]])
        mean, scale = fit_standardizer(train)
        np.testing.assert_allclose(mean, [1.0, 3.0])
        np.testing.assert_allclose(scale, [1.0, 1.0])
        extreme_test = np.array([[10000.0, -10000.0]])
        mean_again, scale_again = fit_standardizer(train)
        np.testing.assert_array_equal(mean, mean_again)
        np.testing.assert_array_equal(scale, scale_again)
        self.assertFalse(np.isclose(extreme_test.mean(), mean.mean()))

    def test_alpha_tie_chooses_stronger_regularization(self) -> None:
        x_train = np.zeros((16, 3))
        y_train = np.ones(16)
        result = choose_alpha(x_train, y_train, np.zeros((4, 3)), np.ones(4))
        self.assertEqual(result["alpha"], 1000.0)
        self.assertEqual(len(result["candidates"]), 6)

    def test_structural_baselines_do_not_read_test_outcomes(self) -> None:
        folds = make_folds(self.data["pair_id"])
        attached = self.data.merge(folds, on="pair_id")
        train = attached[attached["fold"] != 0].copy()
        test = attached[attached["fold"] == 0].copy()
        changed_test = test.copy()
        changed_test["y_gt"] = 10000000.0
        marginal = fit_corrected_marginals(train, self.data)
        pred_a = predict_corrected_marginal_product(marginal, test)
        pred_b = predict_corrected_marginal_product(marginal, changed_test)
        np.testing.assert_array_equal(pred_a, pred_b)
        gravity = fit_corrected_gravity(train, self.data)
        pred_a = predict_corrected_gravity(gravity, test)
        pred_b = predict_corrected_gravity(gravity, changed_test)
        np.testing.assert_array_equal(pred_a, pred_b)


if __name__ == "__main__":
    unittest.main()
