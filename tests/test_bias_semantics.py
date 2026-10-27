"""Fixed expected answers, independent of the implementation's formula."""
import importlib.util
import math
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    'diagnose_incremental',
    Path(__file__).resolve().parents[1] / 'scripts/diagnose_incremental.py')
diagnostics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostics)


class BiasSemanticsTests(unittest.TestCase):
    def test_signed_difference_and_absolute_improvement(self):
        # comparator, candidate, signed difference, absolute-bias improvement
        cases = [
            (-10, -20, 10, -10),   # underprediction worsens
            (-20, -10, -10, 10),   # underprediction improves
            (20, 10, 10, 10),      # overprediction improves
            (10, 20, -10, -10),    # overprediction worsens
            (-20, 10, -30, 10),    # sign crossing improves
            (-10, 20, -30, -10),   # sign crossing worsens
            (20, -10, 30, 10),     # reverse crossing improves
            (10, -20, 30, -10),    # reverse crossing worsens
            (-10, 10, -20, 0),     # equal absolute bias
            (10, -10, 20, 0),
            (10, 10, 0, 0),        # unchanged bias
            (0, 0, 0, 0),
            (-10, 0, -10, 10),     # reaches zero
            (0, -10, 10, -10),     # leaves zero
        ]
        for reference, candidate, signed, improvement in cases:
            with self.subTest(reference=reference, candidate=candidate):
                result = diagnostics.bias_comparison(candidate, reference)
                self.assertEqual(result['signed_bias_difference_pp'], signed)
                self.assertEqual(result['absolute_bias_improvement_pp'], improvement)
                self.assertEqual(result['improvement_total_bias_pct'], signed)

    def test_undefined_bias_stays_undefined(self):
        for candidate, comparator in [(math.nan, 10), (10, math.nan)]:
            self.assertTrue(all(math.isnan(v) for v in
                                diagnostics.bias_comparison(candidate, comparator).values()))


if __name__ == '__main__':
    unittest.main()
