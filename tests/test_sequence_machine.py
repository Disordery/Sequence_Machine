"""
Unit tests for Sequence Machine.

Run with:

    python -m unittest discover -s tests -v

No third-party test framework is required - everything here uses the
standard library's ``unittest``.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from fractions import Fraction as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sequence_machine import analyze, Sequence
from sequence_machine.core import to_fraction, RecurrenceRule, build_polynomial_expr, simplify_expr, Const, Index, BinOp
from sequence_machine.fast_surf import berlekamp_massey


class TestNumberParsing(unittest.TestCase):
    def test_integers_and_floats(self):
        self.assertEqual(to_fraction(3), F(3))
        self.assertEqual(to_fraction(-3.5), F(-7, 2))

    def test_fraction_strings(self):
        self.assertEqual(to_fraction("1/3"), F(1, 3))
        self.assertEqual(to_fraction("-7/2"), F(-7, 2))
        self.assertEqual(to_fraction("3.14"), F("3.14"))

    def test_rejects_garbage(self):
        with self.assertRaises(ValueError):
            to_fraction("not-a-number")

    def test_rejects_non_finite(self):
        with self.assertRaises(ValueError):
            to_fraction(float("inf"))


class TestSequenceContainer(unittest.TestCase):
    def test_requires_at_least_two_terms(self):
        with self.assertRaises(ValueError):
            Sequence.from_raw([1])

    def test_indices_respect_start_index(self):
        seq = Sequence.from_raw([10, 20, 30], start_index=5)
        self.assertEqual(seq.indices(), [5, 6, 7])

    def test_differences(self):
        seq = Sequence.from_raw([1, 4, 9, 16, 25])
        self.assertEqual(seq.differences(1), (F(3), F(5), F(7), F(9)))
        self.assertEqual(seq.differences(2), (F(2), F(2), F(2)))


class TestExprAndSimplify(unittest.TestCase):
    def test_polynomial_expr_eval(self):
        expr = build_polynomial_expr([F(1), F(0), F(1)])  # 1 + n^2
        self.assertEqual(expr.eval(F(3)), F(10))

    def test_simplify_drops_zero_terms(self):
        # (n - n) + (1 * n)  should simplify down to just "n"
        messy = BinOp("+", BinOp("-", Index(), Index()), BinOp("*", Const(F(1)), Index()))
        simplified = simplify_expr(messy)
        self.assertEqual(simplified, Index())


class TestBerlekampMassey(unittest.TestCase):
    def test_fibonacci_recurrence(self):
        fib = [F(x) for x in [1, 1, 2, 3, 5, 8, 13, 21]]
        coeffs = berlekamp_massey(fib)
        self.assertEqual(coeffs, [F(1), F(1)])

    def test_all_zero_sequence(self):
        self.assertEqual(berlekamp_massey([F(0)] * 5), [])


class TestRecurrenceRule(unittest.TestCase):
    def test_generate_and_value_at(self):
        rule = RecurrenceRule(order=2, coeffs=(F(1), F(1)), initial_terms=(F(1), F(1)), start_index=0)
        self.assertEqual(rule.generate(8), [F(x) for x in [1, 1, 2, 3, 5, 8, 13, 21]])
        self.assertEqual(rule.value_at(7), F(21))


class TestEndToEndDetection(unittest.TestCase):
    """One assertion per canonical sequence family; this is the main
    correctness contract of the whole pipeline."""

    def _assert_best_exact(self, values, start_index=0, **kwargs):
        result = analyze(values, start_index=start_index, **kwargs)
        self.assertTrue(result.candidates, f"no candidates found for {values}")
        best = result.best
        self.assertTrue(best.exact, f"best candidate for {values} was not exact: {best.description}")
        ok, err = best.verify(result.sequence)
        self.assertTrue(ok, f"best candidate failed independent re-verification: {best.description}")
        return result

    def test_fibonacci(self):
        r = self._assert_best_exact([1, 1, 2, 3, 5, 8, 13, 21])
        self.assertIn("Fibonacci", r.best.description)

    def test_arithmetic_progression(self):
        self._assert_best_exact([2, 4, 6, 8, 10, 12])

    def test_geometric_progression(self):
        self._assert_best_exact([3, 6, 12, 24, 48, 96])

    def test_squares_dictionary(self):
        self._assert_best_exact([1, 4, 9, 16, 25, 36, 49])

    def test_quadratic_polynomial(self):
        self._assert_best_exact([1, 2, 4, 7, 11, 16, 22, 29])

    def test_constant(self):
        self._assert_best_exact([7, 7, 7, 7])

    def test_alternating_sign(self):
        self._assert_best_exact([1, -4, 9, -16, 25, -36, 49])

    def test_interleaved(self):
        self._assert_best_exact([1, 10, 2, 20, 3, 30, 4, 40])

    def test_fraction_input(self):
        self._assert_best_exact(["1/2", "3/2", "5/2", "7/2"])

    def test_shifted_geometric(self):
        r = self._assert_best_exact([1, 3, 7, 15, 31, 63])  # 2^(n+1) - 1
        self.assertFalse(r.used_deep_surf)  # should short-circuit

    def test_nonhomogeneous_recurrence_needs_deep_surf(self):
        vals = [1, 3, 8, 19, 42, 89, 184, 375, 758]  # a(n) = 2a(n-1) + n
        r = self._assert_best_exact(vals, deep_surf_timeout=2.0)
        self.assertTrue(r.used_deep_surf)
        self.assertIn("nonhomogeneous_recurrence", r.best.source)

    def test_periodic_cycle(self):
        self._assert_best_exact([3, 1, 4, 3, 1, 4, 3, 1, 4, 3], deep_surf_timeout=2.0)

    def test_no_pattern_returns_no_exact_candidate(self):
        result = analyze([3, 17, 2, 99, 41, 6, 88, 5], deep_surf_timeout=1.0)
        if result.candidates:
            self.assertFalse(result.candidates[0].exact)

    def test_predictions_extend_beyond_input(self):
        result = analyze([1, 1, 2, 3, 5, 8, 13, 21])
        preds = result.predictions(3)
        self.assertEqual(preds, [F(34), F(55), F(89)])


class TestCLI(unittest.TestCase):
    def _run_cli(self, args, stdin_text=None):
        cmd = [sys.executable, "-m", "sequence_machine.cli"] + args
        return subprocess.run(
            cmd,
            input=stdin_text,
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parents[1]),
            timeout=20,
        )

    def test_basic_args(self):
        proc = self._run_cli(["1", "1", "2", "3", "5", "8", "13"])
        self.assertEqual(proc.returncode, 0)
        self.assertIn("Fibonacci", proc.stdout)

    def test_json_output(self):
        proc = self._run_cli(["1", "4", "9", "16", "25", "--json"])
        self.assertEqual(proc.returncode, 0)
        payload = json.loads(proc.stdout)
        self.assertIn("best", payload)
        self.assertTrue(payload["best"]["exact"])

    def test_stdin_input(self):
        proc = self._run_cli([], stdin_text="2 4 6 8 10 12\n")
        self.assertEqual(proc.returncode, 0)
        self.assertIn("2*n", proc.stdout.replace(" ", ""))

    def test_too_short_input_errors_cleanly(self):
        proc = self._run_cli(["5"])
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("Error", proc.stderr)


if __name__ == "__main__":
    unittest.main()
