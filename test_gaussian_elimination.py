"""Tests with known solutions, pivoting cases, and invalid inputs."""

import math
import random
import unittest

from gaussian_elimination import gaussian_elimination


class GaussianEliminationTests(unittest.TestCase):
    def assert_solution(self, A, b, expected):
        actual = gaussian_elimination(A, b)
        self.assertEqual(len(actual), len(expected))
        for computed, target in zip(actual, expected):
            self.assertAlmostEqual(computed, target, delta=1e-10 * max(1, abs(target)))
        # Check the original equations independently of the elimination steps.
        for row, rhs in zip(A, b):
            lhs = sum(value * x for value, x in zip(row, actual))
            scale = abs(rhs) + sum(abs(value * x) for value, x in zip(row, actual))
            self.assertLessEqual(abs(lhs - rhs), 1e-12 * scale)

    def test_worked_example(self):
        self.assert_solution(
            [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]],
            [8, -11, -3],
            [2, 3, -1],
        )

    def test_zero_leading_entry_requires_row_swap(self):
        self.assert_solution([[0, 2], [1, 3]], [4, 7], [1, 2])

    def test_row_swap_after_first_column(self):
        self.assert_solution([[1, 1, 0], [1, 1, 1], [0, 1, 1]], [3, 6, 5], [1, 2, 3])

    def test_tiny_leading_entry(self):
        self.assert_solution([[1e-20, 1], [1, 1]], [1, 2], [1, 1])

    def test_one_equation(self):
        self.assert_solution([[4]], [10], [2.5])

    def test_inputs_are_not_modified(self):
        A = [[0, 2], [1, 3]]
        b = [4, 7]
        original_A = [row[:] for row in A]
        original_b = b[:]
        gaussian_elimination(A, b)
        self.assertEqual(A, original_A)
        self.assertEqual(b, original_b)

    def test_tuple_inputs(self):
        self.assert_solution(((2, 0), (0, 4)), (6, 8), [3, 2])

    def test_singular_systems(self):
        for A, b in [
            ([[1, 2], [2, 4]], [3, 6]),  # Infinitely many solutions.
            ([[1, 2], [2, 4]], [3, 7]),  # No solution.
            ([[0, 0], [0, 0]], [0, 0]),
            ([[0]], [1]),
        ]:
            with self.subTest(A=A, b=b), self.assertRaises(ValueError):
                gaussian_elimination(A, b)

    def test_uniform_scaling(self):
        for factor in [1e-150, 1e150]:
            with self.subTest(factor=factor):
                A = [[factor * value for value in row] for row in [[2, 1], [1, 3]]]
                b = [factor * value for value in [5, 5]]
                self.assert_solution(A, b, [2, 1])

    def test_tolerance_is_configurable(self):
        A, b = [[1, 0], [0, 1e-14]], [1, 2e-14]
        with self.assertRaises(ValueError):
            gaussian_elimination(A, b)
        self.assertEqual(gaussian_elimination(A, b, rtol=0), [1, 2])

    def test_invalid_dimensions(self):
        for A, b in [([], []), ([[1, 2]], [1]), ([[1, 2], [3]], [1, 2]), ([[1]], [])]:
            with self.subTest(A=A, b=b), self.assertRaises(ValueError):
                gaussian_elimination(A, b)

    def test_invalid_entries(self):
        for value in [math.nan, math.inf, -math.inf, 1j, "not a number"]:
            for A, b in [([[value]], [1]), ([[1]], [value])]:
                with self.subTest(A=A, b=b), self.assertRaises(ValueError):
                    gaussian_elimination(A, b)

    def test_invalid_tolerance(self):
        for rtol in [-1, 1, math.inf, math.nan]:
            with self.subTest(rtol=rtol), self.assertRaises(ValueError):
                gaussian_elimination([[1]], [1], rtol=rtol)

    def test_overflow_is_reported(self):
        for A, b in [
            ([[1e308, 1e308], [-1e308, 1e308]], [1, 1]),
            ([[1e-308]], [1e308]),
        ]:
            with self.subTest(A=A, b=b), self.assertRaises(ArithmeticError):
                gaussian_elimination(A, b)

    def test_systems_with_constructed_solutions(self):
        rng = random.Random(2026)
        for n in range(1, 9):
            for trial in range(5):
                # Strict diagonal dominance ensures these matrices are nonsingular.
                A = [[rng.randint(-5, 5) for _ in range(n)] for _ in range(n)]
                for i in range(n):
                    A[i][i] = sum(abs(A[i][j]) for j in range(n) if j != i) + 1
                expected = [rng.randint(-16, 16) / 8 for _ in range(n)]
                b = [sum(a * x for a, x in zip(row, expected)) for row in A]
                with self.subTest(n=n, trial=trial):
                    self.assert_solution(A, b, expected)


if __name__ == "__main__":
    unittest.main()
