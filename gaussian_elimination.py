"""Solve a real square linear system using Gaussian elimination.

The implementation uses partial pivoting and only the Python standard library.
Run this file directly to see a worked example.
"""

from collections.abc import Sequence
from math import isfinite


def gaussian_elimination(
    A: Sequence[Sequence[float]],
    b: Sequence[float],
    *,
    rtol: float = 1e-12,
) -> list[float]:
    """Return the solution of Ax = b without modifying A or b.

    A must be a nonempty square matrix, and b must have the same number
    of entries as A has rows. Entries are converted to finite real floats.

    At each step, partial pivoting selects the largest absolute entry in
    the remaining part of the current column. A pivot is rejected when
    its magnitude is at most rtol * max(abs(A[i][j])). Set rtol=0 to
    reject only exactly zero pivots.

    Raises:
        ValueError: Invalid dimensions or values, invalid rtol, or a
            singular or numerically singular matrix at the chosen tolerance.
        ArithmeticError: A calculation produces a non-finite value.

    Complexity: O(n**3) time and O(n**2) additional storage.
    """
    n = len(A)
    if n == 0 or any(len(row) != n for row in A):
        raise ValueError("A must be a nonempty square matrix.")
    if len(b) != n:
        raise ValueError("b must have one entry for each row of A.")
    if not isfinite(rtol) or not 0 <= rtol < 1:
        raise ValueError("rtol must be finite and satisfy 0 <= rtol < 1.")

    # Copy the inputs into the augmented matrix [A | b].
    try:
        augmented = [
            [float(value) for value in row] + [float(rhs)]
            for row, rhs in zip(A, b)
        ]
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("A and b must contain finite real values.") from error
    if any(not isfinite(value) for row in augmented for value in row):
        raise ValueError("A and b must contain finite real values.")

    scale = max(abs(value) for row in augmented for value in row[:-1])
    tolerance = rtol * scale

    # Forward elimination: produce an upper triangular system.
    for k in range(n):
        pivot_row = max(range(k, n), key=lambda i: abs(augmented[i][k]))
        if abs(augmented[pivot_row][k]) <= tolerance:
            raise ValueError(
                "A is singular or numerically singular at the chosen rtol."
            )
        augmented[k], augmented[pivot_row] = augmented[pivot_row], augmented[k]

        for i in range(k + 1, n):
            factor = augmented[i][k] / augmented[k][k]
            augmented[i][k] = 0.0
            for j in range(k + 1, n + 1):
                augmented[i][j] -= factor * augmented[k][j]
            if any(not isfinite(value) for value in augmented[i][k + 1 :]):
                raise ArithmeticError("Non-finite intermediate value; rescale A and b.")

    # Back substitution: solve from the last equation upwards.
    solution = [0.0] * n
    for i in range(n - 1, -1, -1):
        known_terms = sum(augmented[i][j] * solution[j] for j in range(i + 1, n))
        solution[i] = (augmented[i][n] - known_terms) / augmented[i][i]
        if not isfinite(solution[i]):
            raise ArithmeticError("Non-finite solution value; rescale A and b.")

    return solution


if __name__ == "__main__":
    coefficients = [
        [2, 1, -1],
        [-3, -1, 2],
        [-2, 1, 2],
    ]
    right_hand_side = [8, -11, -3]
    x = gaussian_elimination(coefficients, right_hand_side)

    print("Solving Ax = b using Gaussian elimination with partial pivoting.")
    for index, value in enumerate(x, start=1):
        print(f"x{index} = {value:.6g}")
