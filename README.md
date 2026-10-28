# Gaussian elimination in Python

A readable implementation of **Gaussian elimination with partial pivoting** for solving a real square linear system

$$Ax=b.$$

The algorithm is implemented directly using Python lists and the standard library. It includes input validation, an example, and tests. No dependencies need to be installed.

## Quick start

Requires **Python 3.10 or later**. From this directory, run:

```bash
python3 gaussian_elimination.py
```

Output:

```text
Solving Ax = b using Gaussian elimination with partial pivoting.
x1 = 2
x2 = 3
x3 = -1
```

To solve another system, import the function:

```python
from gaussian_elimination import gaussian_elimination

A = [
    [2, 1, -1],
    [-3, -1, 2],
    [-2, 1, 2],
]
b = [8, -11, -3]

x = gaussian_elimination(A, b)
print(x)  # Approximately [2.0, 3.0, -1.0]
```

This example represents the equations

$$
\begin{aligned}
2x_1+x_2-x_3&=8,\\
-3x_1-x_2+2x_3&=-11,\\
-2x_1+x_2+2x_3&=-3.
\end{aligned}
$$

Both `A` and `b` are left unchanged.

## How the algorithm works

Start with the augmented matrix $[A\mid b]$.

1. **Choose a pivot.** In column $k$, find the row with the largest absolute entry among rows $k,\ldots,n-1$. Swap it with row $k$, including the right-hand side.
2. **Eliminate below the pivot.** For each row $i>k$, use the multiplier $m_{ik}=a_{ik}/a_{kk}$ and perform $R_i\leftarrow R_i-m_{ik}R_k$.
3. **Repeat** until the coefficient matrix is upper triangular, giving a system $Ux=c$.
4. **Back substitute.** Starting at the last row, compute

$$
x_i=\frac{c_i-\sum_{j=i+1}^{n-1}u_{ij}x_j}{u_{ii}}.
$$

Partial pivoting avoids dividing by a zero pivot when a suitable row is available and reduces the risk of large elimination multipliers. It does not guarantee an accurate solution for every matrix.

## Inputs, tolerance, and limitations

`gaussian_elimination(A, b, *, rtol=1e-12)` returns a list of floats.

- `A` must be a nonempty square matrix and `b` must have one entry per row.
- Entries are converted to finite real floating-point numbers.
- At every elimination step, a pivot is rejected if its magnitude is at most `rtol * scale`, where `scale` is the largest absolute entry of the **original coefficient matrix**. The right-hand side does not enter this scale.
- Singular systems raise `ValueError`, whether they have no solution or infinitely many solutions. This implementation does not distinguish those cases.
- A nonsingular matrix can also be rejected if a pivot is small relative to that scale. The tolerance is a practical screening rule, not a condition-number estimate.
- `rtol=0` rejects only exactly zero pivots; it does not remove floating-point error. Non-finite intermediate or solution values raise `ArithmeticError`.
- Rectangular systems, complex entries, symbolic calculations, and exact rational arithmetic are outside this implementation's scope.

This is an educational implementation. Floating-point rounding, ill-conditioning, and extreme scaling can affect the result. For demanding numerical work, use an established numerical linear algebra library.

## Complexity

For an $n\times n$ matrix, the algorithm takes $O(n^3)$ time. The augmented copy uses $O(n^2)$ additional storage. Back substitution takes $O(n^2)$ time.

## Tests

```bash
python3 -m unittest -v
```

The tests cover known solutions, required row swaps, a tiny initial pivot, singular systems, invalid inputs, tolerance behavior, uniform scaling, overflow, and preservation of the inputs. They also check 40 deterministically generated systems against constructed solutions and the original equations.

## Files

- `gaussian_elimination.py`: implementation and runnable example.
- `test_gaussian_elimination.py`: standard-library unit tests.
- `README.md`: usage and mathematical explanation.
