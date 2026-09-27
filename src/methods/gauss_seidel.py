from __future__ import annotations

import time

import numpy as np

from src.methods.base import IterativeResult, IterativeSolver


class GaussSeidelSolver(IterativeSolver):

    @property
    def name(self) -> str:
        return "Gauss-Seidel"

    def solve(self, system, x0=None):

        A = system.A
        b = system.b

        x = (
            self.config.initial_guess.copy()
            if x0 is None
            else np.array(x0, dtype=float)
        )

        max_iterations = self.config.max_iterations
        tolerance = self.config.tolerance

        error_history = []

        start = time.perf_counter()

        for iteration in range(
            1,
            max_iterations + 1
        ):

            x_old = x.copy()

            for i in range(len(b)):

                sum_before = np.dot(A[i, :i],x[:i])

                sum_after = np.dot( A[i, i + 1:], x_old[i + 1:])

                x[i] = (b[i] - sum_before - sum_after ) / A[i, i]

            error = self.residual(x_old, x)

            error_history.append(error)

            step_error = np.max(
                np.abs(x - x_old)
            )

            if step_error < tolerance:

                return IterativeResult(
                    method=self.name,
                    solution=x,
                    iterations=iteration,
                    error=error,
                    converged=True,
                    execution_time=(
                        time.perf_counter() - start
                    ),
                    error_history=error_history,
                )

        return IterativeResult(
            method=self.name,
            solution=x,
            iterations=max_iterations,
            error=self.residual(system, x),
            converged=False,
            execution_time=(
                time.perf_counter() - start
            ),
            error_history=error_history,
        )


class GaussSeidelN2Solver(IterativeSolver):

    @property
    def name(self) -> str:
        return "Gauss-Seidel N=2"

    def solve(self, system, x0=None):

        A = system.A
        b = system.b

        x = (
            self.config.initial_guess.copy()
            if x0 is None
            else np.array(x0, dtype=float)
        )

        error_history = []

        start = time.perf_counter()

        for _ in range(2):

            x_old = x.copy()

            for i in range(len(b)):

                sum_before = np.dot(
                    A[i, :i],
                    x[:i]
                )

                sum_after = np.dot(
                    A[i, i + 1:],
                    x_old[i + 1:]
                )

                x[i] = (
                    b[i]
                    - sum_before
                    - sum_after
                ) / A[i, i]

            error = self.residual(x_old, x)

            error_history.append(error)

        return IterativeResult(
            method=self.name,
            solution=x,
            iterations=2,
            error=self.residual(x_old, x),
            converged=False,
            execution_time=(
                time.perf_counter() - start
            ),
            error_history=error_history,
        )