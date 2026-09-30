import numpy as np
import time

from src.methods.base import IterativeResult, IterativeSolver


class GradienteSolver(IterativeSolver):

    @property
    def name(self) -> str:
        return "Gradiente"

    def solve(self, system, x0=None) -> IterativeResult:

        start = time.process_time()

        X = (
            x0.copy()
            if x0 is not None
            else self.config.initial_guess.copy()
        )

        error_history = []
        final_error = float("inf")

        for k in range(1, self.config.max_iterations + 1):

            # f(x_k)
            F = system.evaluate_f(X)

            # W(x_k)
            W = system.evaluate_jacobian(X)

            # g_k = W(x_k)^T f(x_k)
            g = W.T @ F

            # W(x_k) g_k
            Wg = W @ g

            # Numerador:
            # [W^T f]^T [W^T f]
            numerator = np.dot(g, g)

            # Denominador:
            # [W W^T f]^T [W W^T f]
            denominator = np.dot(Wg, Wg)

            # Proteção contra overflow
            if not np.isfinite(numerator) or not np.isfinite(denominator):
                return IterativeResult(
                    method=self.name,
                    solution=X,
                    iterations=k,
                    error=float("inf"),
                    converged=False,
                    execution_time=time.process_time() - start,
                    error_history=error_history
                )

            # Proteção contra divisão por zero
            if abs(denominator) < 1e-14:
                return IterativeResult(
                    method=self.name,
                    solution=X,
                    iterations=k,
                    error=final_error,
                    converged=False,
                    execution_time=time.process_time() - start,
                    error_history=error_history
                )

            # Fator da fórmula
            lambda_k = numerator / denominator

            # x_(k+1) = x_k - lambda_k * g_k
            X_new = X - lambda_k * g
            # Critério de parada:
            # max |x_(k+1) - x_k|
            final_error = float(
                np.max(np.abs(X_new - X))
            )

            error_history.append(final_error)

            X = X_new

            if final_error < self.config.tolerance:
                return IterativeResult(
                    method=self.name,
                    solution=X,
                    iterations=k,
                    error=final_error,
                    converged=True,
                    execution_time=time.process_time() - start,
                    error_history=error_history
                )

        return IterativeResult(
            method=self.name,
            solution=X,
            iterations=self.config.max_iterations,
            error=final_error,
            converged=False,
            execution_time=time.process_time() - start,
            error_history=error_history
        )