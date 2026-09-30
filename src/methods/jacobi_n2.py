import numpy as np
import time

from src.methods.base import IterativeSolver, IterativeResult


class JacobiN2Solver(IterativeSolver):

    @property
    def name(self) -> str:
        return "Jacobi de 2ª Ordem (M2)"

    def solve(self, system, x0=None):

        A = system.A
        b = system.b
        x = self.config.initial_guess

        D = np.diag(A)
        R = A - np.diag(D)

        error_history = []
        start = time.process_time()

        for iteration in range(1, self.config.max_iterations + 1):

            # Primeiro sub-passo de Jacobi
            x_mid = (b - R @ x) / D
            
            # Segundo sub-passo de Jacobi (compondo a 2ª ordem)
            x_new = (b - R @ x_mid) / D

            error = self.residual(x, x_new)
            error_history.append(error)

            if error < self.config.tolerance:
                return IterativeResult(
                    method=self.name,
                    solution=x_new,
                    iterations=iteration,
                    error=error,
                    converged=True,
                    execution_time=time.process_time() - start,
                    error_history=error_history
                )

            x = x_new

        return IterativeResult(
            method=self.name,
            solution=x,
            iterations=self.config.max_iterations,
            error=self.residual(x, x),  # ou o cálculo de resíduo adequado
            converged=False,
            execution_time=time.process_time() - start,
            error_history=error_history
        )