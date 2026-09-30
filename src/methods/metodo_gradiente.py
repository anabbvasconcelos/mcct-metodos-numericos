
import numpy as np
import time

from src.methods.base import IterativeResult, IterativeSolver

class GradientConjugadoSolver(IterativeSolver):
    @property
    def name(self) -> str:
        return "Gradiente Conjugado (M2)"

    def solve(self,system, x0= None):
        A = system.A
        b = system.b
        x = self.config.initial_guess
        max_iteractions = self.config.max_iterations
        tol = self.config.tolerance
        r = b - np.dot(A, x)
        p = r.copy()

        error_history = []
        start = time.process_time()
        for iteration in range(1,max_iteractions + 1):
            Ap = np.dot(A, p)
            pAp = np.dot(p.T, Ap)

            if abs(pAp) < 1e-15:

                return IterativeResult(
                    method=self.name,
                    solution=x,
                    iterations=max_iteractions,
                    error=self.residual(system, x),
                    converged=False,
                    execution_time=time.process_time() - start,
                    error_history=error_history
                )

            alpha = np.dot(r.T, r) / pAp
            x_new = x + alpha * p
            r_new = r - alpha * Ap

            error = self.residual(system, x_new)
            error_history.append(error)
            # Critério de paragem: norma infinita da diferença entre iterações
            if np.max(np.abs(x_new - x)) < tol:
                    return IterativeResult(
                        method=self.name,
                        solution=x_new,
                        iterations=iteration,
                        error=error,
                        converged=True,
                        execution_time=time.process_time() - start,
                        error_history=error_history
                    )

            beta = np.dot(r_new.T, r_new) / np.dot(r.T, r)
            p = r_new + beta * p

            x = x_new
            r = r_new

            return IterativeResult(
                method=self.name,
                solution=x,
                iterations=max_iteractions,
                error=self.residual(system, x),
                converged=False,
                execution_time=time.process_time() - start,
                error_history=error_history
            )
