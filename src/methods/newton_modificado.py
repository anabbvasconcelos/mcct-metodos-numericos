from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np
import time

from src.methods.base import IterativeResult, IterativeSolver
from src.reader.txt_reader import Inputs

class NewtonModificadoSolver(IterativeSolver):

    @property
    def name(self) -> str:
        return "Newton Modificado"

    def solve(self, system: Inputs, x0: np.ndarray | None = None) -> IterativeResult:
        start = time.perf_counter()
        
        X = self.config.initial_guess
        error_history = [] 
        iterations = 0
        final_error = float('inf')

        # Calcula a Jacobiana apenas uma vez no início (Newton Modificado)
        J0 = system.evaluate_jacobian(X)

        for k in range(1, self.config.max_iterations + 1):
            iterations = k
            F_val = system.evaluate_f(X)

            try:
                # Resolve J0 * delta_X = -F(X) usando a Jacobiana fixa
                delta_X = np.linalg.solve(J0, -F_val)
            except np.linalg.LinAlgError:
                raise ValueError("Matriz Jacobiana singular encontrada.")

            X = X + delta_X
            final_error = float(np.max(np.abs(delta_X)))

            if error_history is not None:
                error_history.append(final_error)

            if final_error < self.config.tolerance:
                return IterativeResult(
                    method=self.name,
                    solution=X,
                    iterations=iterations,
                    error=final_error,
                    converged=True,
                    execution_time=time.perf_counter() - start,
                    error_history=error_history
                )

        return IterativeResult(
            method=self.name,
            solution=X,
            iterations=iterations,
            error=final_error,
            converged=False,
            execution_time=time.perf_counter() - start,
            error_history=error_history
        )

