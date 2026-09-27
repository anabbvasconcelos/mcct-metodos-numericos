from __future__ import annotations

import time

import numpy as np


class GaussianEliminationSolver:

    @property
    def name(self) -> str:
        return "Eliminação de Gauss"

    def solve(self, system):

        A = system.A.copy()
        b = system.b.copy()

        n = len(b)

        tempo_passo1 = 0.0
        tempo_passo2 = 0.0

        # ======================================================
        # PASSO 1 + PASSO 2
        # ======================================================

        for e in range(n - 1):

            # --------------------------------------------------
            # PIVOTAMENTO
            # --------------------------------------------------

            pivot_row = e + np.argmax(
                np.abs(A[e:, e])
            )

            if np.isclose(A[pivot_row, e], 0.0):
                raise np.linalg.LinAlgError(
                    "Eliminação interrompida: matriz singular."
                )

            if pivot_row != e:
                A[[e, pivot_row]] = A[[pivot_row, e]]
                b[[e, pivot_row]] = b[[pivot_row, e]]

            # --------------------------------------------------
            # PASSO 1
            # divisor_linha_elemento
            # --------------------------------------------------

            start = time.perf_counter()

            pivot = A[e, e]

            A[e, e:] /= pivot
            b[e] /= pivot

            tempo_passo1 += (
                time.perf_counter() - start
            )

            # --------------------------------------------------
            # PASSO 2
            # --------------------------------------------------

            start = time.perf_counter()

            for i in range(e + 1, n):

                fator = A[i, e]

                A[i, e:] -= fator * A[e, e:]
                b[i] -= fator * b[e]

                A[i, e] = 0.0

            tempo_passo2 += (
                time.perf_counter() - start
            )

        # ======================================================
        # PASSO 3 - RETROSUBSTITUIÇÃO
        # ======================================================

        start = time.perf_counter()

        x = np.zeros(n)

        for i in range(n - 1, -1, -1):

            if np.isclose(A[i, i], 0.0):
                raise np.linalg.LinAlgError(
                    "Matriz singular durante retrosubstituição."
                )

            x[i] = (
                b[i]
                - np.dot(A[i, i + 1:], x[i + 1:])
            ) / A[i, i]

        tempo_passo3 = (
            time.perf_counter() - start
        )

        return GaussianResult(
            solution=x,
            triangular_matrix=A,
            time_step1=tempo_passo1,
            time_step2=tempo_passo2,
            time_step3=tempo_passo3,
        )


class GaussianResult:

    def __init__(
        self,
        solution: np.ndarray,
        triangular_matrix: np.ndarray,
        time_step1: float,
        time_step2: float,
        time_step3: float,
    ):
        self.solution = solution
        self.triangular_matrix = triangular_matrix
        self.time_step1 = time_step1
        self.time_step2 = time_step2
        self.time_step3 = time_step3