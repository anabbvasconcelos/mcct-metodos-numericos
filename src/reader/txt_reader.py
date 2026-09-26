from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Inputs:
    A: np.ndarray
    b: np.ndarray
    
    @property
    def size(self) -> int:
        return self.A.shape[0]


class TXTReader:

    @staticmethod
    def _read_numbers(path: str | Path) -> list[float]:
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {path}")

        content = path.read_text(encoding="utf-8")

        values = []

        for line in content.splitlines():
            line = line.strip()

            if not line:
                continue

            values.extend(
                float(value)
                for value in line.replace(",", ".").split()
            )

        return values

    def read_matrix(self, path: str | Path) -> np.ndarray:
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {path}")

        rows = []

        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()

            if not line:
                continue

            row = [
                float(value)
                for value in line.replace(",", ".").split()
            ]

            rows.append(row)

        if not rows:
            raise ValueError("Arquivo da matriz está vazio.")

        column_count = len(rows[0])

        if any(len(row) != column_count for row in rows):
            raise ValueError(
                "Todas as linhas da matriz devem possuir "
                "a mesma quantidade de valores."
            )

        matrix = np.array(rows, dtype=float)

        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(
                f"A matriz deve ser quadrada. "
                f"Dimensão encontrada: {matrix.shape}"
            )

        return matrix

    def read_vector(self, path: str | Path) -> np.ndarray:
        values = self._read_numbers(path)

        if not values:
            raise ValueError("Arquivo do vetor está vazio.")

        return np.array(values, dtype=float)

    def read(
        self,
        matrix_path: str | Path,
        vector_path: str | Path
    ) -> Inputs:

        A = self.read_matrix(matrix_path)
        b = self.read_vector(vector_path)

        if A.shape[0] != b.shape[0]:
            raise ValueError(
                f"Dimensões incompatíveis: "
                f"A possui {A.shape[0]} linhas e "
                f"b possui {b.shape[0]} elementos."
            )

        return Inputs(A=A, b=b)