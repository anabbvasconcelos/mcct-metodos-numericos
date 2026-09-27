from __future__ import annotations
import csv
from pathlib import Path
import numpy as np


def norm_1(A: np.ndarray) -> float:
    """
    Norma 1:
    máximo da soma dos valores absolutos das colunas.
    """
    return float(np.max(np.sum(np.abs(A), axis=0)))


def norm_inf(A: np.ndarray) -> float:
    """
    Norma infinito:
    máximo da soma dos valores absolutos das linhas.
    """
    return float(np.max(np.sum(np.abs(A), axis=1)))


def norm_frobenius(A: np.ndarray) -> float:
    """
    Norma de Frobenius.
    """
    return float(np.sqrt(np.sum(A * A)))


def norm_max(A: np.ndarray) -> float:
    """
    Máximo valor absoluto dos elementos.
    """
    return float(np.max(np.abs(A)))


def norm_2(A: np.ndarray) -> float:
    """
    Norma espectral 2 utilizando o método da potência
    sobre A^T A.

    Equivalente à implementação C++ norma2().
    """
    n = A.shape[0]

    ata = A.T @ A

    x = np.full(n, 1.0 / np.sqrt(n), dtype=float)

    lambda_value = 0.0

    for _ in range(10_000):
        # y = AtA * x
        y = ata @ x

        norm_y = np.sqrt(np.sum(y * y))

        if norm_y < 1e-18:
            break

        y /= norm_y

        # Rayleigh quotient
        aty = ata @ y

        numerator = np.dot(y, aty)
        denominator = np.dot(y, y)

        lambda_new = (
            numerator / denominator
            if denominator > 0.0
            else 0.0
        )

        if abs(lambda_new - lambda_value) < 1e-12:
            lambda_value = lambda_new
            break

        lambda_value = lambda_new
        x = y.copy()

    return float(np.sqrt(lambda_value))


NORMS = [
    ("Euclidiana (2)", norm_2),
    ("Soma max colunas (1)", norm_1),
    ("Soma max linhas (inf)", norm_inf),
    ("Frobenius", norm_frobenius),
    ("Maximo", norm_max),
]


def invert_matrix(A: np.ndarray) -> np.ndarray:
    """
    Inverte uma matriz utilizando Gauss-Jordan
    com pivoteamento parcial.

    Equivalente à função inverter() do C++.
    """
    A = np.asarray(A, dtype=float)

    n = A.shape[0]

    # [A | I]
    aug = np.hstack(
        (
            A.copy(),
            np.eye(n, dtype=float),
        )
    )

    for i in range(n):

        # Pivoteamento parcial
        pivot_row = i
        max_abs = abs(aug[i, i])

        for k in range(i + 1, n):
            if abs(aug[k, i]) > max_abs:
                max_abs = abs(aug[k, i])
                pivot_row = k

        if max_abs < 1e-15:
            raise np.linalg.LinAlgError(
                "Matriz singular, não foi possível inverter."
            )

        # Troca das linhas
        if pivot_row != i:
            aug[[i, pivot_row]] = aug[[pivot_row, i]]

        # Normaliza linha do pivô
        pivot = aug[i, i]
        aug[i, :] /= pivot

        # Elimina outras linhas
        for k in range(n):
            if k == i:
                continue

            factor = aug[k, i]

            if abs(factor) < 1e-18:
                continue

            aug[k, :] -= factor * aug[i, :]

    # Parte direita = inversa
    return aug[:, n:]


def calculate_weights(A: np.ndarray) -> np.ndarray:
    """
    Pesos definidos pela diagonal de A.

    Equivalente a:

        w[i] = fabs(A[i * n + i]);

        if (w[i] < 1e-12)
            w[i] = 1.0;
    """
    weights = np.abs(np.diag(A)).astype(float)

    weights[weights < 1e-12] = 1.0

    return weights


def weighted_matrix(
    A: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """
    Calcula:

        B = W A W^-1

    ou, elemento a elemento:

        B[i,j] = w[i] * A[i,j] / w[j]

    Equivalente à normaPonderada() do C++.
    """
    return (
        weights[:, np.newaxis]
        * A
        / weights[np.newaxis, :]
    )


def weighted_norm(
    A: np.ndarray,
    weights: np.ndarray,
    norm_func,
) -> float:
    """
    Calcula ||A||_W = ||W A W^-1||.
    """
    B = weighted_matrix(A, weights)

    return norm_func(B)


def calculate_conditioning(
    A: np.ndarray,
) -> list[dict[str, float | str]]:
    """
    Calcula o condicionamento da matriz A para
    todas as normas utilizadas pelo programa C++.

    Retorna:

        Norma
        ||A||
        ||A^-1||
        kappa(A)
        ||A||_W
        ||A^-1||_W
        kappa_W(A)
    """

    A = np.asarray(A, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(
            "A matriz A deve ser quadrada."
        )

    # 1) Inverter A
    A_inv = invert_matrix(A)

    # 2) Pesos = diagonal de A
    weights = calculate_weights(A)

    results = []

    # 3) Calcular todas as normas
    for name, norm_func in NORMS:

        # ||A||
        norm_A = norm_func(A)

        # ||A^-1||
        norm_A_inv = norm_func(A_inv)

        # kappa(A) = ||A|| * ||A^-1||
        kappa = norm_A * norm_A_inv

        # ||A||_W
        norm_A_weighted = weighted_norm(
            A,
            weights,
            norm_func,
        )

        # ||A^-1||_W
        norm_A_inv_weighted = weighted_norm(
            A_inv,
            weights,
            norm_func,
        )

        # kappa_W(A)
        kappa_weighted = (
            norm_A_weighted
            * norm_A_inv_weighted
        )

        results.append(
            {
                "norm": name,
                "norm_A": norm_A,
                "norm_A_inv": norm_A_inv,
                "kappa": kappa,
                "norm_A_weighted": norm_A_weighted,
                "norm_A_inv_weighted": norm_A_inv_weighted,
                "kappa_weighted": kappa_weighted,
            }
        )

    return results



def save_conditioning(
    results: list[dict],
    path: str = "results/condicionamento_A.csv",
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter=";")

        writer.writerow([
            "Norma",
            "||A||",
            "||A^-1||",
            "kappa(A)",
            "||A||_W",
            "||A^-1||_W",
            "kappa_W(A)",
        ])

        for row in results:
            writer.writerow([
                row["norm"],
                f"{row['norm_A']:.6e}",
                f"{row['norm_A_inv']:.6e}",
                f"{row['kappa']:.6e}",
                f"{row['norm_A_weighted']:.6e}",
                f"{row['norm_A_inv_weighted']:.6e}",
                f"{row['kappa_weighted']:.6e}",
            ])