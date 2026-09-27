import argparse

import numpy as np

from analysis.convergence import save_error_history
from src.methods.conditioning import calculate_conditioning, save_conditioning
from src.methods.base import SolverConfig
from src.methods.jacobi import JacobiSolver
from src.methods.metodo_gradiente import GradientConjugadoSolver
from src.methods.metodo_GCS import GradientConjugadoQuadradoeSolver
from src.methods.gauss_elimination import GaussianEliminationSolver
from src.methods.gauss_seidel import  GaussSeidelN2Solver, GaussSeidelSolver

from src.reader.txt_reader import TXTReader


def parse_args():
    parser = argparse.ArgumentParser(
        description="Resolução de sistemas lineares"
    )

    parser.add_argument(
        "--matrix",
        required=True
    )

    parser.add_argument(
        "--vector",
        required=True
    )

    parser.add_argument(
        "--tolerance",
        type=float,
        required=True
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        required=True
    )

    parser.add_argument(
        "--init_guess",
        type=str,
        required=True
    )

    parser.add_argument(
        "--benchmark",
        action="store_true"
    )

    return parser.parse_args()


def main():

    args = parse_args()

    init_guess = np.array(
        [
            float(x)
            for x in args.init_guess.split()
        ],
        dtype=float
    )

    config = SolverConfig(
        tolerance=args.tolerance,
        max_iterations=args.max_iterations,
        initial_guess=init_guess
    )

    reader = TXTReader()

    system = reader.read(
        args.matrix,
        args.vector
    )

    # ==========================================================
    # ELIMINAÇÃO DE GAUSS
    # ==========================================================

    gauss_solver = GaussianEliminationSolver()

    gauss_result = gauss_solver.solve(system)

    print("=" * 60)
    print(gauss_solver.name)

    print(
        f"Tempo PASSO 1: "
        f"{gauss_result.time_step1 * 1000:.12f} ms"
    )

    print(
        f"Tempo PASSO 2: "
        f"{gauss_result.time_step2 * 1000:.12f} ms"
    )

    print(
        f"Tempo PASSO 3: "
        f"{gauss_result.time_step3 * 1000:.12f} ms"
    )

    print(
        f"Solução:\n"
        f"{gauss_result.solution}"
    )
    # CONDICIONAMENTO

    conditioning = calculate_conditioning(system.A)
    save_conditioning(conditioning)

    # for row in conditioning:
    #     print(
    #         f"{row['norm']:25} -> "
    #         f"kappa(A) = {row['kappa']:.6e} | "
    #         f"kappa_W(A) = {row['kappa_weighted']:.6e}"
    #     )
    # ==========================================================
    # MÉTODOS ITERATIVOS
    # ==========================================================

    solvers = [
        JacobiSolver(config),
        GaussSeidelN2Solver(config),
        GaussSeidelSolver(config),
    ]

    # ==========================================================
    # GRADIENTE CONJUGADO / CGS
    # ==========================================================

    if np.allclose(
        system.A,
        system.A.T
    ):
        solvers.append(
            GradientConjugadoSolver(config)
        )
    else:
        solvers.append(
            GradientConjugadoQuadradoeSolver(config)
        )

    result_files = [
        "results/jacobi_error.csv",
        "results/gauss_seidel_n2_error.csv",
        "results/gauss_seidel_error.csv",
        "results/mgc_error.csv",
    ]

    # ==========================================================
    # EXECUÇÃO
    # ==========================================================

    for solver, result_file in zip(
        solvers,
        result_files
    ):

        result = solver.solve(system)

        print("=" * 60)
        print(result.method)

        print(
            f"Convergiu: "
            f"{result.converged}"
        )

        print(
            f"Iterações: "
            f"{result.iterations}"
        )

        print(
            f"Erro: "
            f"{result.error:.6e}"
        )

        print(
            f"Tempo: "
            f"{result.execution_time * 1000:.12f} ms"
        )

        print(
            f"Solução:\n"
            f"{result.solution}"
        )

        save_error_history(
            result.error_history,
            result_file
        )


if __name__ == "__main__":
    main()