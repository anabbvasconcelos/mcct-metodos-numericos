import argparse

import numpy as np

from analysis.convergence import save_error_history
from src.methods.metodo_GCS import GradientConjugadoQuadradoeSolver
from src.methods.metodo_gradiente import GradientConjugadoSolver
from src.reader.txt_reader import TXTReader
from src.methods.base import SolverConfig
from src.methods.jacobi import JacobiSolver


def parse_args():
    parser = argparse.ArgumentParser(
        description="Resolução iterativa de sistemas lineares"
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
        [float(x) for x in args.init_guess.split()],
        dtype=float
    )
    
    config = SolverConfig(
        tolerance=args.tolerance,
        max_iterations=args.max_iterations,
        initial_guess= init_guess
    )

    reader = TXTReader()

    system = reader.read(
        args.matrix,
        args.vector
    )
    solvers = [
        JacobiSolver(config)
    ]

    if np.allclose(system.A, system.A.T):
        solvers.append(GradientConjugadoSolver(config))
    else:
        solvers.append(GradientConjugadoQuadradoeSolver(config))

    i = 0
    for solver in solvers:
        names = ["results/jacobi_error.csv","results/mgc_error.csv"]
        result = solver.solve(system)
        print("solve", solver.name)
        save_error_history(
            result.error_history,
            names[i]
        )
        print("=" * 60)
        print(result.method)
        print(f"Convergiu: {result.converged}")
        print(f"Iterações: {result.iterations}")
        print(f"Erro: {result.error:.6e}")
        print(f"Tempo: {result.execution_time* 1_000_000:.12f} ms")
        print(f"Solução:\n{result.solution}")
        i +=1


if __name__ == "__main__":
    main()