import argparse

import numpy as np

from analysis.convergence import save_error_history
from src.methods.gradiente import GradienteSolver
from src.methods.newton import NewtonSolver
from src.methods.newton_modificado import NewtonModificadoSolver
from src.reader.non_lin_inputs import SystemNonLinInputs
from src.methods.jacobi_n2 import JacobiN2Solver
from src.methods.conditioning import calculate_conditioning, calculate_iteration_matrices_conditioning, save_conditioning
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
        required=False
    )

    parser.add_argument(
        "--vector",
        required=False
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
        "--num_runs",
        type=int,
        required=True
    )

    parser.add_argument(
        "--init_guess",
        type=str,
        required=True
    )

    parser.add_argument(
        "--is_linear",
        type=str,
        required=True
    )
    return parser.parse_args()


def main():

    args = parse_args()
    is_linear = args.is_linear.lower() in ("true", "1", "yes", "t")
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
    if (is_linear):

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
        resultados_iteracao = calculate_iteration_matrices_conditioning(system.A)
        for metodo, metricas in resultados_iteracao.items():
            save_conditioning(metricas,f"results/condicionamento_{metodo}.csv")
            

        # ==========================================================
        # MÉTODOS ITERATIVOS
        # ==========================================================

        solvers = [
            JacobiSolver(config),
            JacobiN2Solver(config),
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
            "results/jacobi_n2error.csv",
            "results/gauss_seidel_n2_error.csv",
            "results/gauss_seidel_error.csv",
            "results/mgc_error.csv",
        ]

        # # ==========================================================
        # # EXECUÇÃO
        # # ==========================================================

        # for solver, result_file in zip(
        #     solvers,
        #     result_files
        # ):

        #     result = solver.solve(system)

        #     print("=" * 60)
        #     print(result.method)

        #     print(
        #         f"Convergiu: "
        #         f"{result.converged}"
        #     )

        #     print(
        #         f"Iterações: "
        #         f"{result.iterations}"
        #     )

        #     print(
        #         f"Erro: "
        #         f"{result.error:.6e}"
        #     )

        #     print(
        #         f"Tempo: "
        #         f"{result.execution_time:.6f} s"
        #     )

        #     print(
        #         f"Solução:\n"
        #         f"{result.solution}"
        #     )

        #     save_error_history(
        #         result.error_history,
        #         result_file
        #     )
    else:
        print("=== Executando Modo Não Linear ===")
        system = SystemNonLinInputs()
        solvers = [NewtonSolver(config), NewtonModificadoSolver(config), GradienteSolver(config)]
        result_files = [
            "results/newtom_error.csv",
            "results/newtom_mod_error.csv",
            "results/gradiente.csv"
        ]
        # result = solver.solve(system)
        # print(f"Solução Não Linear: {result.solution}")

    # ==========================================================
    # EXECUÇÃO
    # ==========================================================
    times = args.num_runs

    for solver, result_file in zip(solvers, result_files):

        cpu_times = []
        result = None

        for _ in range(times):
            result = solver.solve(system)

            cpu_times.append(result.execution_time)

        mean_cpu_time = np.mean(cpu_times)

        print("=" * 60)
        print(result.method)

        print(f"Convergiu: {result.converged}")
        print(f"Iterações: {result.iterations}")
        print(f"Erro: {result.error:.6e}")

        print(
            f"Tempo médio de CPU ({times} execuções): "
            f"{mean_cpu_time:.12f} s"
        )

        print(
            f"Tempo médio de CPU ({times} execuções): "
            f"{mean_cpu_time * 1_000_000:.6f} µs"
        )

        print(f"Solução:\n{result.solution}")

        save_error_history(
            result.error_history,
            result_file
        )

if __name__ == "__main__":
    main()