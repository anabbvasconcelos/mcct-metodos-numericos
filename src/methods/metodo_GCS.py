import numpy as np
import time

from src.methods.base import IterativeResult, IterativeSolver
class GradientConjugadoQuadradoeSolver(IterativeSolver):
    @property
    def name(self) -> str:
        return "Gradiente Conjugado Quadrado (M3)"
    
    def solve(self,system, x0= None):
        A = system.A
        b = system.b
        x = self.config.initial_guess
        max_iteractions = self.config.max_iterations
        tol = self.config.tolerance
        # n = len(b)
        # x = np.zeros(n)
        r = b - np.dot(A, x)


        # Se o vetor inicial já for solução
        if np.max(np.abs(r)) < tol:
            return x, 0, True

        r_tilde = r.copy() # vetor sombra inicial
        p = r.copy()
        u = r.copy()
        rho_prev = np.dot(r_tilde.T, r)

        error_history = []
        start = time.process_time()
        for k in range(max_iteractions):
            # Evitar divisão por zero se rho_prev for quase nulo (reinicia o vetor sombra)
            if abs(rho_prev) < 1e-14:
                r_tilde = r.copy()
                rho_prev = np.dot(r_tilde.T, r)
                if abs(rho_prev) < 1e-14:
                    return IterativeResult(
                        method=self.name,
                        solution=x,
                        iterations=k + 1,
                        error=self.residual(system, x),
                        converged=False,
                        execution_time=time.process_time() - start,
                        error_history=error_history
                    )
                    # return x, k + 1, False

            v = np.dot(A, p)
            rtv = np.dot(r_tilde.T, v)

            # Evitar divisão por zero se r_tilde.T @ v for quase nulo
            if abs(rtv) < 1e-14:
                r_tilde = r.copy()
                rtv = np.dot(r_tilde.T, v)
                if abs(rtv) < 1e-14:
                    return IterativeResult(
                        method=self.name,
                        solution=x,
                        iterations=k + 1,
                        error=self.residual(system, x),
                        converged=False,
                        execution_time=time.process_time() - start,
                        error_history=error_history
                    )

            alpha = rho_prev / rtv
            q = u - alpha * v
            d = u + q

            x_new = x + alpha * d
            Ad = np.dot(A, d)
            r_new = r - alpha * Ad
            
            error = self.residual(x, x_new)
            error_history.append(error)
            # Critério de paragem: norma infinita da diferença entre iterações
            if np.max(np.abs(x_new - x)) < tol:
                    return IterativeResult(
                        method=self.name,
                        solution=x_new,
                        iterations=k + 1,
                        error=error,
                        converged=True,
                        execution_time=time.process_time() - start,
                        error_history=error_history
                    )

            rho_new = np.dot(r_tilde.T, r_new)
            beta = rho_new / rho_prev

            u_new = r_new + beta * q
            p_new = u_new + beta * (q + beta * p)

            x = x_new
            r = r_new
            u = u_new
            p = p_new
            rho_prev = rho_new

        return IterativeResult(
            method=self.name,
            solution=x,
            iterations=max_iteractions,
            converged=False,
            execution_time=time.process_time() - start,
            error_history=error_history
        )


def gradiente_conjugado_quadrado(A, b, tol=1e-4, max_iter=10000):
    n = len(b)
    x = np.zeros(n)
    r = b - np.dot(A, x)

    # Se o vetor inicial já for solução
    if np.max(np.abs(r)) < tol:
        return x, 0, True

    r_tilde = r.copy() # vetor sombra inicial
    p = r.copy()
    u = r.copy()
    rho_prev = np.dot(r_tilde.T, r)

    for k in range(max_iter):
        # Evitar divisão por zero se rho_prev for quase nulo (reinicia o vetor sombra)
        if abs(rho_prev) < 1e-14:
            r_tilde = r.copy()
            rho_prev = np.dot(r_tilde.T, r)
            if abs(rho_prev) < 1e-14:
                return x, k + 1, False

        v = np.dot(A, p)
        rtv = np.dot(r_tilde.T, v)

        # Evitar divisão por zero se r_tilde.T @ v for quase nulo
        if abs(rtv) < 1e-14:
            r_tilde = r.copy()
            rtv = np.dot(r_tilde.T, v)
            if abs(rtv) < 1e-14:
                return x, k + 1, False

        alpha = rho_prev / rtv
        q = u - alpha * v
        d = u + q

        x_new = x + alpha * d
        Ad = np.dot(A, d)
        r_new = r - alpha * Ad

        # Critério de paragem: norma infinita da diferença entre iterações
        if np.max(np.abs(x_new - x)) < tol:
            return x_new, k + 1, True

        rho_new = np.dot(r_tilde.T, r_new)
        beta = rho_new / rho_prev

        u_new = r_new + beta * q
        p_new = u_new + beta * (q + beta * p)

        x = x_new
        r = r_new
        u = u_new
        p = p_new
        rho_prev = rho_new

    return x, max_iter, False

# def main():
#     print("A carregar ficheiros de dados...")
#     try:
#         with open('Matriz_A.txt', 'r') as f:
#             A = np.loadtxt((linha.replace(',', '.') for linha in f))

#         with open('Vetor_b.txt', 'r') as f:
#             b = np.loadtxt((linha.replace(',', '.') for linha in f))

#     except FileNotFoundError:
#         print("Erro: Certifica-te de que 'Matriz_A.txt' e 'Vetor_b.txt' estão na mesma pasta do script.")
#         return

#     # Verifica se a matriz é simétrica
#     is_symmetric = np.allclose(A, A.T, atol=1e-8)

#     inicio = time.time()

#     if is_symmetric:
#         nome_metodo = "Método do Gradiente Conjugado (CG)"
#         print(f"Matriz Simétrica detetada. A executar o {nome_metodo}...")
#         x_sol, iteracoes, convergiu = gradiente_conjugado(A, b)
#     else:
#         nome_metodo = "Método do Gradiente Conjugado Quadrado (CGS)"
#         print(f"Matriz Não Simétrica detetada. A executar o {nome_metodo}...")
#         x_sol, iteracoes, convergiu = gradiente_conjugado_quadrado(A, b)

#     fim = time.time()
#     tempo_execucao = fim - inicio

#     print("\n--- RESULTADOS ---")
#     print(f"Método utilizado: {nome_metodo}")
#     print(f"Iterações realizadas: {iteracoes}")
#     print(f"Status: {'Convergiu com sucesso' if convergiu else 'Não convergiu / Interrompido'}")
#     print(f"Tempo de execução: {tempo_execucao:.6f} segundos")

#     # Criar e gravar o ficheiro Resultado_Sistema_Linear.txt
#     nome_ficheiro_txt = "Resultado_Sistema_Linear.txt"

#     with open(nome_ficheiro_txt, 'w', encoding='utf-8') as f:
#         f.write("==================================================\n")
#         f.write("          RESUMO DA EXECUÇÃO DO SISTEMA          \n")
#         f.write("==================================================\n")
#         f.write(f"Método Utilizado:        {nome_metodo}\n")
#         f.write(f"Tipo de Matriz:          {'Simétrica' if is_symmetric else 'Não Simétrica'}\n")
#         f.write(f"Iterações Realizadas:    {iteracoes}\n")
#         f.write(f"Status de Convergência:  {'Convergiu com sucesso' if convergiu else 'Não convergiu / Interrompido'}\n")
#         f.write(f"Tempo de Execução (s):   {tempo_execucao:.6f}\n")
#         f.write(f"Critério de Paragem:     < 1e-4\n")
#         f.write("==================================================\n\n")
#         f.write("VETOR SOLUÇÃO X:\n")
#         f.write("--------------------------------------------------\n")
#         for i, val in enumerate(x_sol):
#             f.write(f"x[{i:2d}] = {val:12.6f}\n")
#         f.write("--------------------------------------------------\n")

#     print(f"\nFicheiro '{nome_ficheiro_txt}' gerado e guardado localmente com sucesso!")

# if __name__ == "__main__":
#     main()