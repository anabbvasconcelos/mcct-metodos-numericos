# Métodos Numéricos

Implementação em Python de métodos numéricos para a resolução de **sistemas lineares e sistemas não lineares**

O projeto tem como objetivo implementar, executar e comparar diferentes métodos quanto à **convergência, número de iterações e custo computacional**.

## Objetivos

O projeto contempla dois grupos principais de problemas:

* **Sistemas lineares**

  * Método de Jacobi;
  * Método de Gauss-Seidel;
  * Métodos de Gauss-Seidel de ordem superior;
  * Gradiente Conjugado;
  * Gradiente Conjugado Quadrado (CGS);
  * outros métodos iterativos implementados no projeto.

* **Sistemas não lineares**

  * Método de Newton;
  * Método de Newton Modificado;
  * Método do Gradiente.

Além da implementação dos métodos, o projeto permite realizar benchmarks para comparar o desempenho computacional das diferentes abordagens.

---

## Estrutura do projeto

```text
.
├── data/
│   ├── Matriz_A.txt
│   └── Vetor_b.txt
│
├── src/
│   ├── main.py
│   └── ...
│
├── results/
│   └── ...
│
├── requirements.txt
├── Makefile
└── README.md
```

# Requisitos

* Python 3.11 ou versão compatível;
* `pip`;
* `make`;
* ambiente virtual Python.

As dependências do projeto estão especificadas em:

```text
requirements.txt
```

---

# Configuração do ambiente

O projeto utiliza um ambiente virtual localizado em:

```text
.venv/
```

O interpretador Python utilizado pelo `Makefile` é:

```text
.venv\Scripts\python.exe
```

Para instalar as dependências:

```bash
make install
```

Esse comando executa:

```bash
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

# Execução

O `Makefile` possui comandos separados para os sistemas lineares e não lineares.

## Sistema linear

Para executar os métodos destinados ao sistema linear:

```bash
make run_linear
```

Por padrão, são utilizados:

```text
Matriz:       data/Matriz_A.txt
Vetor:        data/Vetor_b.txt
Tolerância:   1e-4
Máximo:       100000 iterações
Chute inicial: vetor de 36 zeros
Execuções:    1000
```

O sistema pode ser executado diretamente com:

```bash
.venv\Scripts\python.exe -m src.main \
    --matrix data/Matriz_A.txt \
    --vector data/Vetor_b.txt \
    --tolerance 1e-4 \
    --is_linear 1 \
    --max-iterations 100000 \
    --init_guess "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0" \
    --num_runs 1000
```

---

## Sistema não linear

Para executar os métodos destinados ao sistema não linear:

```bash
make run_nonlinear
```

O chute inicial padrão é:

```text
(0.6, 0.4, 0.3)
```

A execução utiliza:

```text
Tolerância:    1e-4
Máximo:        100000 iterações
Chute inicial: (0.6, 0.4, 0.3)
Execuções:     1000
```

O sistema pode ser executado diretamente com:

```bash
.venv\Scripts\python.exe -m src.main \
    --tolerance 1e-4 \
    --max-iterations 100000 \
    --is_linear 0 \
    --init_guess "0.6 0.4 0.3" \
    --num_runs 1000
```

---

# Configuração pelo Makefile

Os parâmetros podem ser sobrescritos diretamente na linha de comando.

## Alterando a tolerância

```bash
make run_linear TOLERANCE=1e-10
```

ou:

```bash
make run_nonlinear TOLERANCE=1e-10
```

## Alterando o número máximo de iterações

```bash
make run_linear MAX_ITERATIONS=50000
```

## Alterando o chute inicial do sistema linear

```bash
make run_linear INIT_GUESS_LIN="1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1"
```

## Alterando o chute inicial do sistema não linear

```bash
make run_nonlinear INIT_GUESS_NLIN="1 1 1"
```

## Alterando o número de execuções do benchmark

```bash
make run_nonlinear NUM_RUNS=10000
```

O mesmo parâmetro pode ser utilizado no sistema linear:

```bash
make run_linear NUM_RUNS=10000
```

---
# Reprodutibilidade

```bash
make run_nonlinear \
    TOLERANCE=1e-4 \
    MAX_ITERATIONS=100000 \
    INIT_GUESS_NLIN="0.6 0.4 0.3" \
    NUM_RUNS=1000
```

Para os experimentos de sensibilidade, basta alterar `INIT_GUESS_NLIN`:

```bash
make run_nonlinear INIT_GUESS_NLIN="0 0 0"
```

```bash
make run_nonlinear INIT_GUESS_NLIN="1 1 1"
```

```bash
make run_nonlinear INIT_GUESS_NLIN="0.5 0.5 0.5"
```

Os resultados podem então ser comparados quanto ao número de iterações, erro e tempo de execução.

---

# Comandos disponíveis

| Comando              | Descrição                                    |
| -------------------- | -------------------------------------------- |
| `make install`       | Instala as dependências do projeto           |
| `make run_linear`    | Executa os métodos para o sistema linear     |
| `make run_nonlinear` | Executa os métodos para o sistema não linear |

---

# Parâmetros principais

| Parâmetro         | Valor padrão        | Descrição                          |
| ----------------- | ------------------- | ---------------------------------- |
| `MATRIX_FILE`     | `data/Matriz_A.txt` | Arquivo da matriz \(A\)            |
| `VECTOR_FILE`     | `data/Vetor_b.txt`  | Arquivo do vetor \(b\)             |
| `TOLERANCE`       | `1e-4`              | Critério de tolerância             |
| `MAX_ITERATIONS`  | `100000`            | Número máximo de iterações         |
| `INIT_GUESS_LIN`  | 36 zeros            | Chute inicial linear               |
| `INIT_GUESS_NLIN` | `0.6 0.4 0.3`       | Chute inicial não linear           |
| `NUM_RUNS`        | `1000`              | Número de execuções para benchmark |

---

# Tecnologias

* Python
* NumPy
* Make
* Ambiente virtual Python
