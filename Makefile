PYTHON := python
VENV := .venv
VENV_PYTHON := .venv\Scripts\python.exe
VENV_PIP := $(VENV)/Scripts/pip.exe

MATRIX_FILE ?= data/Matriz_A.txt
VECTOR_FILE ?= data/Vetor_b.txt

TOLERANCE ?= 1e-4
MAX_ITERATIONS ?= 100000
INIT_GUESS_LIN ?= 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
INIT_GUESS_NLIN ?= 0.6 0.4 0.3
NUM_RUNS = 1000
.PHONY: setup install run test benchmark clean

install:
	$(VENV_PYTHON) -m pip install -r requirements.txt

run_linear:
	$(VENV_PYTHON) -m src.main \
		--matrix $(MATRIX_FILE) \
		--vector $(VECTOR_FILE) \
		--tolerance $(TOLERANCE) \
		--is_linear "1" \
		--max-iterations $(MAX_ITERATIONS) \
		--init_guess "$(INIT_GUESS_LIN)" \
		--num_runs "$(NUM_RUNS)"

run_nonlinear:
	$(VENV_PYTHON) -m src.main \
		--tolerance $(TOLERANCE) \
		--max-iterations $(MAX_ITERATIONS) \
		--is_linear "0" \
		--init_guess "$(INIT_GUESS_NLIN)" \
		--num_runs "$(NUM_RUNS)"
