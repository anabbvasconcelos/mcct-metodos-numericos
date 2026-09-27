PYTHON := python
VENV := .venv
VENV_PYTHON := .venv\Scripts\python.exe
VENV_PIP := $(VENV)/Scripts/pip.exe

MATRIX_FILE ?= data/Matriz_A.txt
VECTOR_FILE ?= data/Vetor_b.txt

TOLERANCE ?= 1e-4
MAX_ITERATIONS ?= 100000
INIT_GUESS ?= 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
.PHONY: setup install run test benchmark clean

install:
	$(VENV_PYTHON) -m pip install -r requirements.txt

run:
	$(VENV_PYTHON) -m src.main \
		--matrix $(MATRIX_FILE) \
		--vector $(VECTOR_FILE) \
		--tolerance $(TOLERANCE) \
		--max-iterations $(MAX_ITERATIONS) \
		--init_guess "$(INIT_GUESS)"
