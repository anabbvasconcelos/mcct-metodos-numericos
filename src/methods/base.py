from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np

from src.reader.txt_reader import Inputs

@dataclass(frozen=True)
class SolverConfig:
    tolerance: float = 1e-10
    max_iterations: int = 100_000
    initial_guess: np.ndarray | None = None

@dataclass
class IterativeResult:
    method: str
    solution: np.ndarray
    iterations: int
    error: float
    converged: bool
    execution_time: float
    error_history: list[float]
    

class IterativeSolver(ABC):

    def __init__(self, config):
        self.config = config

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def solve(self, system: Inputs, x0: np.ndarray | None = None):
        pass

    def residual(self, x_antigo: np.ndarray, x: np.ndarray) -> float:
        # difference between successive iterations:
        return float(np.max(np.abs(x - x_antigo)))

    def initial_guess(self, system: Inputs, x0: np.ndarray | None) -> np.ndarray:

        if x0 is None:
            return np.zeros(system.size)

        x0 = np.asarray(x0, dtype=float)

        if x0.shape != (system.size,):
            raise ValueError("x0 possui dimensão incompatível.")

        return x0.copy()