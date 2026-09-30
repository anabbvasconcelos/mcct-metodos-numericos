import numpy as np


class SystemNonLinInputs:
    def __init__(self):
        self.size = 3

    def evaluate_f(self, v: np.ndarray) -> np.ndarray:
        x, y, z = v
        f1 = (x - 1.0)**2 + (y - 1.0)**2 + (z - 1.0)**2 - 1.0
        f2 = 2.0 * x**2 + (y - 1.0)**2 - 4.0 * z
        f3 = 3.0 * x**2 + 2.0 * z**2 - 4.0 * y
        return np.array([f1, f2, f3])
    
    def evaluate_jacobian(self, v: np.ndarray) -> np.ndarray:
        x, y, z = v
        return np.array([
            [2.0 * (x - 1.0), 2.0 * (y - 1.0), 2.0 * (z - 1.0)],
            [4.0 * x,          2.0 * (y - 1.0), -4.0             ],
            [6.0 * x,         -4.0,             4.0 * z          ]
        ])