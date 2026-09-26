import csv
from pathlib import Path


def save_error_history(
    error_history: list[float],
    path: str = "results/jacobi_error.csv"
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow(["iteration", "error"])

        for iteration, error in enumerate(error_history, start=1):
            writer.writerow([iteration, error])