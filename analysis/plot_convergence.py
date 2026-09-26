import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_error_history(path: str):
    iterations = []
    errors = []

    with open(path, newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            iterations.append(int(row["iteration"]))
            errors.append(float(row["error"]))

    return iterations, errors


def plot_convergence(path: str):
    iterations, errors = load_error_history(path)

    method = Path(path).stem.replace("_", " ").title()

    plt.figure()

    plt.plot(iterations, errors)

    plt.xlabel("Iteration")
    plt.ylabel("Residual error")
    plt.title(f"{method} convergence")

    plt.grid(True)
    plt.tight_layout()
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot iterative method convergence"
    )

    parser.add_argument(
        "--file",
        required=True,
        help="CSV file containing iteration,error"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    plot_convergence(args.file)


if __name__ == "__main__":
    main()