"""
Builds a first‑order Markov transition matrix of characters for any plain‑text
input and visualises it as a heat‑map.  Designed as a minimal, hackable core
for craftpurist experiments.

Usage (CLI):
    python main.py path/to/text.txt

Or import the functions in a notebook and iterate:
    from transition_matrix_heatmap import build_matrix, plot_heatmap
"""

import argparse
import pathlib
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Extended German alphabet (a‑z + äöüß).  Adapt as needed.
ALPHABET: Sequence[str] = list("abcdefghijklmnopqrstuvwxyzäöüß")
N = len(ALPHABET)
INDEX = {ch: i for i, ch in enumerate(ALPHABET)}

# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def normalise(text: str, *, alphabet: Sequence[str] = ALPHABET) -> str:
    """Lower‑case and drop all characters not in *alphabet*."""
    text = text.lower()
    return "".join(ch for ch in text if ch in alphabet)


def build_matrix(text: str, *, alphabet: Sequence[str] = ALPHABET) -> pd.DataFrame:
    """Return a Laplace‑smoothed transition matrix as a DataFrame."""
    n = len(alphabet)
    counts = np.ones((n, n), dtype=int)  # +1 Laplace smoothing

    for a, b in zip(text, text[1:]):
        counts[INDEX[a], INDEX[b]] += 1

    probs = counts / counts.sum(axis=1, keepdims=True)
    return pd.DataFrame(probs, index=alphabet, columns=alphabet)


def plot_heatmap(matrix: pd.DataFrame, *, title: str | None = None) -> None:
    """Display a heat‑map of the transition probabilities."""
    plt.figure(figsize=(9, 8))
    ax = sns.heatmap(matrix, cmap="viridis", cbar_kws={"label": "P(next | current)"})
    ax.set_xlabel("Next character")
    ax.set_ylabel("Current character")
    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_letter_frequencies(text: str, alphabet=ALPHABET, cmap="viridis"):
    counts = np.array([text.count(ch) for ch in alphabet])
    freqs = counts / counts.sum()

    # Scale color values
    colors = plt.get_cmap(cmap)(freqs / freqs.max())

    plt.figure(figsize=(10, 4))
    plt.bar(alphabet, freqs, color=colors, edgecolor="black")
    plt.ylabel("Relative frequency")
    plt.title("Letter frequency")
    plt.tight_layout()
    plt.show()


def process_directory(input_dir: str, output_dir: str, alphabet=ALPHABET):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for txt_file in input_path.glob("*.txt"):
        print(f"Processing {txt_file.name} ...")

        # Read and normalize
        raw = txt_file.read_text(encoding="utf-8", errors="ignore")
        clean = normalise(raw, alphabet=alphabet)

        freqs = np.array([clean.count(ch) for ch in alphabet]) / len(clean)

        plt.figure(figsize=(10, 4))
        colors = plt.cm.viridis(freqs / freqs.max())
        plt.bar(alphabet, freqs, color=colors, edgecolor="black")
        plt.title(f"Letter frequencies – {txt_file.name}")
        plt.ylabel("Relative frequency")
        plt.tight_layout()
        plt.savefig(output_path / f"{txt_file.stem}_freq.png")
        plt.close()

        matrix = build_matrix(clean, alphabet=alphabet)
        plt.figure(figsize=(9, 8))
        ax = sns.heatmap(
            matrix, cmap="viridis", cbar_kws={"label": "P(next | current)"}
        )
        ax.set_xlabel("Next character")
        ax.set_ylabel("Current character")
        ax.set_title(f"Transitions – {txt_file.name}")
        plt.tight_layout()
        plt.savefig(output_path / f"{txt_file.stem}_matrix.png")
        plt.close()

    print(f"Done. Plots saved to {output_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot character transition heatmap")
    parser.add_argument("textfile", type=pathlib.Path, help="Path to UTF‑8 text file")
    args = parser.parse_args()
    raw = args.textfile.read_text(encoding="utf-8", errors="ignore")

    clean = normalise(raw)
    matrix = build_matrix(clean)
    plot_heatmap(matrix, title=f"Transitions – {args.textfile.name}")
    plot_letter_frequencies(text=clean, alphabet=ALPHABET)


if __name__ == "__main__":
    process_directory("data", "output")
    # main()
