from pathlib import Path
from collections import Counter
import re

DATA_DIR = "data"
TOP_N = 10


def clean_and_split(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-zäöüß\s]", " ", text)
    words = text.split()
    return words


def top_words_for_file(filepath: Path, n: int = TOP_N) -> list[tuple[str, int]]:
    text = filepath.read_text(encoding="utf-8", errors="ignore")
    words = clean_and_split(text)
    counter = Counter(words)
    return counter.most_common(n)


def process_directory(data_dir: str = DATA_DIR, n: int = TOP_N):
    path = Path(data_dir)
    for file in path.glob("*.txt"):
        print(f"\n--- {file.name} ---")
        top_words = top_words_for_file(file, n)
        for word, count in top_words:
            print(f"{word:<15} {count}")


if __name__ == "__main__":
    process_directory()
