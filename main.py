"""Command-line launcher for the neural cellular automata experiment."""

from pathlib import Path
import runpy


def main() -> None:
    experiment = Path(__file__).parent / "title-cellular-automata" / "nca.py"
    runpy.run_path(str(experiment), run_name="__main__")


if __name__ == "__main__":
    main()
