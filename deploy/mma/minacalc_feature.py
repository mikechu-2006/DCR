"""Process-isolated MinaCalc 505 feature for the local CPU service."""
from pathlib import Path
import sys

from legacy.scripts.step0h_minacalc_msd import LibBackend, parse_osu_mania


def main():
    chart, library = sys.argv[1:]
    keycount, rows, taps, holds = parse_osu_mania(Path(chart).read_text(encoding="utf-8-sig"))
    if keycount != 4 or taps + holds < 2 or not rows:
        raise ValueError("Chart has too few playable notes")
    calc = LibBackend(library, 1, 0.93)
    if calc.version != 505:
        raise ValueError("MinaCalc 505 is required")
    print(calc.msd(rows)["overall"])


if __name__ == "__main__":
    main()
