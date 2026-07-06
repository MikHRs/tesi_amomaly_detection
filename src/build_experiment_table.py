"""Crea un template CSV per la revisione manuale degli esperimenti."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


COLUMNS = [
    "frame",
    "source_video",
    "method",
    "anomaly_present",
    "detected",
    "false_positive",
    "false_negative",
    "localization_quality",
    "notes",
]
LOCALIZATION_QUALITY_VALUES = {
    "good",
    "partial",
    "bad",
    "not_detected",
    "uncertain",
}


def build_experiment_table(output_path: Path, overwrite: bool = False) -> None:
    """Scrive soltanto l'intestazione: non inventa righe sperimentali."""
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Il file esiste già: {output_path}. Usare --overwrite se intenzionale."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crea una tabella vuota per valutare gli esperimenti."
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Sovrascrive intenzionalmente un template esistente",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        build_experiment_table(args.out, args.overwrite)
    except (FileExistsError, OSError) as exc:
        print(f"ERRORE: {exc}")
        return 2

    print(f"Template creato: {args.out}")
    print(
        "localization_quality: "
        + ", ".join(sorted(LOCALIZATION_QUALITY_VALUES))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
