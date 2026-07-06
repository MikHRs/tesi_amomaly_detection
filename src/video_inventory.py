"""Crea un inventario tecnico dei video usando OpenCV."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import cv2


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
FIELDNAMES = [
    "file_name",
    "path",
    "ok",
    "duration_seconds",
    "fps",
    "frame_count",
    "width",
    "height",
    "codec",
    "notes",
    "error",
]


def decode_fourcc(value: float) -> str:
    """Converte il codice FourCC numerico restituito da OpenCV in testo."""
    code = int(value)
    chars = [chr((code >> (8 * index)) & 0xFF) for index in range(4)]
    return "".join(char for char in chars if char.isprintable()).strip()


def inspect_video(path: Path) -> dict[str, Any]:
    """Legge i metadati di un video senza modificarlo."""
    row: dict[str, Any] = {
        "file_name": path.name,
        "path": str(path.resolve()),
        "ok": False,
        "duration_seconds": "",
        "fps": "",
        "frame_count": "",
        "width": "",
        "height": "",
        "codec": "",
        "notes": "",
        "error": "",
    }

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        row["error"] = "OpenCV non riesce ad aprire il video"
        capture.release()
        return row

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        codec = decode_fourcc(capture.get(cv2.CAP_PROP_FOURCC))

        valid_fps = fps if math.isfinite(fps) and fps > 0 else 0.0
        duration = frame_count / valid_fps if valid_fps and frame_count >= 0 else ""

        row.update(
            {
                "ok": True,
                "duration_seconds": round(duration, 3) if duration != "" else "",
                "fps": round(valid_fps, 6) if valid_fps else "",
                "frame_count": frame_count,
                "width": width,
                "height": height,
                "codec": codec,
            }
        )
    except Exception as exc:  # pragma: no cover - difesa da backend OpenCV
        row["error"] = f"Errore durante la lettura: {exc}"
    finally:
        capture.release()

    return row


def find_videos(input_dir: Path) -> list[Path]:
    """Trova ricorsivamente i formati video supportati."""
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def build_inventory(input_dir: Path, output_csv: Path) -> list[dict[str, Any]]:
    """Ispeziona i video e salva l'inventario CSV."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Cartella di input non trovata: {input_dir}")

    rows = [inspect_video(path) for path in find_videos(input_dir)]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crea un inventario CSV dei video presenti in una cartella."
    )
    parser.add_argument("--input", type=Path, required=True, help="Cartella dei video")
    parser.add_argument("--output", type=Path, required=True, help="CSV da creare")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        rows = build_inventory(args.input, args.output)
    except FileNotFoundError as exc:
        print(f"ERRORE: {exc}")
        return 2

    valid = sum(bool(row["ok"]) for row in rows)
    print(f"Inventario creato: {args.output}")
    print(f"Video trovati: {len(rows)}; leggibili: {valid}; errori: {len(rows) - valid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
