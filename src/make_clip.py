"""Crea una clip video senza richiedere ffmpeg."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2


def codec_for_output(path: Path) -> str:
    """Sceglie un codec OpenCV semplice in base all'estensione."""
    return "XVID" if path.suffix.lower() == ".avi" else "mp4v"


def make_clip(video_path: Path, start: float, end: float, output_path: Path) -> int:
    """Copia i frame nell'intervallo [start, end) in un nuovo video."""
    if start < 0:
        raise ValueError("--start non può essere negativo")
    if end <= start:
        raise ValueError("--end deve essere maggiore di --start")
    if not video_path.is_file():
        raise FileNotFoundError(f"Video non trovato: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"OpenCV non riesce ad aprire: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("Metadati video non validi (fps o risoluzione)")

    start_frame = int(round(start * fps))
    end_frame = int(round(end * fps))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec_for_output(output_path))
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        capture.release()
        writer.release()
        raise RuntimeError(
            f"Impossibile creare {output_path}; provare un output .mp4 o .avi"
        )

    written = 0
    try:
        for _frame_index in range(start_frame, end_frame):
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(frame)
            written += 1
    finally:
        capture.release()
        writer.release()

    if written == 0:
        output_path.unlink(missing_ok=True)
        raise RuntimeError("Nessun frame scritto: controllare l'intervallo richiesto")
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Taglia una clip con OpenCV.")
    parser.add_argument("--video", type=Path, required=True, help="Video sorgente")
    parser.add_argument("--start", type=float, required=True, help="Secondo iniziale")
    parser.add_argument("--end", type=float, required=True, help="Secondo finale escluso")
    parser.add_argument("--out", type=Path, required=True, help="Video di output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        written = make_clip(args.video, args.start, args.end, args.out)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERRORE: {exc}")
        return 2

    print(f"Clip creata: {args.out}")
    print(f"Frame scritti: {written}. Nota: OpenCV non conserva la traccia audio.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
