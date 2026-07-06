"""Estrae frame campionati da un video e salva un indice CSV."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2


def extraction_times(start: float, end: float, target_fps: float) -> list[float]:
    """Genera timestamp nell'intervallo semiaperto [start, end)."""
    count = max(0, int(math.ceil((end - start) * target_fps - 1e-9)))
    return [start + index / target_fps for index in range(count)]


def extract_frames(
    video_path: Path,
    output_dir: Path,
    target_fps: float,
    start: float = 0.0,
    end: float | None = None,
) -> list[dict[str, object]]:
    """Estrae frame a frequenza costante usando indici del video originale."""
    if target_fps <= 0:
        raise ValueError("--fps deve essere maggiore di zero")
    if start < 0:
        raise ValueError("--start non può essere negativo")
    if end is not None and end <= start:
        raise ValueError("--end deve essere maggiore di --start")
    if not video_path.is_file():
        raise FileNotFoundError(f"Video non trovato: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"OpenCV non riesce ad aprire: {video_path}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not math.isfinite(source_fps) or source_fps <= 0:
        capture.release()
        raise RuntimeError("FPS del video non valido: impossibile campionare per tempo")
    if target_fps > source_fps + 1e-9:
        capture.release()
        raise ValueError(
            f"--fps ({target_fps}) non può superare gli FPS sorgente ({source_fps:.3f})"
        )

    duration = frame_count / source_fps if frame_count > 0 else None
    effective_end = end
    if effective_end is None:
        if duration is None:
            capture.release()
            raise RuntimeError("Durata del video non disponibile: specificare --end")
        effective_end = duration
    elif duration is not None:
        effective_end = min(effective_end, duration)

    if effective_end <= start:
        capture.release()
        raise ValueError("L'intervallo richiesto non contiene frame del video")

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    used_indices: set[int] = set()

    try:
        for requested_time in extraction_times(start, effective_end, target_fps):
            source_index = int(round(requested_time * source_fps))
            if frame_count > 0 and source_index >= frame_count:
                break
            if source_index in used_indices:
                continue

            capture.set(cv2.CAP_PROP_POS_FRAMES, source_index)
            ok, frame = capture.read()
            if not ok:
                print(
                    f"ATTENZIONE: lettura fallita al frame {source_index}; "
                    "estrazione interrotta."
                )
                break

            used_indices.add(source_index)
            actual_time = source_index / source_fps
            sequence = len(rows) + 1
            image_name = f"frame_{sequence:06d}_t{actual_time:06.2f}.jpg"
            image_path = output_dir / image_name
            if not cv2.imwrite(str(image_path), frame):
                raise RuntimeError(f"Impossibile scrivere l'immagine: {image_path}")

            rows.append(
                {
                    "image_name": image_name,
                    "frame_index": source_index,
                    "timestamp_seconds": round(actual_time, 6),
                }
            )
    finally:
        capture.release()

    index_path = output_dir / "frames.csv"
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_name", "frame_index", "timestamp_seconds"],
        )
        writer.writeheader()
        writer.writerows(rows)

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estrae frame campionati da un video.")
    parser.add_argument("--video", type=Path, required=True, help="Video sorgente")
    parser.add_argument("--out", type=Path, required=True, help="Cartella di output")
    parser.add_argument("--fps", type=float, required=True, help="Frame estratti al secondo")
    parser.add_argument("--start", type=float, default=0.0, help="Secondo iniziale")
    parser.add_argument("--end", type=float, default=None, help="Secondo finale escluso")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        rows = extract_frames(args.video, args.out, args.fps, args.start, args.end)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERRORE: {exc}")
        return 2

    print(f"Frame estratti: {len(rows)}")
    print(f"Immagini e indice CSV: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
