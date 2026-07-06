"""Estrae frame test esplicitamente selezionati da un video anomalo."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Any, Callable, TypeVar


CSV_FIELDS = [
    "filename",
    "frame_index",
    "timestamp",
    "source_video",
    "split",
    "notes",
]
T = TypeVar("T")


def parse_comma_separated(value: str, cast: Callable[[str], T]) -> list[T]:
    """Converte una lista separata da virgole, rifiutando elementi vuoti."""
    parts = [part.strip() for part in value.split(",")]
    if not parts or any(not part for part in parts):
        raise ValueError("La lista contiene un elemento vuoto")
    try:
        return [cast(part) for part in parts]
    except ValueError as exc:
        raise ValueError(f"Lista non valida: {value}") from exc


def selected_frame_indices(
    *,
    frame_count: int,
    source_fps: float,
    times: list[float] | None = None,
    frames: list[int] | None = None,
) -> list[int]:
    """Valida tempi o indici e restituisce frame unici in ordine cronologico."""
    if frame_count <= 0:
        raise ValueError("Il video non contiene frame")
    if source_fps <= 0 or not math.isfinite(source_fps):
        raise ValueError("FPS del video non valido")
    if (times is None) == (frames is None):
        raise ValueError("Specificare esattamente uno tra --times e --frames")

    if times is not None:
        if any(time < 0 or not math.isfinite(time) for time in times):
            raise ValueError("I timestamp devono essere finiti e non negativi")
        indices = [int(round(time * source_fps)) for time in times]
    else:
        assert frames is not None
        indices = frames

    if any(index < 0 or index >= frame_count for index in indices):
        raise ValueError(
            f"Un frame richiesto è fuori intervallo [0, {frame_count - 1}]"
        )
    return sorted(set(indices))


def load_cv2() -> Any:
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "OpenCV non è installato. Attivare l'ambiente e installare "
            "requirements.txt."
        ) from exc
    return cv2


def safe_stem(path: Path) -> str:
    """Crea un frammento di nome file portabile dal nome del video."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", path.stem).strip("_")
    return cleaned or "video"


def extract_test_anomaly_frames(
    video_path: Path,
    output_dir: Path,
    *,
    times: list[float] | None = None,
    frames: list[int] | None = None,
    jpeg_quality: int = 95,
) -> tuple[list[dict[str, object]], Path]:
    """Salva i frame richiesti e un manifest specifico per il video."""
    if not video_path.is_file():
        raise FileNotFoundError(f"Video non trovato: {video_path}")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("--jpeg-quality deve essere compreso tra 1 e 100")

    cv2 = load_cv2()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"OpenCV non riesce ad aprire: {video_path}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    try:
        indices = selected_frame_indices(
            frame_count=frame_count,
            source_fps=source_fps,
            times=times,
            frames=frames,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = safe_stem(video_path)
        manifest_path = output_dir / f"frames_{stem}.csv"
        filenames = [
            f"test_anomaly_{stem}_{sequence:04d}.jpg"
            for sequence in range(1, len(indices) + 1)
        ]
        conflicts = [
            path
            for path in [manifest_path, *(output_dir / name for name in filenames)]
            if path.exists()
        ]
        if conflicts:
            raise FileExistsError(
                f"Output già presente: {conflicts[0]}. "
                "Usare una cartella nuova o rimuovere esplicitamente il vecchio output."
            )

        rows: list[dict[str, object]] = []
        for filename, frame_index in zip(filenames, indices):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Lettura fallita al frame {frame_index}")
            image_path = output_dir / filename
            written = cv2.imwrite(
                str(image_path),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
            )
            if not written:
                raise RuntimeError(f"Impossibile scrivere: {image_path}")
            rows.append(
                {
                    "filename": filename,
                    "frame_index": frame_index,
                    "timestamp": round(frame_index / source_fps, 6),
                    "source_video": video_path.name,
                    "split": "test_anomaly",
                    "notes": "",
                }
            )
    finally:
        capture.release()

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows, manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estrae frame test selezionati da un video con anomalie."
    )
    parser.add_argument("--video", type=Path, required=True, help="Video sorgente")
    parser.add_argument("--out", type=Path, required=True, help="Cartella di output")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--times",
        help="Timestamp in secondi separati da virgole, es. 75,85,95",
    )
    selection.add_argument(
        "--frames",
        help="Indici frame separati da virgole, es. 1875,2125,2375",
    )
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        times = (
            parse_comma_separated(args.times, float)
            if args.times is not None
            else None
        )
        frames = (
            parse_comma_separated(args.frames, int)
            if args.frames is not None
            else None
        )
        rows, manifest_path = extract_test_anomaly_frames(
            args.video,
            args.out,
            times=times,
            frames=frames,
            jpeg_quality=args.jpeg_quality,
        )
    except (FileNotFoundError, FileExistsError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERRORE: {exc}")
        return 2

    print(f"Frame test estratti: {len(rows)}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
