"""Estrae frame normali, uniformemente distanziati, da un video.

Lo script prepara un insieme di riferimento per metodi reference-based come
AnomalyDINO. Non stabilisce automaticamente che il video sia normale: questa
assunzione deve essere verificata e documentata prima dell'estrazione.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


CSV_FIELDS = ["filename", "frame_index", "timestamp", "source_video", "split"]


def evenly_spaced_frame_indices(frame_count: int, num_frames: int) -> list[int]:
    """Seleziona indici distribuiti sull'intero video e non consecutivi."""
    if frame_count <= 0:
        raise ValueError("Il video non contiene frame")
    if num_frames <= 0:
        raise ValueError("--num-frames deve essere maggiore di zero")
    if num_frames == 1:
        return [frame_count // 2]
    if frame_count < 2 * num_frames - 1:
        raise ValueError(
            "Il video è troppo corto per estrarre il numero richiesto di frame "
            "senza selezionare frame consecutivi"
        )

    last_index = frame_count - 1
    indices = [
        int(round(position * last_index / (num_frames - 1)))
        for position in range(num_frames)
    ]
    if any(right - left <= 1 for left, right in zip(indices, indices[1:])):
        raise ValueError(
            "Impossibile ottenere frame non consecutivi: ridurre --num-frames"
        )
    return indices


def load_cv2() -> Any:
    """Importa OpenCV solo quando serve, con un errore comprensibile."""
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "OpenCV non è installato. Attivare l'ambiente e installare "
            "requirements.txt."
        ) from exc
    return cv2


def extract_normal_frames(
    video_path: Path,
    output_dir: Path,
    num_frames: int,
    jpeg_quality: int = 95,
) -> list[dict[str, object]]:
    """Estrae frame normali e salva ``frames.csv`` nella cartella di output."""
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
    if not math.isfinite(source_fps) or source_fps <= 0:
        capture.release()
        raise RuntimeError("FPS del video non valido")

    try:
        selected_indices = evenly_spaced_frame_indices(frame_count, num_frames)
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "frames.csv"
        planned_paths = [
            output_dir / f"normal_{sequence:04d}.jpg"
            for sequence in range(1, len(selected_indices) + 1)
        ]
        conflicts = [path for path in [manifest_path, *planned_paths] if path.exists()]
        if conflicts:
            raise FileExistsError(
                f"Output già presente: {conflicts[0]}. "
                "Usare una cartella nuova per non sovrascrivere il dataset."
            )

        rows: list[dict[str, object]] = []
        for sequence, frame_index in enumerate(selected_indices, start=1):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Lettura fallita al frame {frame_index}")

            filename = f"normal_{sequence:04d}.jpg"
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
                    "split": "normal",
                }
            )
    finally:
        capture.release()

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estrae frame normali uniformemente distanziati da un video."
    )
    parser.add_argument("--video", type=Path, required=True, help="Video normale")
    parser.add_argument("--out", type=Path, required=True, help="Cartella di output")
    parser.add_argument(
        "--num-frames",
        type=int,
        default=200,
        help="Numero di frame da estrarre (default: 200)",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="Qualità JPEG tra 1 e 100 (default: 95)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        rows = extract_normal_frames(
            args.video,
            args.out,
            args.num_frames,
            args.jpeg_quality,
        )
    except (FileNotFoundError, FileExistsError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERRORE: {exc}")
        return 2

    print(f"Frame normali estratti: {len(rows)}")
    print(f"Immagini e CSV: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
