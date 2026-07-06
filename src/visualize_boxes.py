"""Disegna box da CSV sui frame e salva immagini di controllo."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import cv2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
BOX_COLUMNS = {"frame", "label", "xtl", "ytl", "xbr", "ybr"}


def as_active(value: str | None) -> bool:
    """Interpreta il campo outside di CVAT."""
    return str(value or "0").strip().lower() not in {"1", "true", "yes"}


def canonical_frame(value: object) -> str:
    """Normalizza frame numerici letti come 1, 1.0 o stringa."""
    text = str(value).strip()
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def load_boxes(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV box non trovato: {csv_path}")
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = BOX_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Colonne mancanti nel CSV: {sorted(missing)}")
        return [row for row in reader if as_active(row.get("outside"))]


def load_frame_index(frames_dir: Path) -> dict[str, str]:
    """Legge l'indice prodotto da extract_frames.py, se disponibile."""
    index_paths = [frames_dir / "frames.csv"]
    index_paths.extend(sorted(frames_dir.glob("frames_*.csv")))
    index_paths = [path for path in index_paths if path.is_file()]
    if not index_paths:
        return {}

    mapping: dict[str, str] = {}
    for index_path in index_paths:
        with index_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                image_name = row.get("image_name") or row.get("filename")
                if image_name and row.get("frame_index") is not None:
                    mapping[image_name] = canonical_frame(row["frame_index"])
    return mapping


def fallback_frame_from_name(name: str) -> str | None:
    """Interpreta frame_000001 come frame CVAT 0 quando manca frames.csv."""
    match = re.match(r"frame_(\d+)", name)
    return str(int(match.group(1)) - 1) if match else None


def rows_for_image(
    image_path: Path,
    rows_by_frame: dict[str, list[dict[str, str]]],
    rows_by_image: dict[str, list[dict[str, str]]],
    frame_index: dict[str, str],
) -> list[dict[str, str]]:
    """Associa box per nome immagine oppure per indice del frame sorgente."""
    exact = rows_by_image.get(image_path.name, [])
    if exact:
        return exact
    frame = frame_index.get(image_path.name) or fallback_frame_from_name(image_path.name)
    return rows_by_frame.get(frame, []) if frame is not None else []


def visualize(
    frames_dir: Path,
    boxes_csv: Path,
    output_dir: Path,
    color_name: str = "green",
) -> tuple[int, int]:
    """Salva tutti i frame, con box dove disponibili."""
    if not frames_dir.is_dir():
        raise FileNotFoundError(f"Cartella frame non trovata: {frames_dir}")
    rows = load_boxes(boxes_csv)
    images = sorted(
        path
        for path in frames_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise ValueError(f"Nessuna immagine trovata in {frames_dir}")

    rows_by_frame: dict[str, list[dict[str, str]]] = defaultdict(list)
    rows_by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_frame[canonical_frame(row["frame"])].append(row)
        source_name = Path(row.get("video_or_image", "")).name
        if source_name:
            rows_by_image[source_name].append(row)

    frame_index = load_frame_index(frames_dir)
    color = (0, 255, 0) if color_name == "green" else (0, 0, 255)
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated = 0

    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"ATTENZIONE: immagine non leggibile, saltata: {image_path}")
            continue

        image_rows = rows_for_image(
            image_path, rows_by_frame, rows_by_image, frame_index
        )
        if image_rows:
            annotated += 1
        height, width = image.shape[:2]
        for row in image_rows:
            x1 = max(0, min(width - 1, int(round(float(row["xtl"])))))
            y1 = max(0, min(height - 1, int(round(float(row["ytl"])))))
            x2 = max(0, min(width - 1, int(round(float(row["xbr"])))))
            y2 = max(0, min(height - 1, int(round(float(row["ybr"])))))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = row.get("label", "anomalia")
            track_id = row.get("track_id", "").strip()
            if track_id:
                label = f"{label} #{track_id}"
            cv2.putText(
                image,
                label,
                (x1, max(15, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        relative = image_path.relative_to(frames_dir)
        destination = output_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination), image):
            raise RuntimeError(f"Impossibile scrivere: {destination}")

    return len(images), annotated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Disegna box CSV sui frame.")
    parser.add_argument("--frames", type=Path, required=True, help="Cartella frame")
    parser.add_argument("--boxes", type=Path, required=True, help="CSV box")
    parser.add_argument("--out", type=Path, required=True, help="Cartella output")
    parser.add_argument(
        "--color",
        choices=("green", "red"),
        default="green",
        help="Colore delle box",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        total, annotated = visualize(args.frames, args.boxes, args.out, args.color)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERRORE: {exc}")
        return 2

    print(f"Immagini salvate: {total}; immagini con box: {annotated}")
    print(f"Output: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
