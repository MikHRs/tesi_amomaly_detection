"""Converte una anomaly mask in bounding box candidate.

Le coordinate ``xbr`` e ``ybr`` sono estremi esclusivi, coerenti con il
calcolo dell'area ``(xbr - xtl) * (ybr - ytl)``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


CSV_FIELDS = ["image", "xtl", "ytl", "xbr", "ybr", "area", "score", "method"]


def validate_parameters(threshold: float, min_area: int) -> None:
    if not 0 <= threshold <= 1:
        raise ValueError("--threshold deve essere compreso tra 0 e 1")
    if min_area <= 0:
        raise ValueError("--min-area deve essere maggiore di zero")


def load_image_dependencies() -> tuple[Any, Any]:
    try:
        import cv2
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "OpenCV e NumPy sono necessari. Attivare l'ambiente e installare "
            "requirements.txt."
        ) from exc
    return cv2, np


def normalized_grayscale(mask_path: Path) -> tuple[Any, Any, Any]:
    """Carica la maschera e restituisce valori float nell'intervallo [0, 1]."""
    if not mask_path.is_file():
        raise FileNotFoundError(f"Maschera non trovata: {mask_path}")
    cv2, np = load_image_dependencies()
    image = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Formato immagine non leggibile: {mask_path}")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    values = image.astype(np.float32)
    maximum = float(values.max()) if values.size else 0.0
    if maximum > 1.0:
        if np.issubdtype(image.dtype, np.integer):
            values /= float(np.iinfo(image.dtype).max)
        else:
            values /= maximum
    return np.clip(values, 0.0, 1.0), cv2, np


def mask_to_boxes(
    mask_path: Path,
    threshold: float = 0.5,
    min_area: int = 50,
    method: str = "anomaly_mask",
) -> list[dict[str, object]]:
    """Estrae componenti connesse e calcola box, area e score medio."""
    validate_parameters(threshold, min_area)
    normalized, cv2, np = normalized_grayscale(mask_path)
    binary = (normalized >= threshold).astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    boxes: list[dict[str, object]] = []
    for component in range(1, component_count):
        area = int(stats[component, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[component, cv2.CC_STAT_LEFT])
        y = int(stats[component, cv2.CC_STAT_TOP])
        width = int(stats[component, cv2.CC_STAT_WIDTH])
        height = int(stats[component, cv2.CC_STAT_HEIGHT])
        score = float(normalized[labels == component].mean())
        boxes.append(
            {
                "image": mask_path.name,
                "xtl": x,
                "ytl": y,
                "xbr": x + width,
                "ybr": y + height,
                "area": area,
                "score": round(score, 6),
                "method": method,
            }
        )
    return sorted(boxes, key=lambda row: (-int(row["area"]), int(row["ytl"])))


def convert_mask_to_csv(
    mask_path: Path,
    output_path: Path,
    threshold: float = 0.5,
    min_area: int = 50,
    method: str = "anomaly_mask",
) -> list[dict[str, object]]:
    boxes = mask_to_boxes(mask_path, threshold, min_area, method)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(boxes)
    return boxes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Converte una anomaly mask in bounding box candidate."
    )
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-area", type=int, default=50)
    parser.add_argument("--method", default="anomaly_mask")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        boxes = convert_mask_to_csv(
            args.mask,
            args.out,
            args.threshold,
            args.min_area,
            args.method,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERRORE: {exc}")
        return 2

    print(f"Bounding box candidate: {len(boxes)}")
    print(f"CSV: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
