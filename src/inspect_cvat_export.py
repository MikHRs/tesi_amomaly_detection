"""Ispeziona un export XML classico di CVAT e salva un riepilogo."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET


SHAPE_TAGS = {
    "box",
    "polygon",
    "mask",
    "polyline",
    "points",
    "ellipse",
    "cuboid",
    "skeleton",
    "tag",
}


def local_name(tag: str) -> str:
    """Rimuove un eventuale namespace XML."""
    return tag.rsplit("}", 1)[-1]


def inspect_xml(xml_path: Path) -> dict[str, object]:
    """Conta immagini, track, forme e label annotate/configurate."""
    if not xml_path.is_file():
        raise FileNotFoundError(f"XML non trovato: {xml_path}")

    root = ET.parse(xml_path).getroot()
    elements = list(root.iter())
    counts = Counter(local_name(element.tag) for element in elements)
    annotation_labels: Counter[str] = Counter()

    for track in (item for item in elements if local_name(item.tag) == "track"):
        track_label = track.get("label", "<senza_label>")
        for child in track:
            if local_name(child.tag) in SHAPE_TAGS:
                annotation_labels[child.get("label", track_label)] += 1

    for image in (item for item in elements if local_name(item.tag) == "image"):
        for child in image:
            if local_name(child.tag) in SHAPE_TAGS:
                annotation_labels[child.get("label", "<senza_label>")] += 1

    configured_labels = sorted(
        {
            (name.text or "").strip()
            for label in (item for item in elements if local_name(item.tag) == "label")
            for name in label
            if local_name(name.tag) == "name" and (name.text or "").strip()
        }
    )

    return {
        "xml": str(xml_path.resolve()),
        "images": counts["image"],
        "tracks": counts["track"],
        "boxes": counts["box"],
        "polygons": counts["polygon"],
        "masks": counts["mask"],
        "configured_labels": configured_labels,
        "annotation_label_counts": dict(sorted(annotation_labels.items())),
    }


def save_summary(summary: dict[str, object], output_dir: Path, stem: str) -> tuple[Path, Path]:
    """Salva lo stesso riepilogo in JSON e CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{stem}_summary.json"
    csv_path = output_dir / f"{stem}_summary.csv"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    rows: list[dict[str, object]] = []
    for key in ("images", "tracks", "boxes", "polygons", "masks"):
        rows.append({"metric": key, "name": "", "value": summary[key]})
    for label in summary["configured_labels"]:  # type: ignore[union-attr]
        rows.append({"metric": "configured_label", "name": label, "value": 1})
    for label, count in summary["annotation_label_counts"].items():  # type: ignore[union-attr]
        rows.append({"metric": "annotation_label", "name": label, "value": count})

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "name", "value"])
        writer.writeheader()
        writer.writerows(rows)

    return json_path, csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ispeziona un export XML di CVAT.")
    parser.add_argument("--xml", type=Path, required=True, help="File annotations.xml")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/cvat_inspection"),
        help="Cartella dei riepiloghi (default: results/cvat_inspection)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = inspect_xml(args.xml)
        json_path, csv_path = save_summary(summary, args.out_dir, args.xml.stem)
    except (FileNotFoundError, ET.ParseError, OSError) as exc:
        print(f"ERRORE: {exc}")
        return 2

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Riepiloghi salvati: {json_path}, {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
