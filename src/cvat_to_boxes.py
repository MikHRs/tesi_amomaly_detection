"""Converte box da XML CVAT classico a un CSV semplice."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from xml.etree import ElementTree as ET


FIELDNAMES = [
    "video_or_image",
    "frame",
    "track_id",
    "label",
    "xtl",
    "ytl",
    "xbr",
    "ybr",
    "outside",
    "occluded",
    "tipo",
    "visibilita",
    "posizione",
]
ATTRIBUTE_NAMES = ("tipo", "visibilita", "posizione")


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def first_text(root: ET.Element, path_names: tuple[str, ...]) -> str | None:
    """Trova testo seguendo una sequenza di tag, ignorando namespace."""
    candidates = [root]
    for name in path_names:
        candidates = [
            child
            for candidate in candidates
            for child in candidate
            if local_name(child.tag) == name
        ]
        if not candidates:
            return None
    text = candidates[0].text
    return text.strip() if text and text.strip() else None


def box_row(
    box: ET.Element,
    *,
    source: str,
    frame: str,
    track_id: str,
    default_label: str,
) -> dict[str, object]:
    """Converte un elemento <box> in una riga CSV."""
    try:
        coords = {
            name: float(box.attrib[name]) for name in ("xtl", "ytl", "xbr", "ybr")
        }
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Box con coordinate non valide al frame {frame}") from exc

    attributes = {
        child.get("name", ""): (child.text or "").strip()
        for child in box
        if local_name(child.tag) == "attribute"
    }
    return {
        "video_or_image": source,
        "frame": frame,
        "track_id": track_id,
        "label": box.get("label", default_label),
        **coords,
        "outside": int(box.get("outside", "0")),
        "occluded": int(box.get("occluded", "0")),
        **{name: attributes.get(name, "") for name in ATTRIBUTE_NAMES},
    }


def convert_cvat_xml(xml_path: Path) -> list[dict[str, object]]:
    """Legge annotazioni video (<track>) e immagini (<image>)."""
    if not xml_path.is_file():
        raise FileNotFoundError(f"XML non trovato: {xml_path}")

    root = ET.parse(xml_path).getroot()
    source = (
        first_text(root, ("meta", "task", "source"))
        or first_text(root, ("meta", "task", "name"))
        or xml_path.stem
    )
    rows: list[dict[str, object]] = []

    for track in (item for item in root.iter() if local_name(item.tag) == "track"):
        track_id = track.get("id", "")
        label = track.get("label", "")
        for child in track:
            if local_name(child.tag) != "box":
                continue
            frame = child.get("frame")
            if frame is None:
                raise ValueError(f"Box del track {track_id} senza attributo frame")
            rows.append(
                box_row(
                    child,
                    source=source,
                    frame=frame,
                    track_id=track_id,
                    default_label=label,
                )
            )

    for image in (item for item in root.iter() if local_name(item.tag) == "image"):
        image_name = image.get("name", source)
        frame = image.get("id", "0")
        for child in image:
            if local_name(child.tag) != "box":
                continue
            rows.append(
                box_row(
                    child,
                    source=image_name,
                    frame=frame,
                    track_id="",
                    default_label=child.get("label", ""),
                )
            )

    def sort_key(row: dict[str, object]) -> tuple[str, int, str]:
        try:
            frame_number = int(str(row["frame"]))
        except ValueError:
            frame_number = 0
        return str(row["video_or_image"]), frame_number, str(row["track_id"])

    return sorted(rows, key=sort_key)


def save_boxes(rows: list[dict[str, object]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Converte XML CVAT in box CSV.")
    parser.add_argument("--xml", type=Path, required=True, help="File annotations.xml")
    parser.add_argument("--out", type=Path, required=True, help="CSV di output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        rows = convert_cvat_xml(args.xml)
        save_boxes(rows, args.out)
    except (FileNotFoundError, ET.ParseError, OSError, ValueError) as exc:
        print(f"ERRORE: {exc}")
        return 2

    print(f"Box convertite: {len(rows)}")
    print(f"CSV creato: {args.out}")
    if not rows:
        print("ATTENZIONE: nell'XML non sono state trovate annotazioni <box>.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
