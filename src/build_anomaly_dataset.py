"""Prepara uno split one-class da video e annotazioni CVAT.

I frame senza box attive, lontani dai confini delle annotazioni, vengono
divisi in training e validation normali. I frame con box attive e lontani dai
confini formano il test anomalo. La fascia vicina alle transizioni viene
esclusa per ridurre l'effetto di annotazioni temporali imprecise.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

TRUE_VALUES = {"1", "true", "yes"}


@dataclass(frozen=True)
class VideoInfo:
    file_name: str
    path: Path
    fps: float
    frame_count: int


def sample_frame_indices(
    frame_count: int,
    source_fps: float,
    sample_fps: float,
) -> list[int]:
    """Restituisce indici temporali unici campionati a frequenza costante."""
    if frame_count <= 0:
        raise ValueError("frame_count deve essere positivo")
    if source_fps <= 0 or not math.isfinite(source_fps):
        raise ValueError("source_fps deve essere positivo e finito")
    if sample_fps <= 0 or sample_fps > source_fps:
        raise ValueError("sample_fps deve essere tra 0 e source_fps")

    duration = frame_count / source_fps
    count = int(math.ceil(duration * sample_fps - 1e-9))
    return sorted(
        {
            frame_index
            for index in range(count)
            if (frame_index := int(round(index * source_fps / sample_fps)))
            < frame_count
        }
    )


def classify_sampled_frames(
    sampled_frames: list[int],
    active_anomaly_frames: set[int],
    frame_count: int,
    margin_frames: int,
) -> tuple[list[int], list[int], list[int]]:
    """Classifica frame sicuri normali, sicuri anomali e ambigui."""
    if margin_frames < 0:
        raise ValueError("margin_frames non può essere negativo")

    normal: list[int] = []
    anomaly: list[int] = []
    ambiguous: list[int] = []

    for frame_index in sampled_frames:
        low = max(0, frame_index - margin_frames)
        high = min(frame_count - 1, frame_index + margin_frames)
        window = range(low, high + 1)

        if frame_index in active_anomaly_frames:
            target = anomaly if all(i in active_anomaly_frames for i in window) else ambiguous
        else:
            target = normal if all(i not in active_anomaly_frames for i in window) else ambiguous
        target.append(frame_index)

    return normal, anomaly, ambiguous


def split_normal_frames(
    normal_frames: list[int],
    train_fraction: float,
) -> tuple[list[int], list[int]]:
    """Divide cronologicamente i frame normali in training e validation."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction deve essere tra 0 e 1")
    if len(normal_frames) < 2:
        raise ValueError("Servono almeno due frame normali")

    split_index = int(round(len(normal_frames) * train_fraction))
    split_index = min(max(split_index, 1), len(normal_frames) - 1)
    return normal_frames[:split_index], normal_frames[split_index:]


def read_inventory(path: Path) -> dict[str, VideoInfo]:
    """Legge l'inventario dei video validi."""
    videos: dict[str, VideoInfo] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("ok", "").strip().lower() not in TRUE_VALUES:
                continue
            video = VideoInfo(
                file_name=row["file_name"],
                path=Path(row["path"]),
                fps=float(row["fps"]),
                frame_count=int(row["frame_count"]),
            )
            videos[video.file_name] = video
    return videos


def read_active_anomaly_frames(
    annotations_dir: Path,
) -> tuple[
    dict[str, set[int]],
    dict[str, list[str]],
    dict[str, dict[int, list[dict[str, str]]]],
]:
    """Unisce le box CVAT attive per video."""
    active_by_video: dict[str, set[int]] = defaultdict(set)
    sources_by_video: dict[str, list[str]] = defaultdict(list)
    boxes_by_video: dict[str, dict[int, list[dict[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for path in sorted(annotations_dir.glob("*_boxes.csv")):
        videos_in_file: set[str] = set()
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                video_name = row["video_or_image"]
                videos_in_file.add(video_name)
                if row.get("outside", "0").strip().lower() not in TRUE_VALUES:
                    frame_index = int(row["frame"])
                    active_by_video[video_name].add(frame_index)
                    boxes_by_video[video_name][frame_index].append(
                        {
                            "video": video_name,
                            "frame": str(frame_index),
                            "track_id": row["track_id"],
                            "label": row["label"],
                            "xtl": row["xtl"],
                            "ytl": row["ytl"],
                            "xbr": row["xbr"],
                            "ybr": row["ybr"],
                            "occluded": row.get("occluded", "0"),
                            "annotation_source": path.name,
                        }
                    )
        for video_name in videos_in_file:
            sources_by_video[video_name].append(path.name)

    return (
        dict(active_by_video),
        dict(sources_by_video),
        {video: dict(frames) for video, frames in boxes_by_video.items()},
    )


def save_selected_frames(
    video: VideoInfo,
    rows_by_frame: dict[int, dict[str, object]],
    output_dir: Path,
    jpeg_quality: int,
) -> None:
    """Decodifica il video una volta e salva soltanto i frame selezionati."""
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "OpenCV non è installato. Attivare l'ambiente e installare "
            "requirements.txt."
        ) from exc

    capture = cv2.VideoCapture(str(video.path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Impossibile aprire il video: {video.path}")

    pending = set(rows_by_frame)
    frame_index = 0
    try:
        while pending:
            ok, image = capture.read()
            if not ok:
                break
            if frame_index in pending:
                row = rows_by_frame[frame_index]
                image_path = output_dir / str(row["image_path"])
                image_path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(
                    str(image_path),
                    image,
                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
                ):
                    raise RuntimeError(f"Impossibile scrivere: {image_path}")
                pending.remove(frame_index)
            frame_index += 1
    finally:
        capture.release()

    if pending:
        missing = ", ".join(str(i) for i in sorted(pending)[:10])
        raise RuntimeError(f"Frame non decodificati in {video.file_name}: {missing}")


def build_dataset(
    inventory_path: Path,
    annotations_dir: Path,
    output_dir: Path,
    sample_fps: float,
    margin_seconds: float,
    train_fraction: float,
    jpeg_quality: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Crea immagini, manifest e riepilogo dello split."""
    if output_dir.exists():
        raise FileExistsError(
            f"La cartella di output esiste già: {output_dir}. "
            "Scegliere una nuova cartella per conservare la riproducibilità."
        )
    if margin_seconds < 0:
        raise ValueError("margin_seconds non può essere negativo")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality deve essere tra 1 e 100")

    inventory = read_inventory(inventory_path)
    active_by_video, sources_by_video, boxes_by_video = read_active_anomaly_frames(
        annotations_dir
    )
    if not active_by_video:
        raise ValueError("Nessuna annotazione CVAT attiva trovata")

    output_dir.mkdir(parents=True)
    manifest: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    ground_truth_boxes: list[dict[str, str]] = []

    for video_name in sorted(active_by_video):
        if video_name not in inventory:
            raise KeyError(f"Video annotato non presente nell'inventario: {video_name}")
        video = inventory[video_name]
        if not video.path.is_file():
            raise FileNotFoundError(f"Video non trovato: {video.path}")

        sampled = sample_frame_indices(video.frame_count, video.fps, sample_fps)
        margin_frames = int(round(margin_seconds * video.fps))
        normal, anomaly, ambiguous = classify_sampled_frames(
            sampled,
            active_by_video[video_name],
            video.frame_count,
            margin_frames,
        )
        train_normal, validation_normal = split_normal_frames(normal, train_fraction)

        groups = [
            ("train", "normal", train_normal),
            ("validation", "normal", validation_normal),
            ("test", "anomaly", anomaly),
        ]
        rows_by_frame: dict[int, dict[str, object]] = {}
        stem = Path(video_name).stem
        annotation_sources = ";".join(sources_by_video[video_name])

        for split, label, frame_indices in groups:
            for frame_index in frame_indices:
                image_name = f"{stem}_frame_{frame_index:06d}.jpg"
                relative_path = Path(split) / label / stem / image_name
                row: dict[str, object] = {
                    "image_path": relative_path.as_posix(),
                    "split": split,
                    "label": label,
                    "video": video_name,
                    "frame": frame_index,
                    "timestamp_seconds": round(frame_index / video.fps, 6),
                    "source_fps": round(video.fps, 6),
                    "annotation_sources": annotation_sources,
                }
                manifest.append(row)
                rows_by_frame[frame_index] = row
                if label == "anomaly":
                    ground_truth_boxes.extend(
                        boxes_by_video[video_name].get(frame_index, [])
                    )

            summary.append(
                {
                    "video": video_name,
                    "split": split,
                    "label": label,
                    "sampled_frames": len(frame_indices),
                }
            )

        summary.append(
            {
                "video": video_name,
                "split": "excluded",
                "label": "ambiguous_transition",
                "sampled_frames": len(ambiguous),
            }
        )
        save_selected_frames(video, rows_by_frame, output_dir, jpeg_quality)

    fieldnames = [
        "image_path",
        "split",
        "label",
        "video",
        "frame",
        "timestamp_seconds",
        "source_fps",
        "annotation_sources",
    ]
    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest)

    with (output_dir / "split_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["video", "split", "label", "sampled_frames"],
        )
        writer.writeheader()
        writer.writerows(summary)

    box_fieldnames = [
        "video",
        "frame",
        "track_id",
        "label",
        "xtl",
        "ytl",
        "xbr",
        "ybr",
        "occluded",
        "annotation_source",
    ]
    with (output_dir / "ground_truth_boxes.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=box_fieldnames)
        writer.writeheader()
        writer.writerows(ground_truth_boxes)

    metadata = {
        "task": "one-class anomaly detection/localization",
        "normal_definition": "sampled frame without active CVAT anomaly boxes",
        "anomaly_definition": "sampled frame with at least one active CVAT anomaly box",
        "boundary_policy": "exclude samples whose temporal margin crosses a label transition",
        "sample_fps": sample_fps,
        "margin_seconds": margin_seconds,
        "normal_train_fraction": train_fraction,
        "jpeg_quality": jpeg_quality,
        "inventory": str(inventory_path),
        "annotations_dir": str(annotations_dir),
        "image_count": len(manifest),
        "ground_truth_box_count": len(ground_truth_boxes),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepara frame per un pilot one-class di anomaly detection."
    )
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--annotations-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--margin-seconds", type=float, default=2.0)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest, _ = build_dataset(
            args.inventory,
            args.annotations_dir,
            args.out,
            args.sample_fps,
            args.margin_seconds,
            args.train_fraction,
            args.jpeg_quality,
        )
    except (FileExistsError, FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"ERRORE: {exc}")
        return 2

    normal_count = sum(row["label"] == "normal" for row in manifest)
    anomaly_count = sum(row["label"] == "anomaly" for row in manifest)
    print(f"Dataset creato: {args.out}")
    print(f"Frame normali: {normal_count}")
    print(f"Frame anomali: {anomaly_count}")
    print(f"Totale immagini: {len(manifest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
