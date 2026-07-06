"""Create quantitative and qualitative artefacts for E09.

E09 compares the adaptive AnomalyDINO component containing the automatic
maximum point with the bounding box of the SAM2 mask prompted by that point.
Only box ground truth is available, so SAM2 masks are visualized but are not
assigned pixel-level segmentation metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd


def decode_masks(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(npz_path, allow_pickle=False) as archive:
        shape = tuple(int(value) for value in archive["mask_shape"])
        pixel_count = int(np.prod(shape))
        anomaly = np.unpackbits(
            archive["anomaly_masks"],
            axis=1,
            count=pixel_count,
        ).reshape((-1, *shape)).astype(bool)
        normal = np.unpackbits(
            archive["normal_masks"],
            axis=1,
            count=pixel_count,
        ).reshape((-1, *shape)).astype(bool)
    return anomaly, normal


def read_video_frame(capture: cv2.VideoCapture, frame_index: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"Cannot decode clip frame {frame_index}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def add_mask(axis: plt.Axes, mask: np.ndarray) -> None:
    overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
    overlay[mask] = (0.0, 0.85, 1.0, 0.38)
    axis.imshow(overlay)


def add_box(
    axis: plt.Axes,
    box: tuple[float, float, float, float],
    color: str,
    linewidth: float = 2.0,
) -> None:
    axis.add_patch(
        Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            fill=False,
            edgecolor=color,
            linewidth=linewidth,
        )
    )


def choose_cases(comparison: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    selected_frames: set[int] = set()
    selected: list[tuple[str, pd.Series]] = []

    def take(label: str, candidates: pd.DataFrame, order: str, ascending: bool) -> None:
        available = candidates[
            ~candidates["clip_frame"].isin(selected_frames)
        ].sort_values(order, ascending=ascending)
        if available.empty:
            return
        row = available.iloc[0]
        selected_frames.add(int(row["clip_frame"]))
        selected.append((label, row))

    take(
        "IoU SAM2 più elevata",
        comparison,
        "best_person_iou_sam2",
        False,
    )
    take(
        "Massimo miglioramento",
        comparison,
        "iou_delta_sam2_minus_anomalydino",
        False,
    )

    point_hits = comparison[comparison["point_hit_person_sam2"] == 1].copy()
    median_delta = point_hits["iou_delta_sam2_minus_anomalydino"].median()
    point_hits["distance_from_median_delta"] = np.abs(
        point_hits["iou_delta_sam2_minus_anomalydino"] - median_delta
    )
    take(
        "Miglioramento tipico",
        point_hits,
        "distance_from_median_delta",
        True,
    )
    take(
        "SAM2 peggiora la box",
        comparison,
        "iou_delta_sam2_minus_anomalydino",
        True,
    )
    take(
        "Punto fuori dalle persone",
        comparison[comparison["point_hit_person_sam2"] == 0],
        "best_person_iou_sam2",
        True,
    )
    take(
        "Fallimento nonostante il punto",
        comparison[
            (comparison["point_hit_person_sam2"] == 1)
            & (comparison["best_person_iou_sam2"] < 0.5)
        ],
        "best_person_iou_sam2",
        True,
    )
    return selected


def quantitative_figure(
    comparison: pd.DataFrame,
    summary: pd.DataFrame,
    output: Path,
) -> None:
    baseline = comparison["best_person_iou_anomalydino"].to_numpy()
    sam2 = comparison["best_person_iou_sam2"].to_numpy()
    delta = sam2 - baseline
    figure, axes = plt.subplots(2, 2, figsize=(13, 10))

    axes[0, 0].scatter(baseline, sam2, s=14, alpha=0.45)
    axes[0, 0].plot([0, 1], [0, 1], "k--", label="Nessun cambiamento")
    axes[0, 0].set(
        title="IoU per frame",
        xlabel="IoU componente AnomalyDINO",
        ylabel="IoU box SAM2",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axes[0, 0].legend()

    axes[0, 1].hist(delta, bins=40, color="#1976d2", alpha=0.8)
    axes[0, 1].axvline(0, color="black", linestyle="--")
    axes[0, 1].axvline(
        delta.mean(),
        color="#d32f2f",
        linestyle="-",
        label=f"Media {delta.mean():.3f}",
    )
    axes[0, 1].set(
        title=f"Variazione IoU: {int(np.sum(delta > 0))}/500 frame migliorati",
        xlabel="IoU SAM2 − IoU AnomalyDINO",
        ylabel="Frame",
    )
    axes[0, 1].legend()

    all_scope = summary[summary["scope"] == "all_500_frames"].copy()
    labels = ["IoU ≥ 0,3", "IoU ≥ 0,5"]
    x = np.arange(2)
    width = 0.36
    for index, row in enumerate(all_scope.itertuples()):
        axes[1, 0].bar(
            x + (index - 0.5) * width,
            [row.hit_rate_iou_0_3, row.hit_rate_iou_0_5],
            width,
            label=(
                "Componente AnomalyDINO"
                if "anomalydino" in row.method
                else "SAM2 da punto"
            ),
        )
    axes[1, 0].set_xticks(x, labels)
    axes[1, 0].set(
        title="Hit rate rispetto ad almeno una persona",
        ylabel="Hit rate",
        ylim=(0, 1.05),
    )
    axes[1, 0].legend()

    ordered = comparison.sort_values("clip_frame")
    baseline_roll = ordered["best_person_iou_anomalydino"].rolling(
        15, center=True, min_periods=1
    ).mean()
    sam2_roll = ordered["best_person_iou_sam2"].rolling(
        15, center=True, min_periods=1
    ).mean()
    axes[1, 1].plot(
        ordered["clip_frame"],
        baseline_roll,
        label="AnomalyDINO, media mobile 15",
    )
    axes[1, 1].plot(
        ordered["clip_frame"],
        sam2_roll,
        label="SAM2, media mobile 15",
    )
    axes[1, 1].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[1, 1].set(
        title="Qualità lungo la clip",
        xlabel="Frame della clip",
        ylabel="IoU",
        ylim=(0, 1),
    )
    axes[1, 1].legend()

    for axis in axes.flat:
        axis.grid(alpha=0.2)
    figure.suptitle(
        "E09 — raffinamento AnomalyDINO → punto automatico → SAM2",
        fontsize=15,
    )
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def qualitative_figure(
    comparison: pd.DataFrame,
    masks: np.ndarray,
    baseline_boxes: pd.DataFrame,
    ground_truth: pd.DataFrame,
    clip_path: Path,
    output: Path,
    cases_csv: Path,
) -> None:
    cases = choose_cases(comparison)
    capture = cv2.VideoCapture(str(clip_path))
    figure, axes = plt.subplots(2, 3, figsize=(16, 10.5))
    rows: list[dict] = []
    for axis, (label, row) in zip(axes.flat, cases):
        clip_frame = int(row["clip_frame"])
        source_frame = int(row["source_frame"])
        image = read_video_frame(capture, clip_frame)
        axis.imshow(image)
        add_mask(axis, masks[clip_frame])

        baseline = baseline_boxes[
            baseline_boxes["source_frame"] == source_frame
        ].iloc[0]
        add_box(
            axis,
            (
                baseline["candidate_xtl"],
                baseline["candidate_ytl"],
                baseline["candidate_xbr"],
                baseline["candidate_ybr"],
            ),
            "#ff9800",
        )
        add_box(
            axis,
            (
                row["sam_xtl"],
                row["sam_ytl"],
                row["sam_xbr"],
                row["sam_ybr"],
            ),
            "#00bcd4",
        )
        for gt in ground_truth[
            ground_truth["frame"] == source_frame
        ].itertuples():
            add_box(axis, (gt.xtl, gt.ytl, gt.xbr, gt.ybr), "#00e676", 1.5)
        axis.scatter(
            row["point_x"],
            row["point_y"],
            marker="x",
            color="yellow",
            s=70,
            linewidths=2,
        )
        axis.set_title(
            f"{label} — clip {clip_frame}\n"
            f"IoU AnomalyDINO {row['best_person_iou_anomalydino']:.3f} → "
            f"SAM2 {row['best_person_iou_sam2']:.3f}",
            fontsize=10,
        )
        axis.axis("off")
        rows.append(
            {
                "case": label,
                "source_frame": source_frame,
                "clip_frame": clip_frame,
                "point_hit_person": int(row["point_hit_person_sam2"]),
                "anomalydino_iou": row["best_person_iou_anomalydino"],
                "sam2_iou": row["best_person_iou_sam2"],
                "iou_delta": row["iou_delta_sam2_minus_anomalydino"],
            }
        )
    capture.release()
    figure.legend(
        handles=[
            Patch(facecolor="#00bcd4", alpha=0.38, label="Maschera SAM2"),
            Line2D([0], [0], color="#ff9800", lw=2, label="Box AnomalyDINO"),
            Line2D([0], [0], color="#00bcd4", lw=2, label="Box SAM2"),
            Line2D([0], [0], color="#00e676", lw=2, label="Box CVAT persona"),
            Line2D(
                [0],
                [0],
                marker="x",
                color="yellow",
                linestyle="None",
                label="Punto automatico",
            ),
        ],
        loc="lower center",
        ncol=5,
    )
    figure.suptitle("E09 — casi qualitativi del raffinamento SAM2", fontsize=15)
    figure.subplots_adjust(
        top=0.90,
        bottom=0.10,
        hspace=0.24,
        wspace=0.04,
    )
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    pd.DataFrame(rows).to_csv(cases_csv, index=False)


def normal_false_positive_figure(
    normal_results: pd.DataFrame,
    normal_masks: np.ndarray,
    normal_image_dir: Path,
    output: Path,
) -> None:
    executed = normal_results[normal_results["sam_executed"] == 1]
    figure, axes = plt.subplots(2, 2, figsize=(12, 10))
    for axis, (index, row) in zip(axes.flat, executed.iterrows()):
        image = cv2.imread(str(normal_image_dir / row["filename"]))
        if image is None:
            raise RuntimeError(f"Missing normal image {row['filename']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        axis.imshow(image)
        add_mask(axis, normal_masks[index])
        add_box(
            axis,
            (row["sam_xtl"], row["sam_ytl"], row["sam_xbr"], row["sam_ybr"]),
            "#00bcd4",
        )
        axis.scatter(
            row["point_x"],
            row["point_y"],
            marker="x",
            color="yellow",
            s=75,
            linewidths=2,
        )
        axis.set_title(
            f"{row['filename']} — score frame {row['anomaly_score']:.3f}\n"
            f"SAM score {row['sam_score']:.3f}",
            fontsize=10,
        )
        axis.axis("off")
    figure.suptitle(
        "E09 — SAM2 segmenta anche i quattro falsi positivi normali",
        fontsize=15,
    )
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("clip", type=Path)
    parser.add_argument("baseline_boxes", type=Path)
    parser.add_argument("ground_truth", type=Path)
    parser.add_argument("normal_images", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    comparison = pd.read_csv(args.checkpoint / "E09_per_frame_comparison.csv")
    summary = pd.read_csv(args.checkpoint / "E09_comparison_summary.csv")
    normal_results = pd.read_csv(
        args.checkpoint / "E09_sam2_normal_control.csv"
    )
    baseline_boxes = pd.read_csv(args.baseline_boxes)
    ground_truth = pd.read_csv(args.ground_truth)
    ground_truth = ground_truth[
        (ground_truth["tipo"] == "persona")
        & (ground_truth["outside"] == 0)
    ]
    anomaly_masks, normal_masks = decode_masks(
        args.checkpoint / "E09_sam2_masks_packbits.npz"
    )

    quantitative_figure(
        comparison,
        summary,
        args.output / "E09_quantitative_comparison.png",
    )
    qualitative_figure(
        comparison,
        anomaly_masks,
        baseline_boxes,
        ground_truth,
        args.clip,
        args.output / "E09_qualitative_six_frames.png",
        args.output / "E09_qualitative_cases.csv",
    )
    normal_false_positive_figure(
        normal_results,
        normal_masks,
        args.normal_images,
        args.output / "E09_normal_false_positives.png",
    )

    delta = comparison["iou_delta_sam2_minus_anomalydino"]
    analysis = {
        "frames": len(comparison),
        "improved_frames": int(np.sum(delta > 0)),
        "unchanged_frames": int(np.sum(delta == 0)),
        "worsened_frames": int(np.sum(delta < 0)),
        "mean_iou_delta": float(delta.mean()),
        "median_iou_delta": float(delta.median()),
        "normal_false_positives_before_sam2": int(
            normal_results["frame_prediction"].sum()
        ),
        "normal_false_positives_after_sam2": int(
            normal_results["frame_prediction"].sum()
        ),
    }
    with (args.output / "E09_postprocessing_summary.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(analysis, file, indent=2)
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
