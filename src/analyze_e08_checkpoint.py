"""Post-process the E08 AnomalyDINO checkpoint without rerunning inference.

The script expects the files extracted from ``E08_base_checkpoint.zip``.  It
computes frame, clip and temporal-event metrics for the nested 16/64/200-shot
memories and saves compact CSV/JSON/PNG artefacts for the thesis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SHOT_COUNTS = (16, 64, 200)
SMOOTHING_WINDOWS = (1, 3, 5, 11)
TEMPORAL_IOU_THRESHOLDS = (0.1, 0.3, 0.5)


def binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predictions = scores >= threshold
    positives = labels == 1
    negatives = ~positives
    tp = int(np.sum(positives & predictions))
    fp = int(np.sum(negatives & predictions))
    fn = int(np.sum(positives & ~predictions))
    tn = int(np.sum(negatives & ~predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Mann-Whitney AUROC with average ranks for tied scores."""

    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    positive_rank_sum = ranks[labels == 1].sum()
    return float(
        (positive_rank_sum - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def precision_recall_curve(
    labels: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return a tie-aware precision-recall curve and average precision."""

    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    sorted_scores = scores[order]
    distinct = np.where(np.diff(sorted_scores))[0]
    threshold_indices = np.r_[distinct, len(scores) - 1]
    true_positives = np.cumsum(sorted_labels)[threshold_indices]
    false_positives = (
        1 + threshold_indices - true_positives
    )
    precision = true_positives / (true_positives + false_positives)
    total_positives = int(labels.sum())
    if total_positives == 0:
        return precision, np.zeros_like(precision), float("nan")
    recall = true_positives / total_positives
    recall_steps = np.diff(np.r_[0.0, recall])
    average_precision = float(np.sum(recall_steps * precision))
    return precision, recall, average_precision


def roc_curve(
    labels: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    sorted_scores = scores[order]
    distinct = np.where(np.diff(sorted_scores))[0]
    threshold_indices = np.r_[distinct, len(scores) - 1]
    true_positives = np.cumsum(sorted_labels)[threshold_indices]
    false_positives = (
        1 + threshold_indices - true_positives
    )
    positives = max(int(labels.sum()), 1)
    negatives = max(len(labels) - int(labels.sum()), 1)
    return (
        np.r_[0.0, false_positives / negatives],
        np.r_[0.0, true_positives / positives],
    )


def binary_segments(values: np.ndarray) -> list[tuple[int, int]]:
    """Convert a binary sequence to inclusive ``(start, end)`` segments."""

    values = np.asarray(values, dtype=np.int8)
    padded = np.pad(values, (1, 1))
    changes = np.diff(padded)
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0] - 1
    return list(zip(starts.tolist(), ends.tolist()))


def temporal_iou(first: tuple[int, int], second: tuple[int, int]) -> float:
    intersection = max(
        0,
        min(first[1], second[1]) - max(first[0], second[0]) + 1,
    )
    union = (
        first[1] - first[0] + 1
        + second[1] - second[0] + 1
        - intersection
    )
    return intersection / union if union else 0.0


def greedy_temporal_matches(
    ground_truth: list[tuple[int, int]],
    predictions: list[tuple[int, int]],
    threshold: float,
) -> list[float]:
    candidates = sorted(
        (
            (temporal_iou(gt, prediction), gt_index, prediction_index)
            for gt_index, gt in enumerate(ground_truth)
            for prediction_index, prediction in enumerate(predictions)
        ),
        reverse=True,
    )
    used_ground_truth: set[int] = set()
    used_predictions: set[int] = set()
    matches: list[float] = []
    for overlap, gt_index, prediction_index in candidates:
        if overlap < threshold:
            break
        if (
            gt_index in used_ground_truth
            or prediction_index in used_predictions
        ):
            continue
        used_ground_truth.add(gt_index)
        used_predictions.add(prediction_index)
        matches.append(overlap)
    return matches


def add_smoothed_scores(results: pd.DataFrame) -> pd.DataFrame:
    results = results.sort_values(
        ["date", "clip", "frame_index"]
    ).reset_index(drop=True)
    for shot_count in SHOT_COUNTS:
        raw_column = f"score_{shot_count}_shot"
        for window in SMOOTHING_WINDOWS:
            output_column = f"{raw_column}_smooth_{window}"
            if window == 1:
                results[output_column] = results[raw_column]
            else:
                results[output_column] = (
                    results.groupby(["date", "clip"])[raw_column]
                    .transform(
                        lambda series: series.rolling(
                            window,
                            center=True,
                            min_periods=1,
                        ).mean()
                    )
                )
    return results


def frame_and_clip_metrics(
    results: pd.DataFrame,
    thresholds: dict[int, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = results["label"].to_numpy(dtype=np.int8)
    frame_rows: list[dict] = []
    clip_rows: list[dict] = []
    for shot_count in SHOT_COUNTS:
        for window in SMOOTHING_WINDOWS:
            column = f"score_{shot_count}_shot_smooth_{window}"
            scores = results[column].to_numpy(dtype=float)
            _, _, average_precision = precision_recall_curve(labels, scores)
            row = {
                "reference_images": shot_count,
                "smoothing_window": window,
                "score_column": column,
                "threshold": thresholds[shot_count],
                **binary_metrics(labels, scores, thresholds[shot_count]),
                "auroc": roc_auc(labels, scores),
                "average_precision": average_precision,
            }
            frame_rows.append(row)
            for (date, clip), group in results.groupby(
                ["date", "clip"],
                sort=True,
            ):
                clip_labels = group["label"].to_numpy(dtype=np.int8)
                clip_scores = group[column].to_numpy(dtype=float)
                _, _, clip_ap = precision_recall_curve(
                    clip_labels,
                    clip_scores,
                )
                clip_rows.append(
                    {
                        "reference_images": shot_count,
                        "smoothing_window": window,
                        "date": date,
                        "clip": clip,
                        "normal_frames": int(np.sum(clip_labels == 0)),
                        "anomaly_frames": int(np.sum(clip_labels == 1)),
                        "auroc": roc_auc(clip_labels, clip_scores),
                        "average_precision": clip_ap,
                        "normal_mean_score": float(
                            np.mean(clip_scores[clip_labels == 0])
                        ),
                        "anomaly_mean_score": float(
                            np.mean(clip_scores[clip_labels == 1])
                        ),
                    }
                )
    return pd.DataFrame(frame_rows), pd.DataFrame(clip_rows)


def event_metrics(
    results: pd.DataFrame,
    thresholds: dict[int, float],
) -> pd.DataFrame:
    rows: list[dict] = []
    for shot_count in SHOT_COUNTS:
        for window in SMOOTHING_WINDOWS:
            column = f"score_{shot_count}_shot_smooth_{window}"
            threshold = thresholds[shot_count]
            for temporal_threshold in TEMPORAL_IOU_THRESHOLDS:
                total_ground_truth = 0
                total_predictions = 0
                matches: list[float] = []
                for _, group in results.groupby(
                    ["date", "clip"],
                    sort=True,
                ):
                    group = group.sort_values("frame_index")
                    ground_truth = binary_segments(
                        group["label"].to_numpy(dtype=np.int8)
                    )
                    predictions = binary_segments(
                        (
                            group[column].to_numpy(dtype=float)
                            >= threshold
                        ).astype(np.int8)
                    )
                    total_ground_truth += len(ground_truth)
                    total_predictions += len(predictions)
                    matches.extend(
                        greedy_temporal_matches(
                            ground_truth,
                            predictions,
                            temporal_threshold,
                        )
                    )
                tp = len(matches)
                fp = total_predictions - tp
                fn = total_ground_truth - tp
                precision = tp / (tp + fp) if tp + fp else 0.0
                recall = tp / (tp + fn) if tp + fn else 0.0
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if precision + recall
                    else 0.0
                )
                rows.append(
                    {
                        "reference_images": shot_count,
                        "smoothing_window": window,
                        "temporal_iou_threshold": temporal_threshold,
                        "gt_events": total_ground_truth,
                        "predicted_events": total_predictions,
                        "tp_events": tp,
                        "fp_events": fp,
                        "fn_events": fn,
                        "event_precision": precision,
                        "event_recall": recall,
                        "event_f1": f1,
                        "mean_matched_tiou": (
                            float(np.mean(matches)) if matches else 0.0
                        ),
                    }
                )
    return pd.DataFrame(rows)


def make_overview(
    results: pd.DataFrame,
    frame_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    labels = results["label"].to_numpy(dtype=np.int8)
    raw = frame_summary[frame_summary["smoothing_window"] == 1]
    figure, axes = plt.subplots(2, 2, figsize=(13, 10))

    for shot_count, color in zip(SHOT_COUNTS, ("#1976d2", "#ef6c00", "#2e7d32")):
        scores = results[f"score_{shot_count}_shot"].to_numpy(dtype=float)
        false_positive_rate, true_positive_rate = roc_curve(labels, scores)
        precision, recall, average_precision = precision_recall_curve(
            labels,
            scores,
        )
        auroc = float(
            raw.loc[
                raw["reference_images"] == shot_count,
                "auroc",
            ].iloc[0]
        )
        axes[0, 0].plot(
            false_positive_rate,
            true_positive_rate,
            color=color,
            label=f"{shot_count}-shot, AUROC {auroc:.3f}",
        )
        axes[0, 1].plot(
            recall,
            precision,
            color=color,
            label=f"{shot_count}-shot, AP {average_precision:.3f}",
        )

    axes[0, 0].plot([0, 1], [0, 1], "k--", label="Caso casuale")
    axes[0, 0].set(title="Curve ROC", xlabel="False positive rate", ylabel="True positive rate")
    axes[0, 1].axhline(labels.mean(), color="black", linestyle="--", label=f"Prevalenza {labels.mean():.3f}")
    axes[0, 1].set(title="Curve Precision–Recall", xlabel="Recall", ylabel="Precision")

    x = np.arange(len(SHOT_COUNTS))
    axes[1, 0].plot(x, raw["precision"], "o-", label="Precision")
    axes[1, 0].plot(x, raw["recall"], "o-", label="Recall")
    axes[1, 0].plot(x, raw["f1"], "o-", label="F1")
    axes[1, 0].set_xticks(x, SHOT_COUNTS)
    axes[1, 0].set(
        title="Soglia calibrata sui normali",
        xlabel="Immagini normali di riferimento",
        ylabel="Metrica",
    )

    best_ranking = int(raw.sort_values("auroc", ascending=False).iloc[0]["reference_images"])
    best_scores = results[f"score_{best_ranking}_shot"].to_numpy(dtype=float)
    axes[1, 1].hist(
        best_scores[labels == 0],
        bins=35,
        alpha=0.65,
        density=True,
        label="Normali",
        color="#43a047",
    )
    axes[1, 1].hist(
        best_scores[labels == 1],
        bins=35,
        alpha=0.65,
        density=True,
        label="Anomali",
        color="#fb8c00",
    )
    threshold = float(
        raw.loc[
            raw["reference_images"] == best_ranking,
            "threshold",
        ].iloc[0]
    )
    axes[1, 1].axvline(
        threshold,
        color="red",
        linestyle="--",
        label=f"Soglia {threshold:.3f}",
    )
    axes[1, 1].set(
        title=f"Distribuzione score {best_ranking}-shot",
        xlabel="Anomaly score",
        ylabel="Densità",
    )

    for axis in axes.flat:
        axis.grid(alpha=0.2)
        axis.legend()
    figure.suptitle(
        "E08 — AnomalyDINO sul Thermal Anomaly Detection Dataset",
        fontsize=15,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    results = pd.read_csv(args.checkpoint / "E08_scores.csv")
    classification = pd.read_csv(
        args.checkpoint / "E08_classification_summary.csv"
    )
    thresholds = {
        int(row.reference_images): float(row.threshold)
        for row in classification.itertuples()
    }
    results = add_smoothed_scores(results)
    frame_summary, per_clip = frame_and_clip_metrics(results, thresholds)
    event_summary = event_metrics(results, thresholds)

    results.to_csv(args.output / "E08_scores_with_smoothing.csv", index=False)
    frame_summary.to_csv(args.output / "E08_smoothing_summary.csv", index=False)
    per_clip.to_csv(args.output / "E08_per_clip_metrics.csv", index=False)
    event_summary.to_csv(
        args.output / "E08_temporal_event_summary.csv",
        index=False,
    )
    make_overview(
        results,
        frame_summary,
        args.output / "E08_quantitative_overview.png",
    )

    raw = frame_summary[frame_summary["smoothing_window"] == 1]
    best_ranking = raw.sort_values(
        ["auroc", "average_precision"],
        ascending=False,
    ).iloc[0]
    best_frame_f1 = frame_summary.sort_values(
        "f1",
        ascending=False,
    ).iloc[0]
    tiou_03 = event_summary[
        event_summary["temporal_iou_threshold"] == 0.3
    ]
    best_event = tiou_03.sort_values(
        "event_f1",
        ascending=False,
    ).iloc[0]
    selection = {
        "best_raw_ranking": {
            "reference_images": int(best_ranking["reference_images"]),
            "auroc": float(best_ranking["auroc"]),
            "average_precision": float(best_ranking["average_precision"]),
        },
        "best_frame_f1_at_fixed_calibrated_threshold": {
            "reference_images": int(best_frame_f1["reference_images"]),
            "smoothing_window": int(best_frame_f1["smoothing_window"]),
            "precision": float(best_frame_f1["precision"]),
            "recall": float(best_frame_f1["recall"]),
            "f1": float(best_frame_f1["f1"]),
        },
        "best_event_f1_at_tiou_0_3": {
            "reference_images": int(best_event["reference_images"]),
            "smoothing_window": int(best_event["smoothing_window"]),
            "predicted_events": int(best_event["predicted_events"]),
            "tp_events": int(best_event["tp_events"]),
            "fp_events": int(best_event["fp_events"]),
            "fn_events": int(best_event["fn_events"]),
            "event_precision": float(best_event["event_precision"]),
            "event_recall": float(best_event["event_recall"]),
            "event_f1": float(best_event["event_f1"]),
        },
    }
    with (args.output / "E08_model_selection.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(selection, file, indent=2)
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
