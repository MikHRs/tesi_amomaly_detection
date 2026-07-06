"""Metriche semplici per bounding box axis-aligned."""

from __future__ import annotations

from collections.abc import Sequence


Box = Sequence[float]


def box_area(box: Box) -> float:
    """Restituisce l'area; coordinate invertite producono area zero."""
    x1, y1, x2, y2 = map(float, box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou(box_a: Box, box_b: Box) -> float:
    """Calcola l'Intersection over Union di due box [x1, y1, x2, y2]."""
    ax1, ay1, ax2, ay2 = map(float, box_a)
    bx1, by1, bx2, by2 = map(float, box_b)

    intersection = box_area(
        [max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)]
    )
    union = box_area(box_a) + box_area(box_b) - intersection
    return intersection / union if union > 0 else 0.0


def match_boxes(
    pred_boxes: Sequence[Box],
    gt_boxes: Sequence[Box],
    iou_threshold: float = 0.5,
) -> tuple[int, int, int]:
    """Esegue matching greedy uno-a-uno e restituisce (TP, FP, FN).

    Le coppie sono considerate in ordine decrescente di IoU. In questo modo
    ogni predizione e ogni ground truth possono essere usate al massimo una
    volta.
    """
    if not 0 <= iou_threshold <= 1:
        raise ValueError("La soglia IoU deve essere compresa tra 0 e 1")

    candidates = sorted(
        (
            (iou(pred_box, gt_box), pred_index, gt_index)
            for pred_index, pred_box in enumerate(pred_boxes)
            for gt_index, gt_box in enumerate(gt_boxes)
        ),
        reverse=True,
    )

    matched_preds: set[int] = set()
    matched_gts: set[int] = set()
    for overlap, pred_index, gt_index in candidates:
        if overlap < iou_threshold:
            break
        if pred_index in matched_preds or gt_index in matched_gts:
            continue
        matched_preds.add(pred_index)
        matched_gts.add(gt_index)

    tp = len(matched_preds)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    return tp, fp, fn


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Calcola precision, recall e F1 con divisioni per zero sicure."""
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    return precision, recall, f1


def evaluate_frame(
    preds: Sequence[Box],
    gts: Sequence[Box],
    threshold: float = 0.5,
) -> dict[str, float | int]:
    """Valuta le box di un singolo frame."""
    tp, fp, fn = match_boxes(preds, gts, threshold)
    precision, recall, f1 = precision_recall_f1(tp, fp, fn)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main() -> None:
    ground_truth = [[10, 10, 100, 100], [150, 20, 200, 90]]
    predictions = [[20, 20, 110, 110], [300, 300, 350, 350]]
    result = evaluate_frame(predictions, ground_truth, threshold=0.5)
    print("Demo metriche (soglia IoU 0.5)")
    print(f"IoU prima coppia: {iou(predictions[0], ground_truth[0]):.3f}")
    print(result)


if __name__ == "__main__":
    main()
