"""Valuta predizioni di bounding box rispetto a ground truth CVAT convertita."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    from .metrics import match_boxes, precision_recall_f1
except ImportError:  # esecuzione diretta: python src/evaluate_predictions.py
    from metrics import match_boxes, precision_recall_f1


COORDINATES = ["xtl", "ytl", "xbr", "ybr"]
GT_REQUIRED = {"frame", "label", *COORDINATES}
PRED_REQUIRED = {"frame", "label", "score", "prompt", "method", *COORDINATES}


def canonical_frame(value: object) -> str:
    text = str(value).strip()
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def active_outside_mask(series: pd.Series) -> pd.Series:
    """True per box attive, tollerando 0/1 e false/true."""
    return ~series.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def validate_columns(dataframe: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(dataframe.columns)
    if missing:
        raise ValueError(f"{name}: colonne mancanti: {sorted(missing)}")


def prepare_data(gt_path: Path, pred_path: Path, score_threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carica, valida e normalizza i due CSV."""
    if not gt_path.is_file():
        raise FileNotFoundError(f"Ground truth non trovata: {gt_path}")
    if not pred_path.is_file():
        raise FileNotFoundError(f"Predizioni non trovate: {pred_path}")

    gt = pd.read_csv(gt_path)
    pred = pd.read_csv(pred_path)
    validate_columns(gt, GT_REQUIRED, "ground truth")
    validate_columns(pred, PRED_REQUIRED, "predizioni")

    if "outside" in gt.columns:
        gt = gt[active_outside_mask(gt["outside"])].copy()
    pred["score"] = pd.to_numeric(pred["score"], errors="raise")
    pred = pred[pred["score"] >= score_threshold].copy()

    for dataframe, name in ((gt, "ground truth"), (pred, "predizioni")):
        dataframe["frame"] = dataframe["frame"].map(canonical_frame)
        for column in COORDINATES:
            dataframe[column] = pd.to_numeric(dataframe[column], errors="raise")
        invalid = (dataframe["xbr"] <= dataframe["xtl"]) | (
            dataframe["ybr"] <= dataframe["ytl"]
        )
        if invalid.any():
            raise ValueError(
                f"{name}: trovate {int(invalid.sum())} box con coordinate non valide"
            )

    pred["prompt"] = pred["prompt"].fillna("<non_specificato>").astype(str)
    pred["method"] = pred["method"].fillna("<non_specificato>").astype(str)
    gt["label"] = gt["label"].astype(str)
    pred["label"] = pred["label"].astype(str)
    return gt, pred


def dataframe_boxes(dataframe: pd.DataFrame) -> list[list[float]]:
    return dataframe[COORDINATES].astype(float).values.tolist()


def evaluate_subset(
    gt: pd.DataFrame,
    pred: pd.DataFrame,
    iou_threshold: float,
    match_label: bool,
) -> dict[str, float | int]:
    """Valuta un insieme di predizioni su tutti i frame della ground truth."""
    tp = fp = fn = 0
    frames = sorted(set(gt["frame"]) | set(pred["frame"]))

    for frame in frames:
        gt_frame = gt[gt["frame"] == frame]
        pred_frame = pred[pred["frame"] == frame]
        if match_label:
            labels: Iterable[str] = sorted(
                set(gt_frame["label"]) | set(pred_frame["label"])
            )
        else:
            labels = ("<class_agnostic>",)

        for label in labels:
            if match_label:
                gt_part = gt_frame[gt_frame["label"] == label]
                pred_part = pred_frame[pred_frame["label"] == label]
            else:
                gt_part = gt_frame
                pred_part = pred_frame
            part_tp, part_fp, part_fn = match_boxes(
                dataframe_boxes(pred_part),
                dataframe_boxes(gt_part),
                iou_threshold,
            )
            tp += part_tp
            fp += part_fp
            fn += part_fn

    precision, recall, f1 = precision_recall_f1(tp, fp, fn)
    return {
        "frames_evaluated": len(frames),
        "gt_boxes": len(gt),
        "pred_boxes": len(pred),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluation_rows(
    gt: pd.DataFrame,
    pred: pd.DataFrame,
    iou_threshold: float,
    match_label: bool,
    score_threshold: float,
) -> list[dict[str, object]]:
    """Crea righe globali e aggregate per metodo/prompt."""
    rows: list[dict[str, object]] = []

    def add_row(
        scope: str,
        subset: pd.DataFrame,
        *,
        prompt: str = "<tutti>",
        method: str = "<tutti>",
    ) -> None:
        rows.append(
            {
                "scope": scope,
                "prompt": prompt,
                "method": method,
                "iou_threshold": iou_threshold,
                "score_threshold": score_threshold,
                "label_matching": match_label,
                **evaluate_subset(gt, subset, iou_threshold, match_label),
            }
        )

    add_row("global", pred)
    for method, subset in pred.groupby("method", sort=True, dropna=False):
        add_row("method", subset, method=str(method))
    for prompt, subset in pred.groupby("prompt", sort=True, dropna=False):
        add_row("prompt", subset, prompt=str(prompt))
    for (method, prompt), subset in pred.groupby(
        ["method", "prompt"], sort=True, dropna=False
    ):
        add_row("method_prompt", subset, method=str(method), prompt=str(prompt))
    return rows


def evaluate_files(
    gt_path: Path,
    pred_path: Path,
    output_path: Path,
    iou_threshold: float,
    score_threshold: float = 0.0,
    match_label: bool = False,
) -> pd.DataFrame:
    """Valuta due CSV e salva la tabella aggregata."""
    if not 0 <= iou_threshold <= 1:
        raise ValueError("--iou deve essere compreso tra 0 e 1")
    gt, pred = prepare_data(gt_path, pred_path, score_threshold)
    rows = evaluation_rows(
        gt, pred, iou_threshold, match_label, score_threshold
    )
    result = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Valuta predizioni di box.")
    parser.add_argument("--gt", type=Path, required=True, help="CSV ground truth")
    parser.add_argument("--pred", type=Path, required=True, help="CSV predizioni")
    parser.add_argument("--out", type=Path, required=True, help="CSV metriche")
    parser.add_argument("--iou", type=float, default=0.5, help="Soglia IoU")
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.0,
        help="Scarta predizioni con score inferiore",
    )
    parser.add_argument(
        "--match-label",
        action="store_true",
        help="Richiede label uguali; di default valuta la localizzazione class-agnostic",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = evaluate_files(
            args.gt,
            args.pred,
            args.out,
            args.iou,
            args.score_threshold,
            args.match_label,
        )
    except (FileNotFoundError, OSError, ValueError, pd.errors.ParserError) as exc:
        print(f"ERRORE: {exc}")
        return 2

    global_row = result.iloc[0]
    print(
        "Globale — "
        f"TP={int(global_row.tp)}, FP={int(global_row.fp)}, FN={int(global_row.fn)}, "
        f"precision={global_row.precision:.3f}, recall={global_row.recall:.3f}, "
        f"F1={global_row.f1:.3f}"
    )
    print(f"Metriche salvate: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
