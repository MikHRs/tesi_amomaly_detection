# %% [markdown]
# # E10 — AnomalyDINO multi-punto → SAM2
#
# Estensione pre-registrata di E09. Le mappe AnomalyDINO E05, la soglia di
# frame, SAM2 e le annotazioni restano congelati. Da ogni mappa si estraggono
# cinque massimi spazialmente separati e si valutano le configurazioni
# annidate K=1, K=3 e K=5. Le metriche distinguono persone, oggetti
# `incerto` sui binari e l'insieme completo delle annotazioni.
#
# Parametri fissati prima del test:
#
# - distanza minima tra punti: 64 px;
# - deduplicazione box SAM2: IoU >= 0,7;
# - matching one-to-one a IoU 0,3 e 0,5;
# - nessun fine-tuning e nessun prompt manuale;
# - tutte le configurazioni vengono riportate, senza selezione post-hoc.

# %%
from pathlib import Path
from IPython.display import FileLink, display
import hashlib
import json
import os
import platform
import gc
import shutil
import subprocess
import sys
import time
import zipfile

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

INPUT_ROOT = Path("/kaggle/input")
WORK = Path("/kaggle/working")

protocol_paths = list(
    INPUT_ROOT.rglob("E09_inputs/protocol.json")
)
assert len(protocol_paths) == 1, (
    "Collega lo stesso dataset di input usato per E09. "
    f"Trovati: {protocol_paths}"
)

E09_ROOT = protocol_paths[0].parent
input_manifest = json.loads(
    (E09_ROOT / "manifest.json").read_text()
)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


for item in input_manifest:
    path = E09_ROOT / item["file"]
    assert path.exists(), path
    assert path.stat().st_size == item["bytes"], path
    assert sha256_file(path) == item["sha256"], path

print("Input E09/E10:", E09_ROOT)
print("File verificati:", len(input_manifest))
print("CUDA:", torch.cuda.is_available())
print("GPU:", [
    torch.cuda.get_device_name(index)
    for index in range(torch.cuda.device_count())
])

# %% [markdown]
# ## Installazione riproducibile di SAM2
#
# In Kaggle deve essere attiva l'opzione Internet. Il notebook blocca commit e
# checksum già usati in E09.

# %%
SAM2_DIR = WORK / "sam2"
SAM2_COMMIT = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
CHECKPOINT_SHA256 = (
    "6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38"
)
CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything_2/"
    "092824/sam2.1_hiera_small.pt"
)

if not SAM2_DIR.exists():
    subprocess.run(
        [
            "git",
            "clone",
            "https://github.com/facebookresearch/sam2.git",
            str(SAM2_DIR),
        ],
        check=True,
    )

subprocess.run(
    ["git", "checkout", SAM2_COMMIT],
    cwd=SAM2_DIR,
    check=True,
)
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "-e",
        str(SAM2_DIR),
    ],
    check=True,
)

checkpoint_dir = SAM2_DIR / "checkpoints"
checkpoint_dir.mkdir(exist_ok=True)
checkpoint = checkpoint_dir / "sam2.1_hiera_small.pt"

if (
    not checkpoint.exists()
    or sha256_file(checkpoint) != CHECKPOINT_SHA256
):
    subprocess.run(
        [
            "wget",
            "-q",
            "--show-progress",
            "-O",
            str(checkpoint),
            CHECKPOINT_URL,
        ],
        check=True,
    )

assert sha256_file(checkpoint) == CHECKPOINT_SHA256

installed_commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"],
    cwd=SAM2_DIR,
    text=True,
).strip()
assert installed_commit == SAM2_COMMIT

print("Commit SAM2:", installed_commit)
print("SHA-256 checkpoint:", sha256_file(checkpoint))

# %% [markdown]
# ## Caricamento modello, dati e funzioni

# %%
os.chdir(SAM2_DIR)

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
FRAME_THRESHOLD = 0.28902236819267274
K_VALUES = (1, 3, 5)
MAX_CANDIDATES = max(K_VALUES)
MIN_POINT_DISTANCE_PX = 64
DEDUPLICATION_IOU = 0.7
IOU_THRESHOLDS = (0.3, 0.5)
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 512

sam2_model = build_sam2(
    MODEL_CONFIG,
    str(checkpoint),
    device="cuda",
)
predictor = SAM2ImagePredictor(sam2_model)

anomaly_manifest = pd.read_csv(
    E09_ROOT / "manifests/E09_anomaly_manifest.csv"
).sort_values("clip_frame")
normal_manifest = pd.read_csv(
    E09_ROOT / "manifests/E09_normal_test_manifest.csv"
)
ground_truth_all = pd.read_csv(
    E09_ROOT / "annotations/visible_ground_truth_clip.csv"
)
ground_truth_all = ground_truth_all[
    ground_truth_all["outside"] == 0
].copy()

evaluation_scopes = {
    "persona": ground_truth_all[
        ground_truth_all["tipo"] == "persona"
    ].copy(),
    "incerto_all": ground_truth_all[
        ground_truth_all["tipo"] == "incerto"
    ].copy(),
    "incerto_sui_binari": ground_truth_all[
        (ground_truth_all["tipo"] == "incerto")
        & (
            ground_truth_all["posizione"]
            == "sui_binari"
        )
    ].copy(),
    "all_sui_binari": ground_truth_all[
        ground_truth_all["posizione"] == "sui_binari"
    ].copy(),
    "all_visible": ground_truth_all.copy(),
}

with np.load(
    E09_ROOT / "anomalydino/raw_anomaly_maps_safe.npz",
    allow_pickle=False,
) as archive:
    raw_maps = archive["maps"].copy()

assert len(anomaly_manifest) == 500
assert len(normal_manifest) == 20
assert len(evaluation_scopes["persona"]) == 3732
assert len(evaluation_scopes["incerto_all"]) == 1473
assert len(
    evaluation_scopes["incerto_sui_binari"]
) == 1000
assert len(evaluation_scopes["all_visible"]) == 5205


def box_iou(first, second):
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(
        0.0,
        first[3] - first[1],
    )
    second_area = max(0.0, second[2] - second[0]) * max(
        0.0,
        second[3] - second[1],
    )
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def mask_to_box(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    )


def select_spatial_peaks(
    anomaly_map,
    count=MAX_CANDIDATES,
    minimum_distance=MIN_POINT_DISTANCE_PX,
    first_point=None,
):
    resized = cv2.resize(
        anomaly_map.astype(np.float32),
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        interpolation=cv2.INTER_LINEAR,
    )
    work = resized.copy()
    yy, xx = np.ogrid[
        :IMAGE_HEIGHT,
        :IMAGE_WIDTH,
    ]
    points = []

    if first_point is not None:
        point_x = int(first_point[0])
        point_y = int(first_point[1])
        peak_value = float(first_point[2])
        points.append({
            "candidate_rank": 1,
            "point_x": point_x,
            "point_y": point_y,
            "peak_value": peak_value,
        })
        suppressed = (
            (xx - point_x) ** 2
            + (yy - point_y) ** 2
            <= minimum_distance ** 2
        )
        work[suppressed] = -np.inf

    first_new_rank = 2 if first_point is not None else 1

    for rank in range(first_new_rank, count + 1):
        flat_index = int(np.argmax(work))
        point_y, point_x = np.unravel_index(
            flat_index,
            work.shape,
        )
        peak_value = float(work[point_y, point_x])
        assert np.isfinite(peak_value)
        points.append({
            "candidate_rank": rank,
            "point_x": int(point_x),
            "point_y": int(point_y),
            "peak_value": peak_value,
        })
        suppressed = (
            (xx - point_x) ** 2
            + (yy - point_y) ** 2
            <= minimum_distance ** 2
        )
        work[suppressed] = -np.inf

    return points


def predict_points(image_rgb, points):
    predictions = []
    with torch.inference_mode():
        predictor.set_image(image_rgb)
        for point in points:
            coordinates = np.array(
                [[point["point_x"], point["point_y"]]],
                dtype=np.float32,
            )
            labels = np.array([1], dtype=np.int32)
            masks, scores, _ = predictor.predict(
                point_coords=coordinates,
                point_labels=labels,
                multimask_output=True,
            )
            best_index = int(np.argmax(scores))
            mask = masks[best_index].astype(bool)
            predictions.append({
                **point,
                "sam_score": float(scores[best_index]),
                "mask": mask,
                "box": mask_to_box(mask),
            })
    return predictions


def deduplicate_prefix(frame_predictions, k):
    kept = []
    for prediction in frame_predictions:
        if prediction["candidate_rank"] > k:
            continue
        box = prediction["box"]
        if box is None:
            continue
        duplicate = any(
            box_iou(box, previous["box"])
            >= DEDUPLICATION_IOU
            for previous in kept
        )
        if not duplicate:
            kept.append(prediction)
    return kept


def greedy_match(prediction_boxes, gt_boxes, threshold):
    candidates = []
    for prediction_index, prediction in enumerate(
        prediction_boxes
    ):
        for gt_index, gt in enumerate(gt_boxes):
            iou = box_iou(prediction, gt)
            if iou >= threshold:
                candidates.append(
                    (iou, prediction_index, gt_index)
                )
    candidates.sort(reverse=True)
    used_predictions = set()
    used_gt = set()
    matches = []
    for iou, prediction_index, gt_index in candidates:
        if prediction_index in used_predictions:
            continue
        if gt_index in used_gt:
            continue
        used_predictions.add(prediction_index)
        used_gt.add(gt_index)
        matches.append(
            (prediction_index, gt_index, iou)
        )
    return matches


def map_for_row(row):
    if hasattr(row, "npz_index"):
        return raw_maps[int(row.npz_index)]
    if hasattr(row, "clip_frame"):
        # Ordine E05 congelato: 20 calibrazione, 20 normal test,
        # quindi i 500 frame anomali.
        return raw_maps[40 + int(row.clip_frame)]
    raise ValueError("Indice della mappa non disponibile")


print("Anomaly manifest:", anomaly_manifest.shape)
print("Normal manifest:", normal_manifest.shape)
print("Ground truth per scope:", {
    scope: len(table)
    for scope, table in evaluation_scopes.items()
})
print("Raw maps:", raw_maps.shape)
print(
    "Protocollo:",
    {
        "K": K_VALUES,
        "distanza_px": MIN_POINT_DISTANCE_PX,
        "dedup_iou": DEDUPLICATION_IOU,
        "iou_metriche": IOU_THRESHOLDS,
    },
)

# %% [markdown]
# ## Controllo visivo prima dell'inferenza completa
#
# Deve apparire lo stesso primo punto di E09, più quattro proposte
# spazialmente separate. Le box verdi sono la ground truth e non entrano nella
# generazione dei punti.

# %%
probe_clip_frame = 250
probe_row = anomaly_manifest[
    anomaly_manifest["clip_frame"] == probe_clip_frame
].iloc[0]

capture = cv2.VideoCapture(
    str(E09_ROOT / "clip/san_donato_pilot_85_105.mp4")
)
capture.set(cv2.CAP_PROP_POS_FRAMES, probe_clip_frame)
ok, probe_bgr = capture.read()
capture.release()
assert ok

probe_rgb = cv2.cvtColor(
    probe_bgr,
    cv2.COLOR_BGR2RGB,
)
probe_points = select_spatial_peaks(
    raw_maps[int(probe_row["npz_index"])]
    if "npz_index" in probe_row.index
    else raw_maps[40 + probe_clip_frame],
    first_point=(
        probe_row["max_x"],
        probe_row["max_y"],
        probe_row["max_value"],
    ),
)
probe_predictions = predict_points(
    probe_rgb,
    probe_points,
)

fig, axis = plt.subplots(figsize=(11, 8))
axis.imshow(probe_rgb)
colors = ["yellow", "cyan", "magenta", "orange", "white"]

for prediction, color in zip(
    probe_predictions,
    colors,
):
    axis.scatter(
        prediction["point_x"],
        prediction["point_y"],
        color=color,
        marker="x",
        s=100,
        linewidths=3,
    )
    if prediction["box"] is not None:
        x1, y1, x2, y2 = prediction["box"]
        axis.add_patch(Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            edgecolor=color,
            linewidth=2,
        ))
        axis.text(
            x1,
            max(12, y1 - 4),
            f"P{prediction['candidate_rank']}",
            color=color,
            fontsize=10,
            weight="bold",
        )

probe_gt = ground_truth_all[
    ground_truth_all["frame"]
    == int(probe_row["source_frame"])
]
for gt in probe_gt.itertuples():
    if (
        gt.tipo == "incerto"
        and gt.posizione == "sui_binari"
    ):
        gt_color = "deepskyblue"
    elif gt.tipo == "incerto":
        gt_color = "red"
    else:
        gt_color = "lime"
    axis.add_patch(Rectangle(
        (gt.xtl, gt.ytl),
        gt.xbr - gt.xtl,
        gt.ybr - gt.ytl,
        fill=False,
        edgecolor=gt_color,
        linewidth=1.5,
    ))

axis.set_title(
    "E10 probe — cinque punti automatici separati di 64 px\n"
    "CVAT: verde persona, azzurro incerto sui binari, rosso altro incerto"
)
axis.axis("off")
plt.show()

display(pd.DataFrame([
    {
        key: value
        for key, value in prediction.items()
        if key not in {"mask"}
    }
    for prediction in probe_predictions
]))

# %% [markdown]
# ## Inferenza completa
#
# La feature di immagine SAM2 viene calcolata una sola volta per frame; i
# cinque prompt sono poi valutati separatamente. Al termine viene creato
# subito `E10_inference_checkpoint.zip`.

# %%
raw_prediction_rows = []
packed_masks = []
frame_predictions_by_clip = {}
prediction_id = 0

torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
start_time = time.time()

capture = cv2.VideoCapture(
    str(E09_ROOT / "clip/san_donato_pilot_85_105.mp4")
)

for row in tqdm(
    anomaly_manifest.itertuples(index=False),
    total=len(anomaly_manifest),
    desc="E10 SAM2 — 500 frame anomali",
):
    ok, frame_bgr = capture.read()
    assert ok, row.clip_frame
    frame_rgb = cv2.cvtColor(
        frame_bgr,
        cv2.COLOR_BGR2RGB,
    )
    gate = int(
        getattr(row, "predicted_anomaly", 1)
    ) == 1
    points = (
        select_spatial_peaks(
            map_for_row(row),
            first_point=(
                row.max_x,
                row.max_y,
                row.max_value,
            ),
        )
        if gate
        else []
    )
    predictions = (
        predict_points(frame_rgb, points)
        if points
        else []
    )
    frame_predictions_by_clip[int(row.clip_frame)] = (
        predictions
    )

    for prediction in predictions:
        box = prediction["box"]
        raw_prediction_rows.append({
            "prediction_id": prediction_id,
            "split": "anomaly_test",
            "source_frame": int(row.source_frame),
            "clip_frame": int(row.clip_frame),
            "frame_prediction": int(gate),
            "candidate_rank":
                prediction["candidate_rank"],
            "point_x": prediction["point_x"],
            "point_y": prediction["point_y"],
            "peak_value": prediction["peak_value"],
            "sam_score": prediction["sam_score"],
            "mask_area": int(
                prediction["mask"].sum()
            ),
            "sam_xtl": box[0] if box else np.nan,
            "sam_ytl": box[1] if box else np.nan,
            "sam_xbr": box[2] if box else np.nan,
            "sam_ybr": box[3] if box else np.nan,
        })
        packed_masks.append(
            np.packbits(
                prediction["mask"].reshape(-1)
            )
        )
        prediction["prediction_id"] = prediction_id
        del prediction["mask"]
        prediction_id += 1

capture.release()

normal_frame_rows = []

for row in tqdm(
    normal_manifest.itertuples(index=False),
    total=len(normal_manifest),
    desc="E10 SAM2 — 20 controlli normali",
):
    gate = int(row.predicted_anomaly) == 1
    predictions = []

    if gate:
        image_path = E09_ROOT / row.image_relative_path
        frame_bgr = cv2.imread(str(image_path))
        assert frame_bgr is not None, image_path
        frame_rgb = cv2.cvtColor(
            frame_bgr,
            cv2.COLOR_BGR2RGB,
        )
        points = select_spatial_peaks(
            raw_maps[int(row.npz_index)]
        )
        predictions = predict_points(
            frame_rgb,
            points,
        )

        for prediction in predictions:
            box = prediction["box"]
            raw_prediction_rows.append({
                "prediction_id": prediction_id,
                "split": "normal_test",
                "source_frame": int(row.source_frame),
                "clip_frame": np.nan,
                "frame_prediction": int(gate),
                "candidate_rank":
                    prediction["candidate_rank"],
                "point_x": prediction["point_x"],
                "point_y": prediction["point_y"],
                "peak_value": prediction["peak_value"],
                "sam_score": prediction["sam_score"],
                "mask_area": int(
                    prediction["mask"].sum()
                ),
                "sam_xtl": box[0] if box else np.nan,
                "sam_ytl": box[1] if box else np.nan,
                "sam_xbr": box[2] if box else np.nan,
                "sam_ybr": box[3] if box else np.nan,
            })
            packed_masks.append(
                np.packbits(
                    prediction["mask"].reshape(-1)
                )
            )
            prediction["prediction_id"] = prediction_id
            del prediction["mask"]
            prediction_id += 1

    normal_output = {
        "filename": row.filename,
        "source_frame": int(row.source_frame),
        "anomaly_score": float(row.anomaly_score),
        "frame_prediction": int(gate),
        "raw_spatial_proposals": len(predictions),
    }
    for k in K_VALUES:
        normal_output[
            f"deduplicated_proposals_k{k}"
        ] = len(deduplicate_prefix(predictions, k))
    normal_frame_rows.append(normal_output)

torch.cuda.synchronize()
inference_seconds = time.time() - start_time
peak_gpu_gb = (
    torch.cuda.max_memory_allocated() / 1024**3
)

raw_predictions = pd.DataFrame(raw_prediction_rows)
normal_control = pd.DataFrame(normal_frame_rows)
normal_summary = pd.DataFrame([
    {
        "k": k,
        "normal_frames": len(normal_control),
        "frame_false_positives": int(
            normal_control["frame_prediction"].sum()
        ),
        "frame_false_positive_rate": float(
            normal_control["frame_prediction"].mean()
        ),
        "spatial_proposals": int(
            normal_control[
                f"deduplicated_proposals_k{k}"
            ].sum()
        ),
    }
    for k in K_VALUES
])

raw_predictions.to_csv(
    WORK / "E10_raw_predictions.csv",
    index=False,
)
normal_control.to_csv(
    WORK / "E10_normal_control.csv",
    index=False,
)
normal_summary.to_csv(
    WORK / "E10_normal_summary.csv",
    index=False,
)
np.savez_compressed(
    WORK / "E10_masks_packbits.npz",
    masks=np.stack(packed_masks).astype(np.uint8),
    prediction_ids=raw_predictions[
        "prediction_id"
    ].to_numpy(dtype=np.int32),
    mask_shape=np.array(
        [IMAGE_HEIGHT, IMAGE_WIDTH],
        dtype=np.int16,
    ),
)
del packed_masks
gc.collect()

inference_metadata = {
    "experiment_id":
        "E10_v1_anomalydino_multipoint_sam2",
    "inference_seconds": inference_seconds,
    "sam_masks": len(raw_predictions),
    "sam_masks_per_second":
        len(raw_predictions) / inference_seconds,
    "peak_gpu_gb": peak_gpu_gb,
    "frames_anomaly": len(anomaly_manifest),
    "frames_normal_control": len(normal_manifest),
    "normal_frames_passing_gate": int(
        normal_control["frame_prediction"].sum()
    ),
    "k_values": list(K_VALUES),
    "minimum_point_distance_px":
        MIN_POINT_DISTANCE_PX,
    "deduplication_iou": DEDUPLICATION_IOU,
    "sam2_commit": SAM2_COMMIT,
    "sam2_checkpoint_sha256":
        CHECKPOINT_SHA256,
    "fine_tuning": False,
}
(WORK / "E10_inference_metadata.json").write_text(
    json.dumps(inference_metadata, indent=2)
)

checkpoint_folder = WORK / "E10_inference_checkpoint"
if checkpoint_folder.exists():
    shutil.rmtree(checkpoint_folder)
checkpoint_folder.mkdir()

for filename in [
    "E10_raw_predictions.csv",
    "E10_normal_control.csv",
    "E10_normal_summary.csv",
    "E10_masks_packbits.npz",
    "E10_inference_metadata.json",
]:
    shutil.copy2(
        WORK / filename,
        checkpoint_folder / filename,
    )

inference_zip = WORK / "E10_inference_checkpoint.zip"
with zipfile.ZipFile(
    inference_zip,
    "w",
    zipfile.ZIP_DEFLATED,
) as archive:
    for path in sorted(checkpoint_folder.iterdir()):
        archive.write(
            path,
            arcname=(
                "E10_inference_checkpoint/"
                f"{path.name}"
            ),
        )

print("Inferenza:", round(inference_seconds, 2), "s")
print("Maschere:", len(raw_predictions))
print(
    "Maschere/s:",
    round(
        len(raw_predictions) / inference_seconds,
        3,
    ),
)
print("Picco GPU:", round(peak_gpu_gb, 2), "GB")
print("\nControllo negativo:")
display(normal_summary)
print("ZIP:", inference_zip)
print("SHA-256:", sha256_file(inference_zip))
display(FileLink(str(inference_zip)))

# %% [markdown]
# ## Matching multi-oggetto e metriche

# %%
per_frame_rows = []
summary_rows = []
deduplicated_rows = []
deduplicated_by_key = {}

for k in K_VALUES:
    for row in anomaly_manifest.itertuples(index=False):
        predictions = deduplicate_prefix(
            frame_predictions_by_clip[int(row.clip_frame)],
            k,
        )
        deduplicated_by_key[
            (k, int(row.clip_frame))
        ] = predictions

        for prediction in predictions:
            deduplicated_rows.append({
                "k": k,
                "prediction_id":
                    prediction["prediction_id"],
                "source_frame": int(row.source_frame),
                "clip_frame": int(row.clip_frame),
                "candidate_rank":
                    prediction["candidate_rank"],
            })

for scope, scope_ground_truth in evaluation_scopes.items():
    gt_by_source_frame = {
        int(source_frame): [
            (
                float(row.xtl),
                float(row.ytl),
                float(row.xbr),
                float(row.ybr),
            )
            for row in group.itertuples()
        ]
        for source_frame, group
        in scope_ground_truth.groupby("frame")
    }

    for k in K_VALUES:
        aggregate = {
            threshold: {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "matched_ious": [],
                "frames_with_match": 0,
            }
            for threshold in IOU_THRESHOLDS
        }
        total_predictions = 0

        for row in anomaly_manifest.itertuples(
            index=False
        ):
            predictions = deduplicated_by_key[
                (k, int(row.clip_frame))
            ]
            prediction_boxes = [
                prediction["box"]
                for prediction in predictions
            ]
            gt_boxes = gt_by_source_frame.get(
                int(row.source_frame),
                [],
            )
            total_predictions += len(prediction_boxes)

            frame_output = {
                "scope": scope,
                "k": k,
                "source_frame": int(row.source_frame),
                "clip_frame": int(row.clip_frame),
                "predictions": len(prediction_boxes),
                "gt_boxes": len(gt_boxes),
            }

            for threshold in IOU_THRESHOLDS:
                matches = greedy_match(
                    prediction_boxes,
                    gt_boxes,
                    threshold,
                )
                tp = len(matches)
                fp = len(prediction_boxes) - tp
                fn = len(gt_boxes) - tp
                state = aggregate[threshold]
                state["tp"] += tp
                state["fp"] += fp
                state["fn"] += fn
                state["matched_ious"].extend(
                    match[2] for match in matches
                )
                state["frames_with_match"] += int(
                    tp > 0
                )
                suffix = str(threshold).replace(
                    ".",
                    "_",
                )
                frame_output[f"tp_iou_{suffix}"] = tp
                frame_output[f"fp_iou_{suffix}"] = fp
                frame_output[f"fn_iou_{suffix}"] = fn

            per_frame_rows.append(frame_output)

        for threshold in IOU_THRESHOLDS:
            state = aggregate[threshold]
            precision = (
                state["tp"]
                / (state["tp"] + state["fp"])
                if state["tp"] + state["fp"]
                else 0.0
            )
            recall = (
                state["tp"]
                / (state["tp"] + state["fn"])
                if state["tp"] + state["fn"]
                else 0.0
            )
            f1 = (
                2 * precision * recall
                / (precision + recall)
                if precision + recall
                else 0.0
            )
            summary_rows.append({
                "scope": scope,
                "k": k,
                "iou_threshold": threshold,
                "frames": len(anomaly_manifest),
                "predictions": total_predictions,
                "gt_boxes": len(scope_ground_truth),
                "tp": state["tp"],
                "fp": state["fp"],
                "fn": state["fn"],
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "mean_matched_iou": (
                    float(np.mean(
                        state["matched_ious"]
                    ))
                    if state["matched_ious"]
                    else 0.0
                ),
                "frames_with_match":
                    state["frames_with_match"],
                "frame_hit_rate":
                    state["frames_with_match"]
                    / len(anomaly_manifest),
                "mean_predictions_per_frame":
                    total_predictions
                    / len(anomaly_manifest),
            })

per_frame_metrics = pd.DataFrame(per_frame_rows)
detection_summary = pd.DataFrame(summary_rows)
deduplicated_predictions = pd.DataFrame(
    deduplicated_rows
)

per_frame_metrics.to_csv(
    WORK / "E10_per_frame_metrics.csv",
    index=False,
)
detection_summary.to_csv(
    WORK / "E10_detection_summary.csv",
    index=False,
)
deduplicated_predictions.to_csv(
    WORK / "E10_deduplicated_predictions.csv",
    index=False,
)

print("Metriche E10:")
display(detection_summary)

print("Controllo di riproduzione E09 (K=1):")
display(
    detection_summary[
        (detection_summary["k"] == 1)
        & (detection_summary["scope"] == "persona")
    ]
)

# %% [markdown]
# ## Figure quantitative e qualitative

# %%
fig, axes = plt.subplots(
    3,
    3,
    figsize=(16, 13),
)

plot_scopes = [
    "persona",
    "incerto_sui_binari",
    "all_sui_binari",
]
plot_metrics = ["precision", "recall", "f1"]

for row_index, scope in enumerate(plot_scopes):
    for column_index, metric in enumerate(plot_metrics):
        axis = axes[row_index, column_index]
        for threshold, linestyle in [
            (0.3, "-"),
            (0.5, "--"),
        ]:
            subset = detection_summary[
                (
                    detection_summary["scope"]
                    == scope
                )
                & (
                    detection_summary[
                        "iou_threshold"
                    ] == threshold
                )
            ]
            axis.plot(
                subset["k"],
                subset[metric],
                marker="o",
                linestyle=linestyle,
                label=f"IoU ≥ {threshold}",
            )
        axis.set_title(
            f"{scope} — {metric}"
        )
        axis.set_xlabel("Budget massimo K")
        axis.set_ylabel(metric)
        axis.set_xticks(K_VALUES)
        axis.grid(alpha=0.25)
        axis.legend()

fig.suptitle(
    "E10 — compromesso multi-candidato AnomalyDINO → SAM2"
)
fig.tight_layout()
quantitative_figure = (
    WORK / "E10_quantitative_tradeoff.png"
)
fig.savefig(
    quantitative_figure,
    dpi=180,
    bbox_inches="tight",
)
plt.show()

selected_frames = [0, 250, 499]
fig, axes = plt.subplots(
    len(selected_frames),
    2,
    figsize=(13, 12),
)
capture = cv2.VideoCapture(
    str(E09_ROOT / "clip/san_donato_pilot_85_105.mp4")
)
candidate_colors = {
    1: "yellow",
    2: "cyan",
    3: "magenta",
    4: "orange",
    5: "white",
}
raw_box_columns = [
    "prediction_id",
    "point_x",
    "point_y",
    "sam_xtl",
    "sam_ytl",
    "sam_xbr",
    "sam_ybr",
]

for row_index, clip_frame in enumerate(selected_frames):
    capture.set(cv2.CAP_PROP_POS_FRAMES, clip_frame)
    ok, frame_bgr = capture.read()
    assert ok
    frame_rgb = cv2.cvtColor(
        frame_bgr,
        cv2.COLOR_BGR2RGB,
    )
    manifest_row = anomaly_manifest[
        anomaly_manifest["clip_frame"] == clip_frame
    ].iloc[0]
    gt_frame = ground_truth_all[
        ground_truth_all["frame"]
        == int(manifest_row["source_frame"])
    ]

    for column, k in enumerate([1, 5]):
        axis = axes[row_index, column]
        axis.imshow(frame_rgb)
        predictions = deduplicated_predictions[
            (deduplicated_predictions["k"] == k)
            & (
                deduplicated_predictions["clip_frame"]
                == clip_frame
            )
        ].merge(
            raw_predictions[raw_box_columns],
            on="prediction_id",
        )
        for prediction in predictions.itertuples():
            color = candidate_colors[
                int(prediction.candidate_rank)
            ]
            axis.add_patch(Rectangle(
                (prediction.sam_xtl, prediction.sam_ytl),
                prediction.sam_xbr - prediction.sam_xtl,
                prediction.sam_ybr - prediction.sam_ytl,
                fill=False,
                edgecolor=color,
                linewidth=2,
            ))
            axis.scatter(
                prediction.point_x,
                prediction.point_y,
                marker="x",
                color=color,
                s=55,
                linewidths=2,
            )
        for gt in gt_frame.itertuples():
            if (
                gt.tipo == "incerto"
                and gt.posizione == "sui_binari"
            ):
                gt_color = "deepskyblue"
            elif gt.tipo == "incerto":
                gt_color = "red"
            else:
                gt_color = "lime"
            axis.add_patch(Rectangle(
                (gt.xtl, gt.ytl),
                gt.xbr - gt.xtl,
                gt.ybr - gt.ytl,
                fill=False,
                edgecolor=gt_color,
                linewidth=1.2,
            ))
        axis.set_title(
            f"Clip {clip_frame} — K={k}, "
            f"{len(predictions)} predizioni"
        )
        axis.axis("off")

capture.release()
fig.suptitle(
    "E10 — box SAM2: K=1 contro K=5\n"
    "CVAT: verde persona, azzurro incerto sui binari, "
    "rosso altro incerto"
)
fig.tight_layout()
qualitative_figure = (
    WORK / "E10_qualitative_k1_vs_k5.png"
)
fig.savefig(
    qualitative_figure,
    dpi=140,
    bbox_inches="tight",
)
plt.show()
plt.close(fig)

# %% [markdown]
# ## Pacchetto finale

# %%
protocol = {
    "experiment_id":
        "E10_v1_anomalydino_multipoint_sam2",
    "status":
        "protocol fixed before execution",
    "source_experiment":
        "E05_v1_anomalydino_16shot",
    "control_experiment":
        "E09_v1_anomalydino_point_sam2",
    "frame_gate_threshold": FRAME_THRESHOLD,
    "k_values": list(K_VALUES),
    "maximum_candidates": MAX_CANDIDATES,
    "minimum_point_distance_px":
        MIN_POINT_DISTANCE_PX,
    "deduplication": {
        "method": "box IoU, preserve peak rank",
        "iou_threshold": DEDUPLICATION_IOU,
    },
    "matching": {
        "method":
            "greedy one-to-one descending IoU",
        "iou_thresholds":
            list(IOU_THRESHOLDS),
    },
    "sam2": {
        "commit": SAM2_COMMIT,
        "checkpoint":
            "sam2.1_hiera_small.pt",
        "checkpoint_sha256":
            CHECKPOINT_SHA256,
        "multimask_selection":
            "highest predicted IoU",
        "fine_tuning": False,
    },
    "selection_policy":
        "report K=1,3,5; no post-hoc best-K claim",
    "evaluation_scopes": {
        scope: len(table)
        for scope, table in evaluation_scopes.items()
    },
    "scope_correction": (
        "added before E10 inference after auditing "
        "the person-only emphasis of E01-E09"
    ),
}
(WORK / "E10_protocol.json").write_text(
    json.dumps(protocol, indent=2)
)

final_metadata = {
    **inference_metadata,
    "python": platform.python_version(),
    "pytorch": torch.__version__,
    "opencv": cv2.__version__,
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "ground_truth_counts": {
        scope: len(table)
        for scope, table in evaluation_scopes.items()
    },
    "outputs": [
        "E10_raw_predictions.csv",
        "E10_deduplicated_predictions.csv",
        "E10_per_frame_metrics.csv",
        "E10_detection_summary.csv",
        "E10_normal_control.csv",
        "E10_normal_summary.csv",
        "E10_masks_packbits.npz",
        "E10_quantitative_tradeoff.png",
        "E10_qualitative_k1_vs_k5.png",
        "E10_protocol.json",
    ],
}
(WORK / "E10_metadata.json").write_text(
    json.dumps(final_metadata, indent=2)
)

readme = f"""# E10 — AnomalyDINO multi-punto → SAM2

E10 estende E09 con massimi spazialmente separati e budget annidati
K=1,3,5. Tutte le configurazioni sono riportate senza selezione post-hoc.

- distanza minima punti: {MIN_POINT_DISTANCE_PX} px;
- deduplicazione box: IoU {DEDUPLICATION_IOU};
- ground truth persona: {len(evaluation_scopes["persona"])} box;
- ground truth incerto: {len(evaluation_scopes["incerto_all"])} box;
- incerto sui binari: {len(evaluation_scopes["incerto_sui_binari"])} box;
- tutte le annotazioni visibili: {len(evaluation_scopes["all_visible"])} box;
- fine-tuning: no;
- inferenza: {inference_seconds:.2f} s;
- maschere SAM2: {len(raw_predictions)};
- picco GPU: {peak_gpu_gb:.2f} GB.

La classificazione di frame resta quella congelata di E05: SAM2 raffina
proposte spaziali ma non modifica la decisione normale/anomalo.
"""
(WORK / "README_E10.md").write_text(readme)

final_folder = (
    WORK
    / "E10_v1_anomalydino_multipoint_sam2"
)
if final_folder.exists():
    shutil.rmtree(final_folder)
final_folder.mkdir()

final_files = [
    "E10_raw_predictions.csv",
    "E10_deduplicated_predictions.csv",
    "E10_per_frame_metrics.csv",
    "E10_detection_summary.csv",
    "E10_normal_control.csv",
    "E10_normal_summary.csv",
    "E10_masks_packbits.npz",
    "E10_inference_metadata.json",
    "E10_metadata.json",
    "E10_protocol.json",
    "E10_quantitative_tradeoff.png",
    "E10_qualitative_k1_vs_k5.png",
    "README_E10.md",
]

for filename in final_files:
    shutil.copy2(
        WORK / filename,
        final_folder / filename,
    )

output_manifest = []
for path in sorted(final_folder.iterdir()):
    output_manifest.append({
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    })
(final_folder / "manifest.json").write_text(
    json.dumps(output_manifest, indent=2)
)

final_zip = (
    WORK
    / "E10_v1_anomalydino_multipoint_sam2_results.zip"
)
with zipfile.ZipFile(
    final_zip,
    "w",
    zipfile.ZIP_DEFLATED,
) as archive:
    for path in sorted(final_folder.iterdir()):
        archive.write(
            path,
            arcname=(
                "E10_v1_anomalydino_multipoint_sam2/"
                f"{path.name}"
            ),
        )

print("Pacchetto finale:", final_zip)
print("Dimensione:", final_zip.stat().st_size, "byte")
print("SHA-256:", sha256_file(final_zip))
display(FileLink(str(final_zip)))

print("\nRiepilogo finale:")
display(detection_summary)
