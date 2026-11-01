import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as functional
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from timm.models._helpers import load_checkpoint

from model import HRNetW18Lite
from utils import save_endpoint_visualization, save_report_visualizations

ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("ECHONET_DATA_ROOT", str(ROOT / "data"))).resolve()
PROCESSED_ROOT = DATA_ROOT / "processed"
OUTPUT_ROOT = Path(os.environ.get("HRNET_OUTPUT_ROOT", str(ROOT / "outputs"))).resolve()
PRETRAINED_PATH = ROOT / "weights" / "pretrained" / "model_w18.safetensors"


def patient_partition(table):
    patients = table.HashedFileName.astype(str).drop_duplicates().to_numpy(copy=True)
    np.random.default_rng(42).shuffle(patients)
    training_end = int(len(patients) * 0.6)
    validation_end = training_end + int(len(patients) * 0.2)
    groups = (patients[:training_end], patients[training_end:validation_end], patients[validation_end:])
    return {
        name: np.flatnonzero(table.HashedFileName.astype(str).isin(group).to_numpy())
        for name, group in zip(("train", "val", "test"), groups)
    }


def load_table():
    table = pd.read_csv(PROCESSED_ROOT / "measurements.csv", encoding="utf-8-sig")
    normalized = table.Calc.astype(str).str.strip().str.upper()
    mapping = {"IVSD": 0, "IVSS": 0, "LVPWD": 1, "LVPWS": 1,
               "LVIDD": 2, "LVIDS": 2, "LA": 3, "AORTIC_ROOT": 4, "AORTA": 5}
    if not normalized.isin(mapping).all():
        raise ValueError("Unexpected measurement task")
    table["task_id"] = normalized.map(mapping).astype(np.int64)
    table["label_name"] = normalized
    columns = ["X1_scaled", "Y1_scaled", "X2_scaled", "Y2_scaled", "scale_ratio"]
    if not np.isfinite(table[columns].to_numpy(dtype=float)).all() or (table.scale_ratio <= 0).any():
        raise ValueError("Nonfinite coordinates or invalid scale ratios")
    partitions = patient_partition(table)
    return table, partitions


class MeasurementDataset(Dataset):
    def __init__(self, table, indices):
        self.table = table
        self.indices = np.asarray(indices, dtype=np.int64)
        self.frames = None
        self.coordinates = table[["X1_scaled", "Y1_scaled", "X2_scaled", "Y2_scaled"]].to_numpy(dtype=np.float32)
        self.coordinates[:, [0, 2]] = self.coordinates[:, [0, 2]].clip(0, 639)
        self.coordinates[:, [1, 3]] = self.coordinates[:, [1, 3]].clip(0, 479)
        self.frame_indices = table.frame_array_index.to_numpy(dtype=np.int64)
        self.task_ids = table.task_id.to_numpy(dtype=np.int64)
        self.scales = table.scale_ratio.to_numpy(dtype=np.float32)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        if self.frames is None:
            self.frames = np.load(PROCESSED_ROOT / "frames.npy", mmap_mode="r", allow_pickle=False)
            if self.frames.ndim != 4 or self.frames.shape[1:] != (480, 640, 3) or self.frames.dtype != np.uint8:
                raise ValueError("Expected uint8 frames at 640x480 with three BGR channels")
            if self.frame_indices.min() < 0 or self.frame_indices.max() >= len(self.frames):
                raise ValueError("Frame index outside the extracted frame array")
        index = int(self.indices[item])
        image = self.frames[self.frame_indices[index]].astype(np.float32) / 255.0
        coordinates = self.coordinates[index]
        rows, columns = np.mgrid[0:480, 0:640]
        points = coordinates.reshape(2, 2)
        heatmaps = np.exp(-(
            (columns[None] - points[:, 0, None, None]) ** 2
            + (rows[None] - points[:, 1, None, None]) ** 2
        ) / (2.0 * 6.0 * 6.0)).astype(np.float32)
        return {
            "image": torch.from_numpy(image).permute(2, 0, 1),
            "mask": torch.from_numpy(heatmaps),
            "coords": torch.from_numpy(coordinates.copy()),
            "task_id": torch.tensor(self.task_ids[index], dtype=torch.long),
            "scale_ratio": torch.tensor(self.scales[index], dtype=torch.float32),
            "data_index": torch.tensor(index, dtype=torch.long),
        }


def make_training_loaders(table, partitions):
    common = dict(batch_size=32, num_workers=8, pin_memory=True,
                  persistent_workers=True, prefetch_factor=4)
    training = DataLoader(MeasurementDataset(table, partitions["train"]),
                          shuffle=True, drop_last=True, **common)
    validation = DataLoader(MeasurementDataset(table, partitions["val"]), shuffle=False, **common)
    return training, validation


def make_evaluation_loader(table, indices):
    return DataLoader(MeasurementDataset(table, indices), batch_size=32,
                      num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)


def decode_endpoints(logits):
    probabilities = logits.flatten(2).softmax(dim=-1).reshape_as(logits)
    rows, columns = torch.meshgrid(
        torch.arange(logits.shape[-2], device=logits.device, dtype=probabilities.dtype),
        torch.arange(logits.shape[-1], device=logits.device, dtype=probabilities.dtype),
        indexing="ij",
    )
    mass = probabilities.sum(dim=(-2, -1)) + 1e-8
    predicted_rows = (rows * probabilities).sum(dim=(-2, -1)) / mass
    predicted_columns = (columns * probabilities).sum(dim=(-2, -1)) / mass
    return torch.stack([predicted_columns, predicted_rows], dim=-1).reshape(-1, 4)


def segment_lengths(coordinates):
    points = coordinates.reshape(-1, 2, 2)
    return torch.norm(points[:, 0] - points[:, 1], dim=1)


def sample_weights(coordinates):
    weights = (segment_lengths(coordinates) + 1e-6).pow(-2)
    return (weights / (weights.mean() + 1e-6)).clamp(0.25, 12.0)


def coordinate_errors(predicted, reference):
    swapped = reference[:, [2, 3, 0, 1]]
    return torch.minimum((predicted - reference).abs().mean(dim=1),
                         (predicted - swapped).abs().mean(dim=1))


def heatmap_errors(logits, reference):
    probabilities = logits.sigmoid()

    def matching_errors(target):
        bce = functional.binary_cross_entropy_with_logits(logits, target, reduction="none").mean(dim=(1, 2, 3))
        intersection = (probabilities * target).sum(dim=(2, 3))
        union = probabilities.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = 1.0 - ((2.0 * intersection + 1e-6) / (union + 1e-6)).mean(dim=1)
        return bce + dice

    return torch.minimum(matching_errors(reference), matching_errors(reference[:, [1, 0]]))


def loss_weights(epoch_index):
    if epoch_index < 2:
        return (1.0, 0.05, 0.0)
    return (0.75, 0.2, 0.4)


def measurement_loss(logits, heatmaps, coordinates, epoch_index):
    predicted = decode_endpoints(logits)
    weights = sample_weights(coordinates)
    heatmap_weight, coordinate_weight, length_weight = loss_weights(epoch_index)
    heatmap_term = (heatmap_errors(logits, heatmaps) * weights).mean()
    coordinate_term = (coordinate_errors(predicted, coordinates) * weights).mean()
    length_term = ((segment_lengths(predicted) - segment_lengths(coordinates)).abs() * weights).mean()
    return heatmap_weight * heatmap_term + coordinate_weight * coordinate_term + length_weight * length_term


def summarize_task(records, tolerance):
    reference = records.gt_length_cm.to_numpy(dtype=float)
    predicted = records.pred_length_cm.to_numpy(dtype=float)
    differences = predicted - reference
    errors = np.abs(differences)
    count = len(reference)
    mae = float(errors.mean()) if count else float("nan")
    half_width = 1.96 * float(errors.std(ddof=1)) / float(np.sqrt(count)) if count >= 2 else 0.0
    interval = (f"{max(0.0, mae - half_width):.3f}-{mae + half_width:.3f}" if count >= 2
                else f"{mae:.3f}-{mae:.3f}")
    mean = float(reference.mean()) if count else float("nan")
    spread = float(reference.std(ddof=1)) if count >= 2 else float("nan")
    pearson = (float(np.corrcoef(reference, predicted)[0, 1])
               if count >= 2 and reference.std() >= 1e-8 and predicted.std() >= 1e-8 else float("nan"))
    denominator = (float(reference.var(ddof=1)) + float(predicted.var(ddof=1))
                   + (mean - float(predicted.mean())) ** 2) if count >= 2 else 0.0
    ccc = float(2.0 * np.cov(reference, predicted, ddof=1)[0, 1] / denominator) if denominator else float("nan")
    bias = float(differences.mean()) if count else float("nan")
    deviation = float(differences.std(ddof=1)) if count >= 2 else float("nan")
    valid = np.abs(reference) > 1e-12
    values = {
        "num_samples": count, "mae": mae, "mae_ci95": interval,
        "mae_std": float(errors.std(ddof=0)) if count else float("nan"),
        "mape": float(np.mean(errors[valid] / np.abs(reference[valid])) * 100.0) if valid.any() else float("nan"),
        "rmse": float(np.sqrt(np.mean(differences ** 2))) if count else float("nan"),
        "pearson_r": pearson,
        "gt_length_mean_std": f"{mean:.3f}\u00b1{spread:.3f}" if np.isfinite(mean) and np.isfinite(spread) else "",
        "acceptable_margin_cm": tolerance,
        "cp": float(np.mean(errors <= tolerance + 1e-12)) if count else float("nan"),
        "ccc": ccc, "ba_bias_cm": bias, "ba_sd_cm": deviation,
        "ba_loa_lower_cm": bias - 1.96 * deviation, "ba_loa_upper_cm": bias + 1.96 * deviation,
    }
    return {name: round(float(value), 3) if isinstance(value, (float, np.floating)) and np.isfinite(value)
            else value for name, value in values.items()}


@torch.no_grad()
def evaluate_by_task(model, loader, table, device, output, subset):
    model.eval()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    visual_counts = {}
    for batch in tqdm(loader, desc=f"EvalByTask[{subset}]", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        task_ids = batch["task_id"].to(device, non_blocking=True)
        predicted = decode_endpoints(model(images, task_ids)).cpu().numpy()
        reference = batch["coords"].cpu().numpy()
        scales = batch["scale_ratio"].cpu().numpy().astype(np.float64)
        predicted_points, reference_points = predicted.reshape(-1, 2, 2), reference.reshape(-1, 2, 2)
        predicted_lengths = np.linalg.norm(predicted_points[:, 0] - predicted_points[:, 1], axis=1) * scales
        reference_lengths = np.linalg.norm(reference_points[:, 0] - reference_points[:, 1], axis=1) * scales
        for position, data_index in enumerate(batch["data_index"].tolist()):
            metadata = table.iloc[data_index]
            task = ("IVS", "LVPW", "LVID", "LA", "AORTIC_ROOT", "AORTA")[int(metadata.task_id)]
            reference_length, predicted_length = float(reference_lengths[position]), float(predicted_lengths[position])
            error = abs(predicted_length - reference_length)
            rows.append({"data_index": data_index, "csv_index": data_index,
                         "frame_array_index": int(metadata.frame_array_index),
                         "patient": str(metadata.HashedFileName), "task_name": task,
                         "label_name": metadata.label_name, "gt_length_cm": reference_length,
                         "pred_length_cm": predicted_length, "abs_error_cm": error})
            count = visual_counts.get(task, 0)
            if count < 20:
                destination = output / task
                destination.mkdir(parents=True, exist_ok=True)
                patient = str(metadata.HashedFileName).strip().replace("/", "_").replace("\\", "_").replace(" ", "_")[:48] or "unknown"
                path = destination / f"sample_{count:03d}_idx_{data_index}_patient_{patient}.png"
                image = (images[position].cpu().permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
                save_endpoint_visualization(path, image, reference[position].copy(), predicted[position].copy(), [
                    f"idx={data_index} patient={metadata.HashedFileName} task={task}",
                    f"Label={metadata.label_name} GT_len={reference_length:.3f}cm Pred_len={predicted_length:.3f}cm",
                    f"AbsErr={error:.3f}cm",
                ])
                visual_counts[task] = count + 1
    records = pd.DataFrame(rows, columns=["data_index", "csv_index", "frame_array_index", "patient", "task_name",
                                        "label_name", "gt_length_cm", "pred_length_cm", "abs_error_cm"])
    report = {}
    for task, tolerance in zip(("IVS", "LVPW", "LVID", "LA", "AORTIC_ROOT", "AORTA"), (.20, .20, .41, .67, .27, .25)):
        details = records.loc[records.task_name.eq(task)]
        report[task] = summarize_task(details, tolerance)
        if len(details):
            destination = output / task
            destination.mkdir(parents=True, exist_ok=True)
            details.round(3).to_csv(destination / f"{subset}_length_comparison.csv", index=False, encoding="utf-8-sig")
    return report


def write_run_reports(model, loaders, table, partitions, device, run, training, started=None):
    run.mkdir(parents=True, exist_ok=True)
    reports = {subset: evaluate_by_task(model, loaders[subset], table, device, run / f"{subset}_outputs", subset)
               for subset in ("train", "val", "test")}
    if started is not None:
        training["training_time_sec"] = round(float(time.perf_counter() - started), 3)
    summary = {
        "run_dir": str(run), "metric_unit": "cm",
        "subset": {"strategy": "subset_random_by_patient", "seed": 42,
                   "patient_counts": {name: int(table.iloc[indices].HashedFileName.nunique()) for name, indices in partitions.items()},
                   "sample_counts": {name: len(indices) for name, indices in partitions.items()}},
        "dataset": {"num_samples": len(table)}, "training": training,
        "task_mapping": {
            "task_to_id": {"IVS": 0, "LVPW": 1, "LVID": 2, "LA": 3, "AORTIC_ROOT": 4, "AORTA": 5},
            "calc_to_task": {"IVSD": "IVS", "IVSS": "IVS", "LVPWD": "LVPW", "LVPWS": "LVPW",
                             "LVIDD": "LVID", "LVIDS": "LVID", "LA": "LA", "AORTIC_ROOT": "AORTIC_ROOT", "AORTA": "AORTA"}},
        "train_report": reports["train"], "validation_report": reports["val"], "test_report": reports["test"],
    }
    (run / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    save_report_visualizations(run, reports["val"], reports["test"])
    rows = [{"task": task, "dataset": subset, **metrics}
            for subset, report in reports.items() for task, metrics in report.items()]
    pd.DataFrame(rows).to_csv(run / "summary.csv", index=False, encoding="utf-8-sig", float_format="%.3f")
    return summary


def train_epoch(model, loader, optimizer, device, epoch_index):
    model.train()
    total = 0.0
    for batch in tqdm(loader, desc="Training", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        heatmaps = batch["mask"].to(device, non_blocking=True)
        coordinates = batch["coords"].to(device, non_blocking=True)
        task_ids = batch["task_id"].to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(images, task_ids)
            loss = measurement_loss(logits, heatmaps, coordinates, epoch_index)
        loss.backward()
        optimizer.step()
        total += float(loss.item())
    return total / len(loader)


@torch.no_grad()
def validate_epoch(model, loader, device, epoch_index):
    model.eval()
    loss_total = 0.0
    batch_errors = []
    for batch in tqdm(loader, desc="Validation", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        heatmaps = batch["mask"].to(device, non_blocking=True)
        coordinates = batch["coords"].to(device, non_blocking=True)
        task_ids = batch["task_id"].to(device, non_blocking=True)
        scales = batch["scale_ratio"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(images, task_ids)
            predicted = decode_endpoints(logits)
            loss = measurement_loss(logits, heatmaps, coordinates, epoch_index)
        error = (segment_lengths(predicted) - segment_lengths(coordinates)).abs() * scales
        batch_errors.append(float(error.mean().item()))
        loss_total += float(loss.item())
    return loss_total / len(loader), float(np.mean(batch_errors))


def main():
    if len(sys.argv) != 1:
        raise SystemExit("Command-line options are not supported; use path environment variables.")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Training requires a CUDA device supporting bfloat16 autocast")
    device = torch.device("cuda:0")
    table, partitions = load_table()
    training, validation = make_training_loaders(table, partitions)
    model = HRNetW18Lite()
    if not PRETRAINED_PATH.is_file():
        raise FileNotFoundError(PRETRAINED_PATH)
    load_checkpoint(model.backbone, str(PRETRAINED_PATH), strict=False)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    run = OUTPUT_ROOT / f"runs_lvh_fusion_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run.mkdir(parents=True, exist_ok=False)
    best_mae = float("inf")
    best_loss = float("inf")
    stale_epochs = 0
    best_epoch = 0
    history = []
    started = time.perf_counter()
    for epoch_index in range(30):
        training_loss = train_epoch(model, training, optimizer, device, epoch_index)
        validation_loss, validation_mae = validate_epoch(model, validation, device, epoch_index)
        scheduler.step()
        history.append({"epoch": epoch_index + 1, "phase": "stage1" if epoch_index < 2 else "stage2",
                        "loss_weights": dict(zip(("seg", "coord", "length"), loss_weights(epoch_index))),
                        "train_loss": training_loss, "val_loss": validation_loss, "val_mae": validation_mae})
        print(f"Epoch {epoch_index + 1}: train={training_loss:.6f}, val={validation_loss:.6f}, MAE={validation_mae:.8f} cm")
        if not np.isfinite([training_loss, validation_loss, validation_mae]).all():
            raise FloatingPointError("Nonfinite training or validation result")
        if epoch_index >= 2:
            if validation_mae < best_mae:
                best_mae = validation_mae
                best_loss = validation_loss
                best_epoch = epoch_index + 1
                stale_epochs = 0
                torch.save(model.state_dict(), run / "fusion_weights_best.ckpt")
            else:
                stale_epochs += 1
        if stale_epochs >= 5:
            break
    checkpoint = run / "fusion_weights_best.ckpt"
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
    report = {"model_name": "HrnetW18MultiTaskModel", "epochs": 30, "batch_size": 32,
              "lr": 1e-4, "weight_decay": 1e-4, "num_workers": 8,
              "generate_masks_on_the_fly": True, "copy_frames_subset_to_ram": False,
              "precision_requested": "bf16", "precision_active": "bf16", "endpoint_activation": "softmax",
              "early_stopping_patience": 5, "best_epoch": best_epoch,
              "best_val_mae": round(best_mae, 3), "best_val_loss": round(best_loss, 3),
              "training_time_sec": 0.0, "history": history}
    write_run_reports(model, {"train": training, "val": validation,
                             "test": make_evaluation_loader(table, partitions["test"])},
                      table, partitions, device, run, report, started)
    print(f"Checkpoint: {checkpoint}\nSummary json: {run / 'summary.json'}\nSummary csv: {run / 'summary.csv'}")


if __name__ == "__main__":
    main()