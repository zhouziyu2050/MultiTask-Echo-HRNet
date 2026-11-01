import json
import math
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("ECHONET_DATA_ROOT", str(ROOT / "data"))).resolve()
RAW_ROOT = DATA_ROOT / "raw" / "EchoNet-LVH"
PROCESSED_ROOT = DATA_ROOT / "processed"
ANNOTATIONS_PATH = ROOT / "annotations" / "measurements.csv"


def extract_frames(source, indexed_videos, destination):
    unique = source[["HashedFileName", "Frame"]].drop_duplicates().reset_index(drop=True)
    unique["temporary_index"] = np.arange(len(unique))
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite extracted frames: {destination}")
    temporary = destination.with_name("frames_pending.npy")
    if temporary.exists():
        raise FileExistsError(f"Remove incomplete extraction before retrying: {temporary}")
    frames = open_memmap(temporary, mode="w+", dtype=np.uint8, shape=(len(unique), 480, 640, 3))
    present = np.zeros(len(unique), dtype=bool)
    for patient, requests in tqdm(unique.groupby("HashedFileName", sort=False), desc="Extract annotated frames"):
        if patient not in indexed_videos:
            continue
        capture = cv2.VideoCapture(str(indexed_videos[patient]), cv2.CAP_FFMPEG)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(str(indexed_videos[patient]))
        try:
            if not capture.isOpened():
                continue
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            for row in requests.itertuples():
                if total <= 0:
                    continue
                capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(int(row.Frame), total - 1)))
                success, frame = capture.read()
                if not success or frame is None:
                    continue
                frames[row.temporary_index] = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
                present[row.temporary_index] = True
        finally:
            capture.release()
    frames.flush()
    successful = np.flatnonzero(present)
    if not len(successful):
        del frames
        temporary.unlink()
        raise RuntimeError("No frames were extracted. Check the annotation table and videos.")
    compact = open_memmap(destination, mode="w+", dtype=np.uint8, shape=(len(successful), 480, 640, 3))
    for position, old_index in enumerate(successful):
        compact[position] = frames[old_index]
    compact.flush()
    del compact, frames
    temporary.unlink()
    saved = unique.iloc[successful]
    return {(row.HashedFileName, int(row.Frame)): position for position, row in enumerate(saved.itertuples())}


def filter_annotations(source):
    if source.columns.tolist() != ["HashedFileName", "Calc", "CalcValue", "Frame", "X1", "X2", "Y1", "Y2",
                                    "Frames", "FPS", "Width", "Height"]:
        raise ValueError("Expected only the 12 original annotation columns in the documented order")
    frames = pd.to_numeric(source.Frames, errors="coerce")
    numbers = pd.to_numeric(source.Frame, errors="coerce")
    table = source.loc[np.isfinite(frames) & np.isfinite(numbers)].copy()
    table["HashedFileName"] = table.HashedFileName.astype(str).str.strip().str.upper()
    table = table.loc[table.HashedFileName.ne("")].copy()
    table["Frame"] = pd.to_numeric(table.Frame).astype(np.int64)
    return table.reset_index(drop=True)


def prepare_annotations(source, frame_lookup):
    keys = [(patient, int(frame)) for patient, frame in zip(source.HashedFileName, source.Frame)]
    table = source.loc[[key in frame_lookup for key in keys]].copy()
    table["frame_array_index"] = [
        frame_lookup[(patient, int(frame))] for patient, frame in zip(table.HashedFileName, table.Frame)
    ]
    derived = []
    for row in table.to_dict("records"):
        try:
            width, height = float(row["Width"]), float(row["Height"])
            first_x = float(row["X1"]) * (640 / width) if width > 0 else 0.0
            second_x = float(row["X2"]) * (640 / width) if width > 0 else 0.0
            first_y = float(row["Y1"]) * (480 / height) if height > 0 else 0.0
            second_y = float(row["Y2"]) * (480 / height) if height > 0 else 0.0
            distance = math.sqrt((second_x - first_x) ** 2 + (second_y - first_y) ** 2)
            value = float(row["CalcValue"] or "0")
            derived.append((round(first_x, 2), round(second_x, 2), round(first_y, 2), round(second_y, 2),
                            round(distance, 4), round(value / distance, 6) if distance > 0 else 0.0))
        except (ValueError, KeyError):
            derived.append((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    table[["X1_scaled", "X2_scaled", "Y1_scaled", "Y2_scaled", "pixel_dist", "scale_ratio"]] = pd.DataFrame(
        derived, index=table.index, columns=["X1_scaled", "X2_scaled", "Y1_scaled", "Y2_scaled", "pixel_dist", "scale_ratio"]
    )
    return table


def main():
    if len(sys.argv) != 1:
        raise SystemExit("Command-line options are not supported; use path environment variables.")
    source = pd.read_csv(ANNOTATIONS_PATH, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    filtered = filter_annotations(source)
    indexed_videos = {}
    for video in RAW_ROOT.rglob("*.avi"):
        name = video.stem.upper()
        if name in indexed_videos:
            raise ValueError(f"Duplicate video name: {name}")
        indexed_videos[name] = video
    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    frame_lookup = extract_frames(filtered, indexed_videos, PROCESSED_ROOT / "frames.npy")
    table = prepare_annotations(filtered, frame_lookup)
    table.to_csv(PROCESSED_ROOT / "measurements.csv", index=False, encoding="utf-8-sig")
    audit = {"source_records": len(source), "filtered_records": len(filtered), "records": len(table),
             "patients": table.HashedFileName.nunique(), "dropped_after_frame_read": len(filtered) - len(table),
             "unique_frames": len(frame_lookup), "channel_order": "BGR",
             "frame_array_index": "assigned after video processing, in saved frame-array order",
             "frame_selection": "original manually annotated Frame indices; zero-based",
             "resize": "OpenCV INTER_AREA; 640x480"}
    (PROCESSED_ROOT / "preprocessing_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()