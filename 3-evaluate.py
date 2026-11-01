import json
import importlib
import os
import sys
from pathlib import Path

import torch

train = importlib.import_module("2-train")
EVALUATION_PATH = Path(os.environ.get(
    "HRNET_CHECKPOINT", str(train.ROOT / "weights" / "trained" / "hrnet_w18_lite.ckpt")
)).resolve()


def main():
    if len(sys.argv) != 1:
        raise SystemExit("Command-line options are not supported; use path environment variables.")
    if not torch.cuda.is_available():
        raise RuntimeError("Evaluation requires CUDA")
    device = torch.device("cuda:0")
    table, partitions = train.load_table()
    model = train.HRNetW18Lite()
    model.load_state_dict(torch.load(EVALUATION_PATH, map_location="cpu", weights_only=True), strict=True)
    model.to(device)
    training, validation = train.make_training_loaders(table, partitions)
    previous = EVALUATION_PATH.parent / "summary.json"
    training_report = json.loads(previous.read_text(encoding="utf-8"))["training"] if previous.is_file() else {}
    output = train.OUTPUT_ROOT / "evaluation"
    train.write_run_reports(model, {"train": training, "val": validation,
                                   "test": train.make_evaluation_loader(table, partitions["test"])},
                            table, partitions, device, output, training_report)
    print(f"Summary json: {output / 'summary.json'}\nSummary csv: {output / 'summary.csv'}")


if __name__ == "__main__":
    main()