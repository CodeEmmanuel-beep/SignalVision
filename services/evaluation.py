from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict, Iterable, Optional
import pandas as pd
import torch
from ultralytics import YOLO
import yaml


def log_hardware_status() -> tuple[bool, Optional[str]]:
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else None
    print(f"CUDA available: {cuda_available} | Device: {device_name or 'CPU'}")
    return cuda_available, device_name


def build_data_yaml(
    dataset_path: str | Path,
    output_yaml: str | Path = "/kaggle/working/data.yaml",
    names: Optional[Dict[int, str]] = None,
    train_rel: str = "train/images",
    val_rel: str = "val/images",
) -> Path:
    if names is None:
        names = {0: "stop", 1: "warning", 2: "go"}

    payload = {
        "path": str(dataset_path),
        "train": train_rel,
        "val": val_rel,
        "names": names,
    }

    output_path = Path(output_yaml)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)

    return output_path


def run_confidence_sweep(
    weights_path: str | Path = (
        "/kaggle/input/models/withus001/traffic-light-detection-model/"
        "pytorch/default/1/runs/detect/train/weights/best.pt"
    ),
    data_yaml: str | Path = "/kaggle/working/data.yaml",
    dataset_path: (
        str | Path
    ) = "/kaggle/input/datasets/withus001/preprocessed-lisa-traffic-light-dataset/dataset",
    names: Optional[Dict[int, str]] = None,
    conf_thresholds: Iterable[float] = (0.25, 0.35, 0.45, 0.55, 0.70),
    iou: float = 0.45,
    plots: bool = True,
    save_json: bool = True,
    output_dir: str | Path = "/kaggle/working/val_sweep",
) -> pd.DataFrame:
    log_hardware_status()

    weights_file = Path(weights_path)
    if not weights_file.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_file}")

    yaml_file = Path(data_yaml)
    if not yaml_file.exists():
        yaml_file = build_data_yaml(
            dataset_path=dataset_path,
            output_yaml=yaml_file,
            names=names,
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights_file))
    records = []

    for conf in conf_thresholds:
        run_name = f"conf_{str(conf).replace('.', '_')}"
        metrics = model.val(
            data=str(yaml_file),
            conf=conf,
            iou=iou,
            plots=plots,
            save_json=save_json,
            name=run_name,
        )

        row = {
            "conf": conf,
            "iou": iou,
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
        }
        records.append(row)

        print(
            f"Confidence: {conf:.2f} | "
            f"Precision: {row['precision']:.4f} | "
            f"Recall: {row['recall']:.4f} | "
            f"mAP50: {row['map50']:.4f} | "
            f"mAP50-95: {row['map50_95']:.4f}"
        )

    df = pd.DataFrame(records)
    csv_path = out_dir / "sweep_results.csv"
    json_path = out_dir / "sweep_results.json"

    df.to_csv(csv_path, index=False)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "weights": str(weights_file),
                "data": str(yaml_file),
                "results": records,
            },
            f,
            indent=2,
        )

    print(f"Saved sweep summary to {csv_path} and {json_path}")


if __name__ == "__main__":
    results_df = run_confidence_sweep()
