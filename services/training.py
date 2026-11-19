from pathlib import Path
import torch
from ultralytics import YOLO
import yaml


def resolve_device() -> str | int:
    return 0 if torch.cuda.is_available() else "cpu"


def build_data_yaml(
    dataset_path: str | Path,
    output_yaml: str | Path = "/kaggle/working/data.yaml",
    names: dict[int, str] | None = None,
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


def train_yolo_pipeline(
    dataset_path: (
        str | Path
    ) = "/kaggle/input/datasets/withus001/preprocessed-lisa-traffic-light-dataset/dataset",
    weights: str = "yolo26s.pt",
    yaml_path: str | Path = "/kaggle/working/data.yaml",
    epochs: int = 20,
    batch: int = 16,
    imgsz: int = 640,
    workers: int = 4,
    name: str = "signalvision",
    names: dict[int, str] | None = None,
):
    device = resolve_device()
    effective_workers = workers if device != "cpu" else 0

    data_yaml = build_data_yaml(
        dataset_path=dataset_path,
        output_yaml=yaml_path,
        names=names,
    )

    model = YOLO(weights)
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        workers=effective_workers,
        name=name,
    )


if __name__ == "__main__":
    train_yolo_pipeline()
