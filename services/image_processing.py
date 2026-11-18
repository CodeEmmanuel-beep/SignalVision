from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import cv2
import torch
import numpy as np
import shutil
import os
from sklearn.model_selection import train_test_split


class processLisaTrafficLightDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.root = root_dir
        annotations = list(Path(self.root).rglob("frameAnnotationsBOX.csv"))
        print("number of annotation files", len(annotations))
        Frame = []
        self.transform = transform
        for annotation in annotations:
            annot = pd.read_csv(annotation, sep=";")
            clip = annotation.parent.name
            annot["clip"] = str(clip)
            Frame.append(annot)
        annotate = pd.concat(Frame, ignore_index=True)
        self.annotate = annotate
        self.len = len(annotate)
        print("Total annotation rows", self.len)
        annotate["class_id"] = annotate["Annotation tag"].map(self.label_to_id)
        self.images_lookup = {}
        for path in Path(self.root).rglob("*"):
            if path.is_file() and path.suffix.lower() in (".jpeg", ".jpg", ".png"):
                self.images_lookup[(path.parent.parent.name, path.name)] = str(path)
        print("number of files in this dataset", len(self.images_lookup))

    def __label__(self, label_map):
        result = self.annotate["class_id"] = self.annotate["Annotation tag"].map(
            label_map
        )
        return result

    def __extract_head__(self):
        return self.annotate.head(20)

    def __label_dist__(self):
        return self.annotate["Annotation tag"].value_counts(dropna=False)

    def __label_to_clip_dist__(self):
        return (
            self.annotate.groupby(["clip", "Annotation tag"])
            .size()
            .unstack(fill_value=0)
        )

    def __file_to_label_dist__(self):
        return (
            self.annotate.groupby("Filename")["Annotation tag"]
            .nunique()
            .value_counts()
            .sort_index()
        )

    def __label_to_file_dist__(self):
        return self.annotate.groupby("Annotation tag")["Filename"].nunique()

    def __plot_images__(self, idx):
        row = self.annotate.iloc[idx]
        if row is None:
            raise FileNotFoundError("index out of range")
        filename = row["Filename"].replace("\\", "/").split("/")[-1]
        clip_name = Path(row["clip"]).name
        image_path = self.images_lookup.get((clip_name, filename))
        if image_path is None:
            raise FileNotFoundError(
                f"Image '{filename}' was not found in root_dir '{self.root}'"
            )
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image)
        return plt.show()

    def __getitem__(self, idx):
        row = self.annotate.iloc[idx]
        if row is None:
            raise FileNotFoundError("index out of range")
        filename = row["Filename"].replace("\\", "/").split("/")[-1]
        clip_name = Path(row["clip"]).name
        image_path = self.images_lookup.get((clip_name, filename))
        if image_path is None:
            raise FileNotFoundError(
                f"Image and clip '{filename, clip_name}' was not found in root_dir '{self.root}'"
            )
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        box = [
            float(row["Upper left corner X"]),
            float(row["Upper left corner Y"]),
            float(row["Lower right corner X"]),
            float(row["Lower right corner Y"]),
        ]

        target = {
            "boxes": torch.tensor([box], dtype=torch.float32),
            "labels": torch.tensor(
                [self.label_to_id(row["Annotation tag"])], dtype=torch.int64
            ),
        }

        if self.transform:
            image = self.transform(image)
        return image, target

    def __len__(self):
        return self.len

    def label_to_id(self, label_str):
        mapping = {
            "go": 2,
            "stop": 0,
            "warning": 1,
            "goForward": 2,
            "goLeft": 2,
            "stopLeft": 0,
            "warningLeft": 1,
        }
        if label_str not in mapping:
            raise ValueError(f"Unknown annotation tag: {label_str}")
        return mapping.get(label_str)

    def __reveal_data__(self, idx, text_th=2, rect_th=2):
        image, target = self.__getitem__(idx)
        inp = image.detach().cpu().clone()
        if inp.min() < 0:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            inp = inp * std + mean
        inp = inp.permute(1, 2, 0)
        inp = inp.clip(0, 1)
        inp = (inp * 255).byte().cpu().numpy()
        inp = np.ascontiguousarray(inp)

        boxes = target["boxes"]
        labels = target["labels"]

        for i, box in enumerate(boxes):
            pt1 = (int(box[0]), int(box[1]))
            pt2 = (int(box[2]), int(box[3]))

            l_id = labels[i].item() if labels.ndim > 0 else labels.item()
            label = None
            cv2.rectangle(inp, pt1, pt2, (0, 255, 0), rect_th)
            label_map = {2: "go", 1: "warning", 0: "stop"}
            label = label_map.get(l_id)
            cv2.putText(
                inp,
                f"{label}",
                pt1,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                text_th,
            )
            if label is None:
                label = "None"
            plt.title(label)
        plt.imshow(inp)
        plt.axis("off")
        plt.show()

    def __extract_labels__(self):
        output_text_dir = "labels"
        if os.path.exists(output_text_dir):
            shutil.rmtree(output_text_dir)
        os.makedirs(output_text_dir, exist_ok=True)
        dim_cache = {}
        missing = 0
        invalid = 0
        written = 0
        for idx, split in self.annotate.iterrows():
            clip = Path(split["clip"]).name
            file_name = Path(split["Filename"]).name
            image_path = self.images_lookup.get((clip, file_name))
            if image_path is None:
                missing += 1
                continue
            if image_path in dim_cache:
                img_height, img_width = dim_cache[image_path]
            else:
                image = cv2.imread(image_path)
                if image is None:
                    missing += 1
                    continue
                img_height, img_width, _ = image.shape
                dim_cache[image_path] = (img_height, img_width)
            x_min = split["Upper left corner X"]
            y_min = split["Upper left corner Y"]
            x_max = split["Lower right corner X"]
            y_max = split["Lower right corner Y"]
            class_id = split["class_id"]

            if x_max <= x_min or y_max <= y_min:
                invalid += 1
                continue

            if not (
                0 <= x_min < img_width
                and 0 < x_max <= img_width
                and 0 <= y_min < img_height
                and 0 < y_max <= img_height
            ):
                invalid += 1
                continue

            x_center = ((x_min + x_max) / 2.00) / img_width
            y_center = ((y_min + y_max) / 2.00) / img_height
            width = (x_max - x_min) / img_width
            height = (y_max - y_min) / img_height

            parent_name = Path(image_path).parent.parent.name
            base_name = Path(image_path).stem
            text_file = os.path.join(output_text_dir, f"{parent_name}_{base_name}.txt")

            with open(text_file, "a") as f:
                f.write(
                    f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} \n"
                )
                written += 1

        print("\n Missing: ", missing)
        print("\n Invalid: ", invalid)
        print("\n Written: ", written)

    def __extract_images__(self):
        if os.path.exists("images"):
            shutil.rmtree("images")
        image_dir = Path("images")
        image_dir.mkdir(parents=True, exist_ok=True)

        for img_path_str in self.images_lookup.values():
            img_path = Path(img_path_str)
            dest_file = image_dir / f"{img_path.parent.parent.name}_{img_path.name}"
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, dest_file)

        print(f"{len(self.images_lookup)} images copied successfully")

    def __split_files__(self):
        image_f = list(Path("images").rglob("*"))
        self.label_f = Path("labels")
        self.images = Path("images")
        self.output_dir = Path("dataset")

        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)

        for split in ["train", "val"]:
            (self.output_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

        train_dataset, val_dataset = train_test_split(
            image_f, test_size=0.2, random_state=45
        )
        print(f"Total Images: {len(image_f)}")
        print(f"Training: {len(train_dataset)} | Validation: {len(val_dataset)}")
        return train_dataset, val_dataset

    def __copy_split_files__(self):
        train_dataset, val_dataset = self.__split_files__()
        splits = [(train_dataset, "train"), (val_dataset, "val")]

        for file_list, split_name in splits:
            copied_count = 0
            missing_labels = 0

            for img_path in file_list:
                dest_path = self.output_dir / split_name / "images" / f"{img_path.name}"
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img_path, dest_path)

                label_path = (
                    self.output_dir / split_name / "labels" / f"{img_path.stem}.txt"
                )
                label_path.parent.mkdir(parents=True, exist_ok=True)
                label = self.label_f / f"{img_path.stem}.txt"

                if label.exists():
                    shutil.copy2(label, label_path)
                    copied_count += 1
                else:
                    missing_labels += 1

            print(
                f"[{split_name.upper()}] Copied {len(file_list)} images, "
                f"{copied_count} label files, and "
                f"{missing_labels} missing labels."
            )
