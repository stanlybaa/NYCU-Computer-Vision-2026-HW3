import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import tifffile
from PIL import Image
from pycocotools import mask as mask_util


TRAIN_DIR = "train"
TEST_DIR = "test_release"
TEST_MAPPING = "test_image_name_to_ids.json"

OUT_DIR = Path("coco_cell")
TRAIN_RATIO = 0.85
SEED = 42

MIN_MASK_AREA = 6


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)


def normalize_image(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)

    p1, p99 = np.percentile(img, (1, 99))
    img = np.clip(img, p1, p99)

    span = p99 - p1
    if span > 1e-6:
        img = (img - p1) / span
    else:
        img = img / (img.max() + 1e-6)

    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


def to_3ch(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)

    if img.ndim == 3 and img.shape[0] in (1, 2, 3, 4):
        img = np.transpose(img, (1, 2, 0))

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3 and img.shape[-1] == 1:
        img = np.concatenate([img, img, img], axis=-1)
    elif img.ndim == 3 and img.shape[-1] == 2:
        img = np.concatenate([img, img[:, :, :1]], axis=-1)
    elif img.ndim == 3 and img.shape[-1] >= 4:
        img = img[:, :, :3]

    return img


def save_png(img: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path)


def encode_rle(binary_mask):
    binary_mask = np.asfortranarray(binary_mask.astype(np.uint8))
    rle = mask_util.encode(binary_mask)
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def mask_to_bbox(binary_mask):
    pos = np.where(binary_mask > 0)

    if len(pos[0]) == 0:
        return None

    xmin = int(np.min(pos[1]))
    xmax = int(np.max(pos[1])) + 1
    ymin = int(np.min(pos[0]))
    ymax = int(np.max(pos[0])) + 1

    width = xmax - xmin
    height = ymax - ymin

    if width <= 0 or height <= 0:
        return None

    return [float(xmin), float(ymin), float(width), float(height)]


def convert_one_folder(img_dir: Path, split: str, image_id: int, ann_start_id: int):
    image_path = img_dir / "image.tif"

    img = tifffile.imread(str(image_path))
    img = to_3ch(img)
    img = normalize_image(img)

    height, width = img.shape[:2]

    file_name = f"{img_dir.name}.png"
    out_img_path = OUT_DIR / "images" / split / file_name
    save_png(img, out_img_path)

    image_info = {
        "id": int(image_id),
        "file_name": file_name,
        "height": int(height),
        "width": int(width),
    }

    annotations = []
    ann_id = ann_start_id

    for class_id in range(1, 5):
        mask_path = img_dir / f"class{class_id}.tif"

        if not mask_path.exists():
            continue

        mask_data = tifffile.imread(str(mask_path))
        mask_data = np.asarray(mask_data)

        if mask_data.ndim > 2:
            mask_data = np.squeeze(mask_data)

        instance_ids = np.unique(mask_data)
        instance_ids = instance_ids[instance_ids != 0]

        for instance_id in instance_ids:
            binary_mask = (mask_data == instance_id).astype(np.uint8)
            area = int(binary_mask.sum())

            if area < MIN_MASK_AREA:
                continue

            bbox = mask_to_bbox(binary_mask)

            if bbox is None:
                continue

            rle = encode_rle(binary_mask)

            ann = {
                "id": int(ann_id),
                "image_id": int(image_id),
                "category_id": int(class_id),
                "bbox": bbox,
                "area": float(area),
                "iscrowd": 0,
                "segmentation": rle,
            }

            annotations.append(ann)
            ann_id += 1

    return image_info, annotations, ann_id


def convert_split(image_dirs, split):
    images = []
    annotations = []

    ann_id = 1

    for image_id, img_dir in enumerate(image_dirs, start=1):
        image_info, anns, ann_id = convert_one_folder(
            img_dir=img_dir,
            split=split,
            image_id=image_id,
            ann_start_id=ann_id,
        )

        images.append(image_info)
        annotations.extend(anns)

    categories = [
        {"id": 1, "name": "class1"},
        {"id": 2, "name": "class2"},
        {"id": 3, "name": "class3"},
        {"id": 4, "name": "class4"},
    ]

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    ann_path = OUT_DIR / "annotations" / f"{split}.json"
    ann_path.parent.mkdir(parents=True, exist_ok=True)

    with open(ann_path, "w") as f:
        json.dump(coco, f)

    print(f"{split}: images={len(images)}, annotations={len(annotations)}")


def convert_test_images():
    with open(TEST_MAPPING, "r") as f:
        mapping = json.load(f)

    out_dir = OUT_DIR / "images" / "test"
    out_dir.mkdir(parents=True, exist_ok=True)

    converted = 0

    for item in mapping:
        file_name = item["file_name"]
        src_path = Path(TEST_DIR) / file_name

        if not src_path.exists():
            print(f"[WARN] missing test image: {src_path}")
            continue

        img = tifffile.imread(str(src_path))
        img = to_3ch(img)
        img = normalize_image(img)

        out_name = Path(file_name).with_suffix(".png").name
        out_path = out_dir / out_name
        save_png(img, out_path)

        converted += 1

    print(f"Converted test images: {converted}")


def main():
    seed_everything(SEED)

    for subdir in [
        OUT_DIR / "images" / "train",
        OUT_DIR / "images" / "val",
        OUT_DIR / "images" / "test",
        OUT_DIR / "annotations",
    ]:
        subdir.mkdir(parents=True, exist_ok=True)

    image_dirs = sorted(
        [
            Path(TRAIN_DIR) / d
            for d in os.listdir(TRAIN_DIR)
            if (Path(TRAIN_DIR) / d).is_dir()
        ]
    )

    random.shuffle(image_dirs)

    train_size = int(TRAIN_RATIO * len(image_dirs))
    train_dirs = image_dirs[:train_size]
    val_dirs = image_dirs[train_size:]

    print(f"Train folders: {len(train_dirs)}")
    print(f"Val folders: {len(val_dirs)}")

    convert_split(train_dirs, "train")
    convert_split(val_dirs, "val")
    convert_test_images()

    print("Done. COCO dataset is at coco_cell/")


if __name__ == "__main__":
    main()
