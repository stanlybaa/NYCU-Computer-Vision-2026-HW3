import glob
import json
import os
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
from pycocotools import mask as mask_util
from tqdm import tqdm


CONFIG_PATH = "cascade_mask_rcnn_cell.py"
WORK_DIR = "work_dirs/cascade_cell"

TEST_IMAGE_DIR = "coco_cell/images/test"
TEST_MAPPING = "test_image_name_to_ids.json"

OUTPUT_JSON = "test-results.json"
OUTPUT_ZIP = "submission.zip"

SCORE_THRESHOLD = 0.02
MAX_PER_IMAGE = 300
MASK_THRESHOLD = 0.5


def find_checkpoint():
    best_ckpts = sorted(glob.glob(os.path.join(WORK_DIR, "best_*.pth")))

    if len(best_ckpts) > 0:
        print(f"Using best checkpoint: {best_ckpts[-1]}")
        return best_ckpts[-1]

    latest = os.path.join(WORK_DIR, "latest.pth")

    if os.path.exists(latest):
        print(f"Using latest checkpoint: {latest}")
        return latest

    ckpts = sorted(glob.glob(os.path.join(WORK_DIR, "*.pth")))

    if len(ckpts) == 0:
        raise FileNotFoundError(f"No checkpoint found in {WORK_DIR}")

    print(f"Using checkpoint: {ckpts[-1]}")
    return ckpts[-1]


def encode_binary_mask(binary_mask):
    binary_mask = np.asfortranarray(binary_mask.astype(np.uint8))
    rle = mask_util.encode(binary_mask)
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def get_instances(result):
    if hasattr(result, "pred_instances"):
        return result.pred_instances

    raise RuntimeError("Unexpected MMDetection result format.")


def tensor_to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def main():
    register_all_modules()

    checkpoint = find_checkpoint()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")

    model = init_detector(
        CONFIG_PATH,
        checkpoint,
        device=device,
    )

    with open(TEST_MAPPING, "r") as f:
        mapping = json.load(f)

    results = []

    print("Running Cascade Mask R-CNN inference...")

    for item in tqdm(mapping):
        tif_name = item["file_name"]
        image_id = int(item["id"])
        height = int(item["height"])
        width = int(item["width"])

        png_name = Path(tif_name).with_suffix(".png").name
        img_path = os.path.join(TEST_IMAGE_DIR, png_name)

        if not os.path.exists(img_path):
            print(f"[WARN] missing converted test image: {img_path}")
            continue

        result = inference_detector(model, img_path)
        instances = get_instances(result)

        if len(instances) == 0:
            continue

        scores = tensor_to_numpy(instances.scores)
        labels = tensor_to_numpy(instances.labels).astype(np.int64)

        if hasattr(instances, "masks"):
            masks = tensor_to_numpy(instances.masks)
        else:
            continue

        order = np.argsort(-scores)
        order = order[:MAX_PER_IMAGE]

        for idx in order:
            score = float(scores[idx])

            if score < SCORE_THRESHOLD:
                continue

            label_zero_based = int(labels[idx])
            category_id = label_zero_based + 1

            if category_id < 1 or category_id > 4:
                continue

            binary_mask = masks[idx]

            if binary_mask.dtype != np.uint8:
                binary_mask = (binary_mask > MASK_THRESHOLD).astype(np.uint8)
            else:
                binary_mask = (binary_mask > 0).astype(np.uint8)

            if binary_mask.shape[0] != height or binary_mask.shape[1] != width:
                binary_mask = cv2.resize(
                    binary_mask,
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )

            pos = np.where(binary_mask > 0)

            if len(pos[0]) == 0:
                continue

            xmin = int(np.min(pos[1]))
            xmax = int(np.max(pos[1])) + 1
            ymin = int(np.min(pos[0]))
            ymax = int(np.max(pos[0])) + 1

            box_w = xmax - xmin
            box_h = ymax - ymin

            if box_w <= 0 or box_h <= 0:
                continue

            rle = encode_binary_mask(binary_mask)

            results.append(
                {
                    "image_id": int(image_id),
                    "bbox": [
                        float(xmin),
                        float(ymin),
                        float(box_w),
                        float(box_h),
                    ],
                    "score": float(score),
                    "category_id": int(category_id),
                    "segmentation": {
                        "size": [
                            int(rle["size"][0]),
                            int(rle["size"][1]),
                        ],
                        "counts": str(rle["counts"]),
                    },
                }
            )

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f)

    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(OUTPUT_JSON, arcname=OUTPUT_JSON)

    print(f"Done. Predictions: {len(results)}")
    print(f"Saved: {OUTPUT_ZIP}")

    with zipfile.ZipFile(OUTPUT_ZIP, "r") as zipf:
        print("Zip contents:", zipf.namelist())


if __name__ == "__main__":
    main()
