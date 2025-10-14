#!/usr/bin/env python3
"""
converterscript.py
- Converts COCO -> YOLO for exactly 5 classes
- Copies images from a single consolidated folder (cocodata/images_all)
- Splits 80/10/10 into train/val/test
- Writes yolo_dataset/data.yaml
"""

import json, shutil, random
from pathlib import Path
from collections import defaultdict
from PIL import Image

# configure the path
PROJECT_ROOT = Path("/Users/vaisp/PyCharmMiscProject")

COCO_JSON    = PROJECT_ROOT / "cocodata/clean_coco.json"
IMAGES_ROOT  = PROJECT_ROOT / "cocodata/images_all"   
OUT_DIR      = PROJECT_ROOT / "yolo_dataset"

#5 classes to train
KEEP_CLASSES = ["chair", "potted plant", "vase", "book", "cup"]

# test train split values
SPLIT_RATIOS = (0.8, 0.1, 0.1)  # 80/10/10
SEED         = 42

#converts the coco to yaml format for YOLO model
def to_xywh_norm(xywh, W, H):
    x, y, w, h = xywh
    xc = (x + w / 2.0) / max(W, 1)
    yc = (y + h / 2.0) / max(H, 1)
    return xc, yc, w / max(W, 1), h / max(H, 1)

def main():
    random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for sp in ["train", "val", "test"]:
        (OUT_DIR / "images" / sp).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels" / sp).mkdir(parents=True, exist_ok=True)

    with open(COCO_JSON, "r") as f:
        coco = json.load(f)

    # uses the categories predefined
    coco_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}

    # keeps the 5 classes needed
    name_to_yolo = {name: i for i, name in enumerate(KEEP_CLASSES)}
    yolo_to_name = {i: n for n, i in name_to_yolo.items()}
    keep_coco_ids = {cid for cid, n in coco_id_to_name.items() if n in KEEP_CLASSES}

    # groups annotations, but only based on the class
    images = {im["id"]: im for im in coco["images"]}
    anns_by_im = defaultdict(list)
    for a in coco["annotations"]:
        if a.get("iscrowd", 0):
            # skip crowd boxes
            continue
        if a["category_id"] in keep_coco_ids:
            anns_by_im[a["image_id"]].append(a)

    # if an image contains at least one annotation then keep it
    kept_image_ids = [iid for iid, L in anns_by_im.items() if len(L) > 0]
    total_kept = len(kept_image_ids)
    if total_kept == 0:
        print("❌ No images contain the 5 target classes. Check KEEP_CLASSES or your COCO JSON.")
        return

    # split
    random.shuffle(kept_image_ids)
    r_train, r_val, r_test = SPLIT_RATIOS
    n_train = int(r_train * total_kept)
    n_val   = int(r_val   * total_kept)
    train_ids = set(kept_image_ids[:n_train])
    val_ids   = set(kept_image_ids[n_train:n_train+n_val])
    test_ids  = set(kept_image_ids[n_train+n_val:])

    def which_split(iid):
        if iid in train_ids: return "train"
        if iid in val_ids:   return "val"
        return "test"

    copied, missing = 0, 0
    label_counts = {"train": 0, "val": 0, "test": 0}

    for iid in kept_image_ids:
        im = images[iid]
        fname = im["file_name"]  
        src = IMAGES_ROOT / fname
        if not src.exists():
            #test to make sure all images are gathered
            print("Missing image (not in images_all):", fname)
            missing += 1
            continue

        # width / height of the images
        W = im.get("width")
        H = im.get("height")
        if not W or not H:
            try:
                with Image.open(src) as pil:
                    W, H = pil.size
            except Exception:
                print("Could not read image size for:", src)
                missing += 1
                continue

        sp = which_split(iid)
        dst_img = OUT_DIR / "images" / sp / src.name
        shutil.copy2(src, dst_img)
        copied += 1

        # yolo label file
        lines = []
        for a in anns_by_im[iid]:
            cname = coco_id_to_name[a["category_id"]]
            yolo_c = name_to_yolo[cname]          # 0..4
            xc, yc, bw, bh = to_xywh_norm(a["bbox"], W, H)
            lines.append(f"{yolo_c} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        (OUT_DIR / "labels" / sp / (dst_img.stem + ".txt")).write_text("\n".join(lines))
        label_counts[sp] += 1

    # yaml file for the conversion
    yaml = (
        f"path: {OUT_DIR}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n\n"
        f"names:\n" +
        "\n".join([f"  {i}: {yolo_to_name[i]}" for i in range(len(KEEP_CLASSES))]) +
        "\n"
    )
    (OUT_DIR / "data.yaml").write_text(yaml)

    #test just to make sure all images have been copied, data split into test and train and val, and yaml file created
    print(f"   Copied images: {copied}")
    print(f"   Missing images (not found under {IMAGES_ROOT}): {missing}")
    print(f"   Labels written -> train: {label_counts['train']}, val: {label_counts['val']}, test: {label_counts['test']}")
    print(f"YOLO dataset at: {OUT_DIR}")
    print(f"data.yaml at:     {OUT_DIR / 'data.yaml'}")

if __name__ == "__main__":
    main()
