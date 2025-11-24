import json
import os
import shutil
from PIL import Image


def convert_bbox_coco_to_yolo(coco_bbox, img_w, img_h):
    x, y, w, h = coco_bbox
    xc = x + w / 2
    yc = y + h / 2

    return [
        xc / img_w,
        yc / img_h,
        w / img_w,
        h / img_h
    ]


def get_image_size(path):
    with Image.open(path) as img:
        return img.size  # (width, height)


def convert_coco_folder(coco_path, images_dir, output_images, output_labels, category_map):
    with open(coco_path, "r") as f:
        coco = json.load(f)

    # Copy all images first
    for img in coco["images"]:
        src = os.path.join(images_dir, img["file_name"])
        dst = os.path.join(output_images, img["file_name"])
        if not os.path.exists(dst):
            shutil.copy(src, dst)

    # Convert all annotations
    for ann in coco["annotations"]:
        img_id = ann["image_id"]

        # Find image file name
        img_info = next(i for i in coco["images"] if i["id"] == img_id)
        filename = img_info["file_name"]

        img_path = os.path.join(output_images, filename)
        img_w, img_h = get_image_size(img_path)

        yolo_box = convert_bbox_coco_to_yolo(ann["bbox"], img_w, img_h)

        cls = ann["category_id"]  # already 0–4 in this dataset

        # Output label file
        label_path = os.path.join(output_labels, filename.replace(".jpg", ".txt").replace(".png", ".txt"))

        with open(label_path, "a") as f:
            f.write(f"{cls} {' '.join(map(str, yolo_box))}\n")


def merge_folders():
    ROOT = "BTT_Data"
    FOLDERS = ["A", "B"]

    OUT = "dataset"
    IMG_OUT = os.path.join(OUT, "images")
    LAB_OUT = os.path.join(OUT, "labels")

    os.makedirs(IMG_OUT, exist_ok=True)
    os.makedirs(LAB_OUT, exist_ok=True)

    # These match the dataset categories in your project
    category_map = {
        0: "potted plant",
        1: "chair",
        2: "cup",
        3: "vase",
        4: "book",
    }

    for folder in FOLDERS:
        coco_path = os.path.join(ROOT, folder, "coco.json")
        images_dir = os.path.join(ROOT, folder, "images")
        print(f"Processing {coco_path} ...")
        convert_coco_folder(coco_path, images_dir, IMG_OUT, LAB_OUT, category_map)

    # dataset.yaml
    with open("dataset.yaml", "w") as f:
        f.write(
            "path: ./dataset\n"
            "train: images\n"
            "val: images\n"
            "nc: 5\n"
            "names: [\"potted plant\", \"chair\", \"cup\", \"vase\", \"book\"]\n"
        )

    print("✓ COCO → YOLO conversion complete!")
    print("✓ Images saved to: dataset/images/")
    print("✓ Labels saved to: dataset/labels/")


if __name__ == "__main__":
    merge_folders()