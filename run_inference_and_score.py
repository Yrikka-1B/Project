import os
import shutil
import json
from ultralytics import YOLO
import numpy as np

def iou(box1, box2):
    # box format: [xc, yc, w, h] in normalized coords (YOLO)
    # convert to x1,y1,x2,y2
    def convert(box):
        xc, yc, w, h = box
        x1 = xc - w/2
        y1 = yc - h/2
        x2 = xc + w/2
        y2 = yc + h/2
        return x1, y1, x2, y2

    ax1, ay1, ax2, ay2 = convert(box1)
    bx1, by1, bx2, by2 = convert(box2)

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    area1 = (ax2 - ax1) * (ay2 - ay1)
    area2 = (bx2 - bx1) * (by2 - by1)

    return inter_area / (area1 + area2 - inter_area + 1e-6)

def load_ground_truth(label_path):
    if not os.path.exists(label_path):
        return []
    boxes = []
    with open(label_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            cls = int(parts[0])
            xywh = list(map(float, parts[1:5]))
            boxes.append(xywh)
    return boxes

def score_image(gt_boxes, pred_boxes):
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return 0  # perfect
    if len(gt_boxes) == 0 and len(pred_boxes) > 0:
        return 5 + len(pred_boxes)
    if len(gt_boxes) > 0 and len(pred_boxes) == 0:
        return 5 + len(gt_boxes)

    # compute IOU mismatch penalties
    total_iou = 0
    matches = 0
    for gt in gt_boxes:
        best_iou = 0
        for pred in pred_boxes:
            best_iou = max(best_iou, iou(gt, pred))
        total_iou += best_iou
        matches += 1

    avg_iou = total_iou / matches
    error = 1 - avg_iou  # higher = worse

    # add penalty for mismatch in number of boxes
    error += abs(len(gt_boxes) - len(pred_boxes)) * 0.5

    return error

def run():
    model = YOLO("yolo11n.pt")  # load small YOLO

    IMG_DIR = "dataset/images"
    LAB_DIR = "dataset/labels"

    OUT_DIR = "inference_results"
    BAD_OUT = os.path.join(OUT_DIR, "hardest_10_percent")
    os.makedirs(BAD_OUT, exist_ok=True)

    results_file = os.path.join(OUT_DIR, "results.json")
    os.makedirs(OUT_DIR, exist_ok=True)

    image_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(("jpg", "png"))]
    scored = []

    for img in image_files:
        pred = model(os.path.join(IMG_DIR, img))[0]

        pred_boxes = []
        for b in pred.boxes.xywhn.cpu().numpy():
            pred_boxes.append(b.tolist())

        gt = load_ground_truth(os.path.join(LAB_DIR, img.replace(".jpg", ".txt").replace(".png", ".txt")))

        error = score_image(gt, pred_boxes)

        scored.append({
            "image": img,
            "error_score": float(error),
            "num_gt": len(gt),
            "num_pred": len(pred_boxes)
        })

    # Save full results
    with open(results_file, "w") as f:
        json.dump(scored, f, indent=4)

    # Sort by error score descending (worst first)
    scored.sort(key=lambda x: x["error_score"], reverse=True)

    # Select worst 10%
    top10 = scored[:max(1, len(scored)//10)]

    # Copy images
    for item in top10:
        src = os.path.join(IMG_DIR, item["image"])
        dst = os.path.join(BAD_OUT, item["image"])
        shutil.copy(src, dst)

    print("Done! Worst 10% images saved in:")
    print(BAD_OUT)

    

if __name__ == "__main__":
    run()