import json, shutil
from pathlib import Path

# === CONFIGURATION ===
# Your project folder path
PROJECT = Path("/Users/vaisp/PyCharmMiscProject")

# Path to your cleaned COCO annotation file
json_path = PROJECT / "cocodata/clean_coco.json"

# This folder will collect all your images in one place
dest = PROJECT / "cocodata/images_all"
dest.mkdir(parents=True, exist_ok=True)

# These are the folders where your images currently exist.
# (You found them earlier in your 'find' results — add more if needed)
SEARCH_ROOTS = [
    Path("/Users/vaisp/dataset_cls/train/data1"),
    Path("/Users/vaisp/Downloads/yrikka-btt-aistudio-2025-main/BTT_Data"),
    Path("/Users/vaisp/PyCharmMiscProject/data/data1/images"),
]

# === MAIN SCRIPT ===
names = set()
with open(json_path) as f:
    coco = json.load(f)
for im in coco["images"]:
    names.add(im["file_name"])

found, missing = 0, 0
for name in names:
    hit = None
    for root in SEARCH_ROOTS:
        cand = root / name
        if cand.exists():
            hit = cand
            break
        # Try finding it recursively if it's in a subfolder
        hits = list(root.rglob(name))
        if hits:
            hit = hits[0]
            break
    if hit:
        shutil.copy2(hit, dest / name)
        found += 1
    else:
        print("❌ Missing:", name)
        missing += 1

print(f"\n✅ Consolidated {found} images to {dest}")
print(f"⚠️ Missing {missing} images that weren't found in SEARCH_ROOTS.")
print("Done!")
