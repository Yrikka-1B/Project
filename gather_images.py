import json, shutil
from pathlib import Path

PROJECT = Path("/Users/vaisp/PyCharmMiscProject")

# path to cleaned coco file
json_path = PROJECT / "cocodata/clean_coco.json"

# path to keep all images into one place
dest = PROJECT / "cocodata/images_all"
dest.mkdir(parents=True, exist_ok=True)

# folders to use to searth for data, YOUR PATH WILL BE DIFFERENT!!, mine says my user (vaisp)
SEARCH_ROOTS = [
    Path("/Users/vaisp/dataset_cls/train/data1"),
    Path("/Users/vaisp/Downloads/yrikka-btt-aistudio-2025-main/BTT_Data"),
    Path("/Users/vaisp/PyCharmMiscProject/data/data1/images"),
]

#finding the path (recursively if needed)
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
        hits = list(root.rglob(name))
        if hits:
            hit = hits[0]
            break
    if hit:
        shutil.copy2(hit, dest / name)
        found += 1
    else:
        print("Missing:", name)
        missing += 1

#check to make sure all images have been transferred
print(f"\nConsolidated {found} images to {dest}")
print(f"Missing {missing} images that weren't found in SEARCH_ROOTS.")
print("Done!")
