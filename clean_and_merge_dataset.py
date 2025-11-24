import os
import shutil
from pathlib import Path

# folders
original_images = Path("dataset/images")
original_labels = Path("dataset/labels")
hardest_images = Path("inference_results/hardest_10_percent")
corrected_images = Path("inference_results/hardest_10_percent")
corrected_labels = Path("dataset/corrected_images_labels")

# output folder
cleaned_images = Path("dataset/clean_dataset/images")
cleaned_labels = Path("dataset/clean_dataset/labels")

cleaned_images.mkdir(parents=True, exist_ok=True)
cleaned_labels.mkdir(parents=True, exist_ok=True)

# --- Step 1: Get filenames of images to remove ---
bad_filenames = {img.name for img in hardest_images.glob("*.png")}

print(f"Found {len(bad_filenames)} images to remove.")

# --- Step 2: Copy GOOD original images + labels into cleaned_dataset ---
for img_path in original_images.glob("*.png"):
    filename = img_path.name
    label_name = filename.replace(".png", ".txt")

    if filename in bad_filenames:
        continue  # skip bad ones

    # copy good image
    shutil.copy(img_path, cleaned_images / filename)

    # copy matching label if it exists
    label_path = original_labels / label_name
    if label_path.exists():
        shutil.copy(label_path, cleaned_labels / label_name)

print("Copied good images + labels.")

# --- Step 3: Add corrected images + their new labels ---
for img_path in corrected_images.glob("*.png"):
    filename = img_path.name
    label_name = filename.replace(".png", ".txt")

    # copy corrected image
    shutil.copy(img_path, cleaned_images / filename)

    # copy corrected label
    corrected_label = corrected_labels / label_name
    if corrected_label.exists():
        shutil.copy(corrected_label, cleaned_labels / label_name)

print("Added corrected images + labels.")
print("Done! Cleaned dataset created in cleaned_dataset/")
