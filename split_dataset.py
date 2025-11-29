import os
import shutil
import random
from pathlib import Path


def split_dataset(clean_dataset_path, output_dir, train_split=0.8):
    """
    Split clean_dataset into train and validation sets

    Args:
        clean_dataset_path: Path to clean_dataset folder (contains images/ and labels/)
        output_dir: Output directory for split dataset
        train_split: Percentage of data for training (default 0.8 = 80%)
    """
    # Source directories
    source_images = os.path.join(clean_dataset_path, 'images')
    source_labels = os.path.join(clean_dataset_path, 'labels')

    # Create output directories
    train_img_dir = os.path.join(output_dir, 'images', 'train')
    val_img_dir = os.path.join(output_dir, 'images', 'val')
    train_label_dir = os.path.join(output_dir, 'labels', 'train')
    val_label_dir = os.path.join(output_dir, 'labels', 'val')

    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)

    # Get all image files
    image_files = [f for f in os.listdir(source_images)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    print(f"Found {len(image_files)} images in {source_images}")

    # Shuffle and split
    random.seed(42)  # For reproducibility
    random.shuffle(image_files)

    split_idx = int(len(image_files) * train_split)
    train_images = image_files[:split_idx]
    val_images = image_files[split_idx:]

    print(f"\nSplitting dataset:")
    print(f"  Total images: {len(image_files)}")
    print(f"  Training images: {len(train_images)} ({train_split * 100}%)")
    print(f"  Validation images: {len(val_images)} ({(1 - train_split) * 100}%)")

    # Copy training files
    print(f"\nCopying training files...")
    for img_file in train_images:
        # Copy image
        shutil.copy2(
            os.path.join(source_images, img_file),
            os.path.join(train_img_dir, img_file)
        )

        # Copy corresponding label
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(source_labels, label_file)
        if os.path.exists(label_path):
            shutil.copy2(label_path, os.path.join(train_label_dir, label_file))
        else:
            print(f"  Warning: Label not found for {img_file}")

    # Copy validation files
    print(f"Copying validation files...")
    for img_file in val_images:
        # Copy image
        shutil.copy2(
            os.path.join(source_images, img_file),
            os.path.join(val_img_dir, img_file)
        )

        # Copy corresponding label
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(source_labels, label_file)
        if os.path.exists(label_path):
            shutil.copy2(label_path, os.path.join(val_label_dir, label_file))
        else:
            print(f"  Warning: Label not found for {img_file}")

    print(f"\n{'=' * 50}")
    print("Dataset split complete!")
    print(f"{'=' * 50}")
    print(f"Output directory: {output_dir}")
    print(f"  Training images: {train_img_dir}")
    print(f"  Training labels: {train_label_dir}")
    print(f"  Validation images: {val_img_dir}")
    print(f"  Validation labels: {val_label_dir}")


if __name__ == "__main__":
    # Path to your clean_dataset folder
    clean_dataset_path = "dataset/clean_dataset"

    # Output directory for split dataset
    output_dir = "dataset/split_dataset"

    # Check if clean_dataset exists
    if not os.path.exists(clean_dataset_path):
        print(f"Error: {clean_dataset_path} does not exist!")
        print("\nPlease check your folder structure. It should be:")
        print("dataset/")
        print("  └── clean_dataset/")
        print("      ├── images/")
        print("      └── labels/")
    else:
        split_dataset(clean_dataset_path, output_dir, train_split=0.8)