import os


def fix_label_files(labels_dir, max_class_id=4):
    """
    Remove lines with invalid class IDs from label files

    Args:
        labels_dir: Directory containing .txt label files
        max_class_id: Maximum valid class ID (4 for 5 classes: 0-4)
    """
    fixed_count = 0
    removed_lines = 0

    for label_file in os.listdir(labels_dir):
        if not label_file.endswith('.txt'):
            continue

        label_path = os.path.join(labels_dir, label_file)

        with open(label_path, 'r') as f:
            lines = f.readlines()

        # Filter out invalid lines
        valid_lines = []
        file_had_issues = False

        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                if 0 <= class_id <= max_class_id:
                    valid_lines.append(line)
                else:
                    file_had_issues = True
                    removed_lines += 1
                    print(f"Removed invalid class {class_id} from {label_file}")

        # Write back only valid lines
        if file_had_issues:
            with open(label_path, 'w') as f:
                f.writelines(valid_lines)
            fixed_count += 1

    print(f"\n{'=' * 50}")
    print(f"Fixed {fixed_count} label files")
    print(f"Removed {removed_lines} invalid labels")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    # Fix training labels
    print("Fixing training labels...")
    fix_label_files('dataset/split_dataset/labels/train')

    # Fix validation labels
    print("\nFixing validation labels...")
    fix_label_files('dataset/split_dataset/labels/val')

    print("\nDone! You can now run train_yolo.py again.")