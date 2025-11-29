from ultralytics import YOLO


def evaluate_models():
    """
    Compare baseline vs fine-tuned model
    """
    print("=" * 60)
    print("EVALUATING BASELINE vs FINE-TUNED MODEL")
    print("=" * 60)

    # Load models
    baseline_model = YOLO('yolo11n.pt')  # Original pretrained
    finetuned_model = YOLO('runs/train/yolo11_finetuned/weights/best.pt')  # Your trained model

    # Evaluate baseline
    print("\nEvaluating BASELINE model...")
    print("-" * 60)
    baseline_metrics = baseline_model.val(data='dataset.yaml', split='val')

    baseline_map50 = baseline_metrics.box.map50
    baseline_map = baseline_metrics.box.map

    print(f"\nBaseline Results:")
    print(f"  mAP@50: {baseline_map50:.4f}")
    print(f"  mAP@50-95: {baseline_map:.4f}")

    # Evaluate fine-tuned
    print("\nEvaluating FINE-TUNED model...")
    print("-" * 60)
    finetuned_metrics = finetuned_model.val(data='dataset.yaml', split='val')

    finetuned_map50 = finetuned_metrics.box.map50
    finetuned_map = finetuned_metrics.box.map

    print(f"\nFine-tuned Results:")
    print(f"  mAP@50: {finetuned_map50:.4f}")
    print(f"  mAP@50-95: {finetuned_map:.4f}")

    # Calculate improvement
    improvement = finetuned_map50 - baseline_map50

    print("\n" + "=" * 60)
    print("📈 COMPARISON")
    print("=" * 60)
    print(f"Baseline mAP@50:    {baseline_map50:.4f}")
    print(f"Fine-tuned mAP@50:  {finetuned_map50:.4f}")
    print(f"Improvement:        {improvement:.4f} ({improvement * 100:.2f}%)")
    print(f"\nGoal: 0.10 improvement")

    if improvement >= 0.10:
        print(f"Status:  GOAL ACHIEVED!")
    else:
        needed = 0.10 - improvement
        print(f"Status:   Need {needed:.4f} more improvement")
        print(f"Suggestion: Train for more epochs (30-50) to reach goal")

    print("=" * 60)

    # Per-class comparison
    print("\n📊 Per-Class Performance:")
    print("-" * 60)
    class_names = ['potted plant', 'chair', 'cup', 'vase', 'book']

    print(f"{'Class':<15} {'Baseline':<12} {'Fine-tuned':<12} {'Change':<10}")
    print("-" * 60)

    for i, name in enumerate(class_names):
        baseline_class = baseline_metrics.box.maps[i]
        finetuned_class = finetuned_metrics.box.maps[i]
        change = finetuned_class - baseline_class

        change_symbol = "+" if change > 0 else ""
        print(f"{name:<15} {baseline_class:.4f}       {finetuned_class:.4f}       {change_symbol}{change:.4f}")

    return {
        'baseline_map50': baseline_map50,
        'finetuned_map50': finetuned_map50,
        'improvement': improvement,
        'goal_achieved': improvement >= 0.10
    }


if __name__ == "__main__":
    results = evaluate_models()