from ultralytics import YOLO
import os


def train_yolo_model():
    """
    Fine-tune YOLOv11 on the cleaned dataset
    """
    # Load a pretrained YOLOv11 model
    # Use 'yolo11n.pt' for nano (fastest), 'yolo11s.pt' for small, 'yolo11m.pt' for medium
    model = YOLO('yolo11n.pt')

    # Train the model
    results = model.train(
        data='dataset.yaml',  # path to dataset YAML
        epochs=50,  # number of training epochs
        imgsz=640,  # input image size
        batch=16,  # batch size (reduce if memory issues)
        name='yolo11_finetuned',  # name for this training run
        device='cpu',  # use 'cuda' or '0' if you have GPU
        patience=10,  # early stopping patience
        save=True,  # save checkpoints
        save_period=10,  # save checkpoint every N epochs
        project='runs/train',  # project directory
        exist_ok=True,  # overwrite existing project
        pretrained=True,  # use pretrained weights
        optimizer='Adam',  # optimizer (Adam, SGD, etc.)
        lr0=0.001,  # initial learning rate
        lrf=0.01,  # final learning rate (lr0 * lrf)
        momentum=0.937,  # SGD momentum/Adam beta1
        weight_decay=0.0005,  # optimizer weight decay
        warmup_epochs=3,  # warmup epochs
        warmup_momentum=0.8,  # warmup initial momentum
        box=7.5,  # box loss gain
        cls=0.5,  # cls loss gain
        dfl=1.5,  # dfl loss gain
        plots=True,  # save plots and images during training
        val=True,  # validate during training
    )

    # Print training results
    print("\n" + "=" * 50)
    print("Training completed!")
    print("=" * 50)
    print(f"Results saved to: {results.save_dir}")
    print(f"Best model: {os.path.join(results.save_dir, 'weights', 'best.pt')}")
    print(f"Last model: {os.path.join(results.save_dir, 'weights', 'last.pt')}")

    return results

def predict_new_images():
    model = YOLO("runs/train/yolo11_finetuned/weights/best.pt")  # load fine-tuned model
    real_world_images = '/real_world_dataset copy'
    results = model.predict(
        source=real_world_images, 
        save=True,  # save images with predictions
        save_txt=True,  # save predictions as .txt files
        save_conf=True,  # save confidence scores
        conf=0.25,  # confidence threshold
        iou=0.45,  # NMS IOU threshold
        project='runs/predict',
        name='realworld_predictions',
        exist_ok=True,
    )
    
    # Process results
    for i, result in enumerate(results):
        print(f"\nImage {i+1}: {result.path}")
        print(f"Detected {len(result.boxes)} objects")
        
        # Access predictions
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            
            print(f"  Class: {model.names[class_id]}, "
                  f"Confidence: {confidence:.2f}, "
                  f"BBox: {bbox}")
    
    print(f"\nPredictions saved to: runs/predict/realworld_predictions")


if __name__ == "__main__":
    # Make sure you have the ultralytics package installed
    # pip install ultralytics

    print("Starting YOLOv11 fine-tuning...")
    print("=" * 50)

    results = train_yolo_model()
    real_world_results = predict_new_images()

    print("\nTraining metrics:")
    print(f"Final mAP@50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"Final mAP@50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")

    print("\nYOLOv11 fine-tuning and prediction complete!")