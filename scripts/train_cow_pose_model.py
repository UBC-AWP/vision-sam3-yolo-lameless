#!/usr/bin/env python3
"""
Train YOLOv8-Pose model on custom cow pose data for lameness detection.

Usage:
    python scripts/train_cow_pose_model.py --data data/cow_pose_custom/cow_pose.yaml --epochs 100
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def train_cow_pose_model(
    data_yaml: str,
    model: str = "yolov8n-pose.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    project: str = "runs/pose",
    name: str = "cow_pose",
    device: str = "",
):
    """
    Train YOLOv8-Pose model on cow pose data.
    
    Args:
        data_yaml: Path to dataset YAML file
        model: Base model to fine-tune from
        epochs: Number of training epochs
        imgsz: Input image size
        batch: Batch size
        project: Project directory for saving results
        name: Experiment name
        device: Device to train on ('' for auto, '0' for GPU 0, 'cpu' for CPU)
    """
    print("="*60)
    print("COW POSE MODEL TRAINING")
    print("="*60)
    print(f"Dataset: {data_yaml}")
    print(f"Base model: {model}")
    print(f"Epochs: {epochs}")
    print(f"Image size: {imgsz}")
    print(f"Batch size: {batch}")
    print("="*60)
    
    # Load base model
    yolo = YOLO(model)
    
    # Train
    results = yolo.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=name,
        device=device,
        # Augmentation settings optimized for pose
        mosaic=0.5,
        mixup=0.1,
        degrees=10,  # Rotation
        translate=0.1,
        scale=0.3,
        shear=5,
        perspective=0.0001,
        flipud=0.0,  # Don't flip vertically
        fliplr=0.5,  # Horizontal flip (cow can face left or right)
        # Training settings
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        # Pose-specific
        pose=12.0,  # Pose loss weight
        kobj=1.0,   # Keypoint objectness loss
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    
    # Get best model path
    best_model = Path(project) / name / "weights" / "best.pt"
    print(f"Best model: {best_model}")
    
    # Copy to models directory
    models_dir = Path("data/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    if best_model.exists():
        import shutil
        dest = models_dir / "cow_pose_lameness.pt"
        shutil.copy(best_model, dest)
        print(f"Copied to: {dest}")
    
    return results


def validate_model(model_path: str, data_yaml: str):
    """Validate trained model."""
    print("\n" + "="*60)
    print("VALIDATING MODEL")
    print("="*60)
    
    yolo = YOLO(model_path)
    results = yolo.val(data=data_yaml)
    
    print(f"\nValidation Results:")
    print(f"  Box mAP50: {results.box.map50:.3f}")
    print(f"  Box mAP50-95: {results.box.map:.3f}")
    print(f"  Pose mAP50: {results.pose.map50:.3f}")
    print(f"  Pose mAP50-95: {results.pose.map:.3f}")
    
    return results


def test_inference(model_path: str, image_or_video: str, output_dir: str = "runs/pose/test"):
    """Test inference on image or video."""
    print("\n" + "="*60)
    print("TESTING INFERENCE")
    print("="*60)
    
    yolo = YOLO(model_path)
    results = yolo.predict(
        source=image_or_video,
        save=True,
        project=output_dir,
        name="test",
        conf=0.5,
        show_labels=True,
        show_conf=True,
    )
    
    print(f"Results saved to: {output_dir}/test")
    return results


def main():
    parser = argparse.ArgumentParser(description="Train Cow Pose Model")
    parser.add_argument("--data", required=True, help="Path to dataset YAML")
    parser.add_argument("--model", default="yolov8n-pose.pt", help="Base model")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", default="", help="Device ('' for auto)")
    parser.add_argument("--validate", action="store_true", help="Run validation only")
    parser.add_argument("--test", help="Test on image/video")
    parser.add_argument("--weights", help="Model weights for validation/test")
    
    args = parser.parse_args()
    
    if args.validate and args.weights:
        validate_model(args.weights, args.data)
    elif args.test and args.weights:
        test_inference(args.weights, args.test)
    else:
        train_cow_pose_model(
            data_yaml=args.data,
            model=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
        )


if __name__ == "__main__":
    main()

