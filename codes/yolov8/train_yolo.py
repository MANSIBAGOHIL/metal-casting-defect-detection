from ultralytics import YOLO
from pathlib import Path
import torch

# Define paths
base_dir = Path("data/yolo_dataset")
yaml_path = base_dir / "metal_defects.yaml"

def train_model():
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        device = 0
    else:
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    # Load a pretrained model
    model = YOLO('yolov8m.pt')  # medium size model
    print(f"Model loaded: yolov8m.pt")
    
    # Training parameters
    params = {
        'data': str(yaml_path),
        'epochs': 50,              # Start with fewer epochs for testing
        'imgsz': 640,              # Image size
        'batch': 8,                # Adjust based on your GPU memory
        'patience': 15,            # Early stopping patience
        'save': True,              # Save checkpoints
        'cache': True,             # Cache images for faster training
        'device': device,          # Use GPU if available
        'workers': 4,              # Number of worker threads
        'project': 'runs/detect',  # Save results to this project
        'name': 'metal_defects',   # Name of the experiment
        'exist_ok': True,          # Overwrite existing experiment
    }
    
    print("Starting training with parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    # Train the model
    results = model.train(**params)
    
    # Evaluate the model on validation set
    print("\nEvaluating model on validation set...")
    metrics = model.val(data=str(yaml_path))
    
    # Print validation metrics
    print("\nValidation Metrics:")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"Precision: {metrics.box.precision:.4f}")
    print(f"Recall: {metrics.box.recall:.4f}")
    
    # Export the model to ONNX format
    print("\nExporting model to ONNX format...")
    model.export(format='onnx')
    
    return model, results

if __name__ == "__main__":
    # Verify YAML path exists
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML file not found at {yaml_path}. Run dataset conversion first.")
    
    # Train the model
    model, results = train_model()
    print("Training completed!")