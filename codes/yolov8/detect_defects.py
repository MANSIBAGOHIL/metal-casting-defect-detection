from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import random

# Define class names
classes = ["crease", "crescent_gap", "inclusion", "oil_spot", "punching_hole", 
           "rolled_pit", "silk_spot", "waist_folding", "water_spot", "welding_line"]

def detect_defects(image_path, model_path=None, conf_threshold=0.25):
    """
    Detect metal defects in an image using trained YOLOv8 model
    """
    # Use best.pt from training or fallback to specified path
    if model_path is None:
        runs_dir = Path("runs/detect")
        if runs_dir.exists():
            # Try to find the model in the metal_defects folder first
            model_dir = runs_dir / "metal_defects"
            if model_dir.exists():
                model_path = model_dir / "weights/best.pt"
            
            # If not found, look in other training runs
            if not model_path or not Path(model_path).exists():
                train_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
                if train_dirs:
                    latest_train = max(train_dirs, key=lambda x: x.stat().st_mtime)
                    model_path = latest_train / "weights/best.pt"
                    if not model_path.exists():
                        model_path = latest_train / "weights/last.pt"
    
    if model_path is None or not Path(model_path).exists():
        raise FileNotFoundError("Model weights not found. Train the model first or provide a valid model path.")
    
    print(f"Using model: {model_path}")
    
    # Load model
    model = YOLO(model_path)
    
    # Run inference
    results = model(image_path, conf=conf_threshold)
    
    # Process results
    annotated_img = results[0].plot()
    
    # Get detection information
    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = classes[cls_id]
            conf = box.conf[0].item()
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            detections.append({
                "class": cls_name,
                "confidence": round(conf, 3),
                "bbox": [round(x, 2) for x in [x1, y1, x2, y2]]
            })
    
    return annotated_img, detections

def visualize_detections(image_path=None, model_path=None, save_output=True):
    """
    Visualize defect detections and save results
    """
    try:
        # If no image path provided, select a random validation image
        if image_path is None:
            val_dir = Path("data/yolo_dataset/images/val")
            if val_dir.exists():
                image_files = list(val_dir.glob("*.jpg"))
                if image_files:
                    image_path = str(random.choice(image_files))
                    print(f"Using random validation image: {image_path}")
                else:
                    print("No validation images found.")
                    return
            else:
                print("Validation directory not found.")
                return
        
        # Run detection
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image at {image_path}")
            return
        
        print(f"Detecting defects in image: {image_path}")
        annotated_img, detections = detect_defects(image_path, model_path)
        
        # Display results
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title("Metal Defect Detection")
        
        # Save output
        if save_output:
            output_dir = Path("results")
            os.makedirs(output_dir, exist_ok=True)
            output_path = output_dir / f"detection_{Path(image_path).name}"
            plt.savefig(output_path)
            print(f"Saved detection result to {output_path}")
        
        plt.show()
        
        # Print detections
        print("\nDetected defects:")
        for i, det in enumerate(detections):
            print(f"{i+1}. {det['class']} (Confidence: {det['confidence']:.3f}, BBox: {det['bbox']})")
            
        return detections
        
    except Exception as e:
        print(f"Error during detection: {e}")
        return None

if __name__ == "__main__":
    # Get user input for test image
    test_image = input("Enter path to test image (or press Enter for random validation image): ")
    
    if test_image:
        visualize_detections(test_image)
    else:
        visualize_detections()  # Use random validation image