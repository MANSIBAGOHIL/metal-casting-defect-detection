from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import cv2
import random
import os

# Define class names
classes = ["crease", "crescent_gap", "inclusion", "oil_spot", "punching_hole", 
           "rolled_pit", "silk_spot", "waist_folding", "water_spot", "welding_line"]

def evaluate_model():
    """
    Comprehensive evaluation of the trained YOLOv8 model
    """
    # Find the best model
    runs_dir = Path("runs/detect")
    if not runs_dir.exists():
        print("No training runs found. Train the model first.")
        return
    
    # Try to find the model in the metal_defects folder first
    model_dir = runs_dir / "metal_defects"
    if model_dir.exists():
        model_path = model_dir / "weights/best.pt"
    else:
        # If not found, look in other training runs
        train_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
        if not train_dirs:
            print("No training directories found.")
            return
        
        latest_train = max(train_dirs, key=lambda x: x.stat().st_mtime)
        model_path = latest_train / "weights/best.pt"
        if not model_path.exists():
            model_path = latest_train / "weights/last.pt"
    
    if not model_path.exists():
        print("Model weights not found.")
        return
    
    print(f"Using model: {model_path}")
    
    # Load model
    model = YOLO(model_path)
    
    # Path to validation data
    yaml_path = Path("data/yolo_dataset/metal_defects.yaml")
    
    # Run validation
    metrics = model.val(data=str(yaml_path), verbose=True)
    
    # Print summary metrics
    print("\nValidation Metrics:")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"Precision: {metrics.box.precision:.4f}")
    print(f"Recall: {metrics.box.recall:.4f}")
    
    # Create results directory
    results_dir = Path("evaluation_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Visualize some example predictions
    val_img_dir = Path("data/yolo_dataset/images/val")
    if val_img_dir.exists():
        # Get 5 random validation images
        val_images = list(val_img_dir.glob("*.jpg"))
        if len(val_images) > 0:
            sample_images = random.sample(val_images, min(5, len(val_images)))
            
            plt.figure(figsize=(20, 12))
            for i, img_path in enumerate(sample_images):
                # Run inference
                results = model(str(img_path))
                
                # Get image with predictions
                result_img = results[0].plot()
                
                # Display
                plt.subplot(2, 3, i+1)
                plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
                plt.title(f"Validation Image {i+1}")
                plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(results_dir / "sample_predictions.png")
            plt.show()
    
    # Create per-class metrics visualization
    class_metrics = metrics.class_dict
    if class_metrics:
        # Extract metrics per class
        class_data = {
            'Class': classes,
            'Precision': [class_metrics[i].get('precision', 0) for i in range(len(classes))],
            'Recall': [class_metrics[i].get('recall', 0) for i in range(len(classes))],
            'mAP50': [class_metrics[i].get('map50', 0) for i in range(len(classes))],
            'mAP50-95': [class_metrics[i].get('map', 0) for i in range(len(classes))]
        }
        
        # Create DataFrame
        df = pd.DataFrame(class_data)
        
        # Plot class performance
        plt.figure(figsize=(12, 10))
        metrics_to_plot = ['Precision', 'Recall', 'mAP50', 'mAP50-95']
        
        for i, metric in enumerate(metrics_to_plot):
            plt.subplot(2, 2, i+1)
            sns.barplot(x='Class', y=metric, data=df)
            plt.title(f'Per-Class {metric}')
            plt.xticks(rotation=45, ha='right')
            plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(results_dir / "class_performance.png")
        plt.show()
        
        # Save metrics to CSV
        df.to_csv(results_dir / "class_metrics.csv", index=False)
        print(f"Class metrics saved to {results_dir / 'class_metrics.csv'}")
    
    return metrics

if __name__ == "__main__":
    evaluate_model()