import os
import json
import shutil
from pathlib import Path

# Updated to the correct path (with underscore)
root_dir = r"C:\Users\PC\Desktop\Mansiba_Gohil"
base_dir = Path("data/yolo_dataset")
images_dir = base_dir / "images"
labels_dir = base_dir / "labels"

# Create directories
os.makedirs(images_dir / "train", exist_ok=True)
os.makedirs(images_dir / "val", exist_ok=True)
os.makedirs(labels_dir / "train", exist_ok=True)
os.makedirs(labels_dir / "val", exist_ok=True)

# Define class names
classes = ["crease", "crescent_gap", "inclusion", "oil_spot", "punching_hole", 
           "rolled_pit", "silk_spot", "waist_folding", "water_spot", "welding_line"]

# Create a dictionary mapping class names to indices
class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

# Split ratio (80% train, 20% validation)
train_ratio = 0.8

# Check if the directories have the expected structure
ann_json_dir = Path(root_dir) / "classification" / "ann_json"
images_root_dir = Path(root_dir) / "classification" / "Images"

print(f"Checking ann_json directory: {ann_json_dir}")
print(f"Directory exists: {ann_json_dir.exists()}")

if ann_json_dir.exists():
    # List contents to see how the files are organized
    print("\nContents of ann_json directory:")
    for item in ann_json_dir.iterdir():
        print(f"- {item.name} ({'directory' if item.is_dir() else 'file'})")

print(f"\nChecking Images directory: {images_root_dir}")
print(f"Directory exists: {images_root_dir.exists()}")

if images_root_dir.exists():
    # List contents to see how the files are organized
    print("\nContents of Images directory:")
    for item in images_root_dir.iterdir():
        print(f"- {item.name} ({'directory' if item.is_dir() else 'file'})")

# Process each class
for class_name in classes:
    print(f"\nProcessing class: {class_name}")
    # First check if there's a dedicated folder for this class
    json_dir = ann_json_dir / class_name
    img_dir = images_root_dir / class_name
    
    # If class-specific directories don't exist, try using the parent directories
    if not json_dir.exists():
        print(f"Class directory not found at {json_dir}")
        json_dir = ann_json_dir
        
    if not img_dir.exists():
        print(f"Class directory not found at {img_dir}")
        img_dir = images_root_dir
    
    # Find JSON files that match this class
    all_json_files = list(json_dir.glob("*.json"))
    print(f"Found {len(all_json_files)} total JSON files")
    
    # Filter JSON files for the current class if we're using a common directory
    if json_dir == ann_json_dir:
        # We need to check inside each JSON file to determine the class
        class_json_files = []
        for json_path in all_json_files:
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    
                # Check if any object belongs to this class
                if 'objects' in data:
                    for obj in data['objects']:
                        if obj.get('classTitle') == class_name:
                            class_json_files.append(json_path)
                            break
            except Exception as e:
                print(f"Error reading {json_path}: {e}")
                
        json_files = class_json_files
    else:
        json_files = all_json_files
    
    print(f"Processing {len(json_files)} JSON files for class {class_name}")
    
    if len(json_files) == 0:
        continue
        
    total_files = len(json_files)
    train_count = int(total_files * train_ratio)
    
    for i, json_path in enumerate(json_files):
        # Determine if this file goes to train or val
        subset = "train" if i < train_count else "val"
        
        # Get image filename
        img_filename = json_path.stem + ".jpg"
        img_path = img_dir / img_filename
        
        if not img_path.exists():
            # If the image is not found in the class directory, check the parent directory
            if img_dir != images_root_dir:
                img_path = images_root_dir / img_filename
            
        if not img_path.exists():
            print(f"Warning: Image {img_filename} not found")
            continue
        
        # Copy image to YOLO dataset
        shutil.copy(img_path, images_dir / subset / img_filename)
        
        # Read JSON file and convert annotations to YOLO format
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            img_height = data['size']['height']
            img_width = data['size']['width']
            
            # Create YOLO format label file
            label_path = labels_dir / subset / (json_path.stem + ".txt")
            
            with open(label_path, 'w') as f:
                # Process only objects of the current class if we're using a common directory
                if 'objects' in data:
                    relevant_objects = []
                    for obj in data['objects']:
                        if json_dir == ann_json_dir:
                            # Only include objects of the current class
                            if obj.get('classTitle') == class_name:
                                relevant_objects.append(obj)
                        else:
                            # Include all objects
                            relevant_objects.append(obj)
                    
                    if not relevant_objects:
                        print(f"Warning: No relevant objects found in {json_path}")
                        continue
                        
                    for obj in relevant_objects:
                        # Get class index
                        class_title = obj.get('classTitle')
                        if not class_title:
                            print(f"Warning: classTitle not found in object in {json_path}")
                            continue
                            
                        class_idx = class_to_idx.get(class_title)
                        
                        if class_idx is None:
                            print(f"Warning: Unknown class {class_title}")
                            continue
                        
                        # Get bounding box coordinates
                        if ('points' not in obj or 
                            'exterior' not in obj['points'] or 
                            len(obj['points']['exterior']) < 2):
                            print(f"Warning: Invalid points data in {json_path}")
                            continue
                        
                        points = obj['points']['exterior']
                        x_min, y_min = points[0]
                        x_max, y_max = points[1]
                        
                        # Convert to YOLO format (center_x, center_y, width, height) normalized
                        center_x = (x_min + x_max) / (2 * img_width)
                        center_y = (y_min + y_max) / (2 * img_height)
                        width = (x_max - x_min) / img_width
                        height = (y_max - y_min) / img_height
                        
                        # Write to label file
                        f.write(f"{class_idx} {center_x} {center_y} {width} {height}\n")
                else:
                    print(f"Warning: No objects found in {json_path}")
        except Exception as e:
            print(f"Error processing {json_path}: {e}")

# Count the processed files
train_images = list((images_dir / "train").glob("*.jpg"))
val_images = list((images_dir / "val").glob("*.jpg"))
print(f"\nProcessed {len(train_images)} training images and {len(val_images)} validation images")

# Create YAML configuration file
yaml_path = base_dir / "metal_defects.yaml"
with open(yaml_path, 'w') as f:
    yaml_content = f"""
path: {base_dir.absolute()}
train: images/train
val: images/val

nc: {len(classes)}
names: {classes}
"""
    f.write(yaml_content)

print(f"Dataset conversion completed! YAML file created at {yaml_path}")
print(f"YAML content:\n{yaml_content}")