import os
import cv2
import shutil

# Set the path to your main dataset folder (not a specific defect type)
dataset_path = r"D:\Capston Project\Dataset_mansi_300"

# Path to the output directory where processed images will be saved
output_path = r"D:\Capston Project\Processed_Images"
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Function to create directory if it doesn't exist
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Function to load and save images to a separate folder
def load_and_save_images(directory, defect_name):
    images = []
    defect_output_dir = os.path.join(output_path, defect_name)
    create_directory(defect_output_dir)
    
    for filename in os.listdir(directory):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            img = cv2.imread(os.path.join(directory, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                
                # Save a copy of the image to the corresponding folder in the output directory
                output_file_path = os.path.join(defect_output_dir, filename)
                cv2.imwrite(output_file_path, img)
            else:
                print(f"Failed to load: {filename} in {defect_name}")
    
    return images

# Load and save images for each defect type
defect_types = [
    "Cutting_Marks", "Hot_tears_cracks", "Inclusion", "Porosity", "Scabs",
    "Shrink", "Surface_Roughness", "Veining", "Wrinkles_Folds_Coldshuts"
]

# Modify to load and save images from the correct directories
dataset = {defect: load_and_save_images(os.path.join(dataset_path, defect), defect) for defect in defect_types}

# Verify that images were saved correctly
for defect in defect_types:
    defect_dir = os.path.join(output_path, defect)
    print(f"{defect}: {len(os.listdir(defect_dir))} images saved in {defect_dir}")
