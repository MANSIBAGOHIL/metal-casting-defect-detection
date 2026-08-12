import os
import random
import shutil

# Define the source and destination paths
source_folder = r"D:\Capston Project\Dataset_mansi_500\Wrinkles_Folds_Coldshuts"
destination_folder = r"D:\Capston Project\Balanced_Dataset\Defect_Samples"

# List all files in the source folder, filtering for image files only
image_files = [file for file in os.listdir(source_folder) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

# Check if there are enough images to copy
if len(image_files) < 117:
    print(f"Not enough images to select 117. Only {len(image_files)} available.")
else:
    # Randomly select 117 images
    selected_images = random.sample(image_files, 117)

    # Copy each selected file to the destination folder
    for file in selected_images:
        src_path = os.path.join(source_folder, file)
        dst_path = os.path.join(destination_folder, file)
        shutil.copy(src_path, dst_path)

    print("Copying complete. 117 images have been copied to the destination folder.")
