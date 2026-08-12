import os
import random
import shutil

# Define paths
source_folder = r"D:\Capston Project\Augmentation\Rotation_dataset\Rot_Perfect_Samples"
train_folder = r"D:\Capston Project\Augmentation\train"
test_folder = r"D:\Capston Project\Augmentation\test"

# Create train and test folders if they don't exist
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# List all images in the source folder
image_files = [
    file for file in os.listdir(source_folder) 
    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
]

# Rename images starting with "perfect"
renamed_files = []
for file in image_files:
    if file.lower().startswith("perfect"):
        # Add unique suffix to avoid overwriting during renaming
        new_name = f"perfect_{random.randint(1000, 9999)}_{file}"
        os.rename(
            os.path.join(source_folder, file), 
            os.path.join(source_folder, new_name)
        )
        renamed_files.append(new_name)
    else:
        renamed_files.append(file)

# Ensure we have exactly 1,048 images after renaming
if len(renamed_files) != 4192:
    print(f"Expected 1,048 images, but found {len(renamed_files)}. Please check the source folder.")
else:
    # Randomly select 733 images for training
    train_images = random.sample(renamed_files, 3144)

    # Remaining 315 images go to the test set
    test_images = [file for file in renamed_files if file not in train_images]

    # Copy images to the train folder
    for file in train_images:
        src_path = os.path.join(source_folder, file)
        dst_path = os.path.join(train_folder, file)
        shutil.copy(src_path, dst_path)

    # Copy images to the test folder
    for file in test_images:
        src_path = os.path.join(source_folder, file)
        dst_path = os.path.join(test_folder, file)
        shutil.copy(src_path, dst_path)

    print(f"Split complete. 733 images copied to the train folder and 315 images copied to the test folder.")
