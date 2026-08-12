import os
from PIL import Image
import random

# Paths
input_dir = r"D:\Capston Project\Balanced_Dataset\px500_Perfect_Samples"
output_dir = r"D:\Capston Project\Augmentation\Perfect_samples_rot"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Define rotation angles
rotation_angles = [90, 180, 270]

# List of images in the input directory
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

# Check if the input dataset matches the desired number of images
if len(image_files) != 1048:
    raise ValueError(f"The input dataset must contain exactly 1053 images. Current count: {len(image_files)}")

# Process each image
for filename in image_files:
    file_path = os.path.join(input_dir, filename)

    # Open the image
    with Image.open(file_path) as img:
        if img.mode != 'L':  # Convert to grayscale if not already
            img = img.convert('L')

        # Randomly select a rotation angle
        angle = random.choice(rotation_angles)

        # Apply the rotation
        rotated_img = img.rotate(angle, expand=True)

        # Save the augmented image
        output_filename = f"{os.path.splitext(filename)[0]}_rot{angle}.png"
        rotated_img.save(os.path.join(output_dir, output_filename))

print(f"Image augmentation completed. All 1053 images have been rotated and saved in '{output_dir}'.")
