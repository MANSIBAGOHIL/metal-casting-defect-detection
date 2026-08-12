import os
from PIL import Image

# Paths
input_dir = r"D:\Capston Project\Balanced_Dataset\px500_Perfect_Samples"
output_dir = r"D:\Capston Project\Augmentation\Rotation_dataset\Rot_Perfect_Samples"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Define rotation angles
rotation_angles = [90, 180, 270]

# Process each image
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        file_path = os.path.join(input_dir, filename)
        
        # Open the image
        with Image.open(file_path) as img:
            if img.mode != 'L':  # Convert to grayscale if not already
                img = img.convert('L')
            
            # Save original image (if needed)
            img.save(os.path.join(output_dir, filename))
            
            # Apply rotations and save augmented images
            for angle in rotation_angles:
                rotated_img = img.rotate(angle, expand=True)
                output_filename = f"{os.path.splitext(filename)[0]}_rot{angle}.png"
                rotated_img.save(os.path.join(output_dir, output_filename))
                print(f"Saved rotated image: {output_filename}")

print("Image rotation augmentation completed.")
