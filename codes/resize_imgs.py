import os
from PIL import Image

def resize_images(source_folder, destination_folder, size=(500, 500)):
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Loop through all files in the source folder
    for filename in os.listdir(source_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # Open the image
            img_path = os.path.join(source_folder, filename)
            with Image.open(img_path) as img:
                # Resize the image
                img_resized = img.resize(size, Image.LANCZOS)
                
                # Save the resized image
                output_path = os.path.join(destination_folder, filename)
                img_resized.save(output_path)
                print(f"Resized and saved: {output_path}")

# Set your source and destination folders
source_folder = r"D:\Capston Project\Balanced_Dataset\Perfect_Samples"
destination_folder = r"D:\Capston Project\Balanced_Dataset\px500_Perfect_Samples"

# Call the function to resize images
resize_images(source_folder, destination_folder)