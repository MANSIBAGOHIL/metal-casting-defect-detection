import os
from PIL import Image

def check_image_dimensions(folder_path):
    # Dictionary to store image dimensions
    image_dimensions = {}

    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # Construct full file path
            img_path = os.path.join(folder_path, filename)
            
            try:
                # Open the image and get its dimensions
                with Image.open(img_path) as img:
                    width, height = img.size
                    
                    # Store dimensions in the dictionary
                    image_dimensions[filename] = (width, height)
                    
                    print(f"{filename}: {width}x{height}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    
    # Calculate and print statistics
    if image_dimensions:
        unique_dimensions = set(image_dimensions.values())
        print(f"\nTotal images processed: {len(image_dimensions)}")
        print(f"Number of unique dimensions: {len(unique_dimensions)}")
        print("Unique dimensions:")
        for dimension in unique_dimensions:
            count = list(image_dimensions.values()).count(dimension)
            print(f"  {dimension[0]}x{dimension[1]}: {count} image(s)")
    else:
        print("No images found in the specified folder.")

# Set your folder path
folder_path = r"D:\Capston Project\Defects_all\ALL\Defect Type\Veining"

# Call the function to check image dimensions
check_image_dimensions(folder_path)