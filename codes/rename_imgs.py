import os

# Define the path to your folder
folder_path = r"D:\Capston Project\new dataset\XL Pipe Connector"

# List all files in the folder
files = os.listdir(folder_path)

# Filter only image files, making the extension check case-insensitive
image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

# Sort the files to maintain order
image_files.sort()

# Step 1: Temporary renaming to avoid conflicts
for index, file in enumerate(image_files):
    # Generate new temporary file name
    temp_name = f"temp_inclusion_{index + 1}{os.path.splitext(file)[1]}"
    
    # Build full old and temporary file paths
    old_path = os.path.join(folder_path, file)
    temp_path = os.path.join(folder_path, temp_name)
    
    # Rename the file to temporary name
    os.rename(old_path, temp_path)

# Step 2: Final renaming to desired format
temp_files = os.listdir(folder_path)
temp_files = [file for file in temp_files if file.startswith("temp_inclusion_")]

# Sort temporary files to maintain order
temp_files.sort()

for index, file in enumerate(temp_files):
    # Generate final file name
    final_name = f"XL_Pipe_Connector_{index + 1}{os.path.splitext(file)[1]}"
    
    # Build full temporary and final file paths
    temp_path = os.path.join(folder_path, file)
    final_path = os.path.join(folder_path, final_name)
    
    # Rename the file to final name
    os.rename(temp_path, final_path)

print("Renaming complete.")
