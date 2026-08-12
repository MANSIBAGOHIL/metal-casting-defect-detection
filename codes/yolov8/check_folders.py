import os
from pathlib import Path

# Update path to match your directory name (with underscore instead of space)
base_path = Path(r"C:\Users\PC\Desktop\Mansiba_Gohil")

# Check if the base directory exists
print(f"Base directory exists: {base_path.exists()}")

# List contents of the base directory
if base_path.exists():
    print("\nContents of base directory:")
    for item in base_path.iterdir():
        print(f"- {item.name} ({'directory' if item.is_dir() else 'file'})")

    # Look for classification directory or similar
    classification_dirs = [d for d in base_path.iterdir() if d.is_dir()]
    
    for dir_path in classification_dirs:
        print(f"\nChecking contents of {dir_path.name}:")
        try:
            for item in dir_path.iterdir():
                print(f"- {item.name} ({'directory' if item.is_dir() else 'file'})")
        except PermissionError:
            print(f"Permission denied to access {dir_path}")