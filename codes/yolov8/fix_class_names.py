import json
from pathlib import Path

def fix_class_names():
    # Directory containing waist_folding JSON files
    waist_folding_dir = Path("C:/Users/PC/Desktop/Mansiba_Gohil/classification/ann_json/waist_folding")
    
    if not waist_folding_dir.exists():
        print(f"Directory not found: {waist_folding_dir}")
        return
    
    # Count of files modified
    modified_count = 0
    
    # Process each JSON file
    for json_path in waist_folding_dir.glob("*.json"):
        try:
            # Read the JSON file
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Check if we need to update class titles
            needs_update = False
            if 'objects' in data:
                for obj in data['objects']:
                    if obj.get('classTitle') == 'waist folding':
                        obj['classTitle'] = 'waist_folding'
                        needs_update = True
            
            # Save the updated file if needed
            if needs_update:
                with open(json_path, 'w') as f:
                    json.dump(data, f, indent=2)
                modified_count += 1
                
        except Exception as e:
            print(f"Error processing {json_path}: {e}")
    
    print(f"Updated {modified_count} JSON files")

if __name__ == "__main__":
    fix_class_names()