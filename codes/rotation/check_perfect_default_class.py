import pandas as pd

# Path to the CSV file
csv_path = r'D:\Capston Project\Augmentation\PCA_Features_With_Labels_test.csv'

# Load the CSV file
data = pd.read_csv(csv_path)

# Check for unique classes in the "Label" column
unique_classes = data['Label'].unique()

# Output the unique classes
print("Unique classes found in the dataset:", unique_classes)

# Verify if both classes (0 and 1) are present
if 0 in unique_classes and 1 in unique_classes:
    print("The dataset contains both classes: 0 (defect) and 1 (perfect).")
else:
    print("The dataset does not contain both classes.")
