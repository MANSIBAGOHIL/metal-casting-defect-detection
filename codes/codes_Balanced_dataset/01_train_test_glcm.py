import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops

# Define the path to your dataset folder
dataset_path = r"D:\Capston Project\Augmentation\train"
output_csv = r"D:\Capston Project\Augmentation\glcm_features_train.csv"

# Initialize a list to store the features and filenames
features_list = []

# Define GLCM properties to calculate
properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']

# Loop through each file in the dataset folder
for filename in os.listdir(dataset_path):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):

        # Check if the image filename starts with "perfect" for labeling
        if filename.lower().startswith('perfect'):
            label = 0  # Perfect images are labeled as 0
        else:
            label = 1  # All other images are labeled as 1 (defected)

        # Read image in grayscale
        image_path = os.path.join(dataset_path, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Check if image is loaded successfully
        if image is None:
            print(f"Could not load image {filename}. Skipping.")
            continue
        
        # Compute GLCM
        glcm = graycomatrix(image, distances=[1], angles=[0], symmetric=True, normed=True)
        
        # Extract GLCM properties
        feature_values = [filename, label]  # Add label column for the class (0 or 1)
        for prop in properties:
            feature = graycoprops(glcm, prop)[0, 0]
            feature_values.append(feature)
        
        # Append features to the list
        features_list.append(feature_values)

# Create a DataFrame with the results
columns = ['Filename', 'class'] + properties  # Add 'class' column
features_df = pd.DataFrame(features_list, columns=columns)

# Save the DataFrame to a CSV file
features_df.to_csv(output_csv, index=False)
print("GLCM feature extraction complete. Features saved to:", output_csv)
