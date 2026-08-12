import os
import cv2
import matplotlib.pyplot as plt

# Define the directory path
data_dir = r'D:\Capston Project\Dataset_mansi_500\Wrinkles_Folds_Coldshuts'

# List all image files in the directory
image_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('.jpg') or file.endswith('.png')]

print(f"Total images found: {len(image_files)}")

from skimage.feature import hog
import numpy as np

hog_features_all = []

for file in image_files:
    # Load each grayscale image
    image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

    # Compute HOG features
    hog_features, _ = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys')

    # Append to the list
    hog_features_all.append(hog_features)

print("HOG Features for all images have been extracted.")
