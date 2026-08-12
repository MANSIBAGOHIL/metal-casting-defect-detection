import os
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog
from skimage.measure import shannon_entropy
import matplotlib.pyplot as plt

# Set the path to your main dataset folder (not a specific defect type)
dataset_path = r"D:\Capston Project\Dataset_mansi_500"

# Function to load images from a directory
# Function to load images from a directory
def load_images(directory):
    images = []
    extensions = (".png", ".jpg", ".jpeg")  # Change list to tuple
    for filename in os.listdir(directory):
        if filename.lower().endswith(extensions):
            img = cv2.imread(os.path.join(directory, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
    return images
# Load images for each defect type
defect_types = [
    "Cutting_Marks", "Hot_tears_cracks", "Inclusion", "Porosity", "Scabs",
    "Shrink", "Surface_Roughness", "Veining", "Wrinkles_Folds_Coldshuts"
]

# Modify to load images from the correct directories
dataset = {defect: load_images(os.path.join(dataset_path, defect)) for defect in defect_types}

# Feature extraction functions remain unchanged
def extract_hog_features(image):
    hog_features, _ = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return hog_features

def extract_lbp_features(image, n_points=24, radius=3):
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

def extract_glcm_features(image):
    glcm = graycomatrix(image, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast').mean()
    dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    energy = graycoprops(glcm, 'energy').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    return [contrast, dissimilarity, homogeneity, energy, correlation]

def extract_haralick_features(image):
    glcm = graycomatrix(image, [1], [0], 256, symmetric=True, normed=True)
    return [
        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'dissimilarity')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0],
        graycoprops(glcm, 'energy')[0, 0],
        graycoprops(glcm, 'correlation')[0, 0],
        graycoprops(glcm, 'ASM')[0, 0]
    ]

def extract_entropy(image):
    return shannon_entropy(image)

# Function to extract all features from an image
def extract_all_features(image):
    hog_feat = extract_hog_features(image)
    lbp_feat = extract_lbp_features(image)
    glcm_feat = extract_glcm_features(image)
    haralick_feat = extract_haralick_features(image)
    entropy = extract_entropy(image)
    
    return np.concatenate([hog_feat, lbp_feat, glcm_feat, haralick_feat, [entropy]])

# Extract features for all images in the dataset
features = {defect: [] for defect in defect_types}

for defect, images in dataset.items():
    for image in images:
        features[defect].append(extract_all_features(image))

# Convert to numpy arrays
features = {defect: np.array(feat_list) for defect, feat_list in features.items()}

# Print feature shapes for each defect type
for defect, feat_array in features.items():
    print(f"{defect}: {feat_array.shape}")

# Visualize features for all defect types
def plot_all_defect_histograms(features, defect_types, num_features=5):
    for defect_type in defect_types:
        plt.figure(figsize=(15, 10))
        for i in range(num_features):
            plt.subplot(num_features, 1, i+1)
            plt.hist(features[defect_type][:, i], bins=50)
            plt.title(f'Feature {i+1} distribution for {defect_type}')
        plt.tight_layout()
        plt.savefig(f'{defect_type}_feature_distribution.png')
        plt.close()  # Close the figure to free up memory

# Plot histograms for all defect types
plot_all_defect_histograms(features, defect_types)

# Optional: Save features to a file
np.save('metal_defect_features.npy', features)