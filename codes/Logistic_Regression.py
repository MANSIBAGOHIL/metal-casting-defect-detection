import os
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog
from skimage.measure import shannon_entropy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Set the path to your main dataset folder
dataset_path = r"D:\Capston Project\Dataset_mansi_500"

# Function to load images from a directory
def load_images(directory):
    images = []
    extensions = (".png", ".jpg", ".jpeg")
    for filename in os.listdir(directory):
        if filename.lower().endswith(extensions):
            img = cv2.imread(os.path.join(directory, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
    return images

# Define defect types
defect_types = [
    "Cutting_Marks", "Hot_tears_cracks", "Inclusion", "Porosity", "Scabs",
    "Shrink", "Surface_Roughness", "Veining", "Wrinkles_Folds_Coldshuts"
]

# Load images for each defect type
dataset = {defect: load_images(os.path.join(dataset_path, defect)) for defect in defect_types}

# Feature extraction functions
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

# Prepare data for logistic regression
X = []
y = []
label_mapping = {defect: idx for idx, defect in enumerate(defect_types)}

for defect, feat_array in features.items():
    X.extend(feat_array)  # Add all feature vectors from this defect type
    y.extend([label_mapping[defect]] * len(feat_array))  # Label each vector with the defect type

X = np.array(X)
y = np.array(y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Logistic Regression model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# Make predictions
y_pred = log_reg.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=defect_types))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Optional: Save features to a file
np.save('metal_defect_features.npy', features)