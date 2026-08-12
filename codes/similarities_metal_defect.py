import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from PIL import Image
from tqdm import tqdm

# Set the dataset directory
dataset_dir = "D:\Capston Project\Dataset_mansi_300"

# Get a list of all the defect classes
defect_classes = os.listdir(dataset_dir)

# Load all the images and their labels
images = []
labels = []
for defect_class in defect_classes:
    class_dir = os.path.join(dataset_dir, defect_class)
    for filename in os.listdir(class_dir):
        img = Image.open(os.path.join(class_dir, filename))
        images.append(np.array(img))
        labels.append(defect_class)

# Convert the images to a numpy array
images = np.array(images)

# Flatten the images to 1D vectors
images_flat = images.reshape(images.shape[0], -1)

# Perform PCA to reduce dimensionality
pca = PCA(n_components=50)
images_pca = pca.fit_transform(images_flat)

# Perform t-SNE to visualize the similarities
tsne = TSNE(n_components=2)
images_tsne = tsne.fit_transform(images_pca)

# Plot the t-SNE visualization
plt.figure(figsize=(10, 10))
for i, label in enumerate(set(labels)):
    mask = np.array([l == label for l in labels])
    plt.scatter(images_tsne[mask, 0], images_tsne[mask, 1], label=label)
plt.legend()
plt.title("Metal Defect Similarity Visualization")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.savefig("metal_defect_similarity.png")