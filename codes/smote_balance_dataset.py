import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter
import matplotlib.pyplot as plt

# Define the defect types
defect_types = [
    "Cutting_Marks", "Hot_tears_cracks", "Inclusion", "Porosity", "Scabs",
    "Shrink", "Surface_Roughness", "Veining", "Wrinkles_Folds_Coldshuts"
]

# Load the features from the .npy file
features = np.load('metal_defect_features.npy', allow_pickle=True).item()

# Combine all features into a single array
X = np.vstack([features[defect] for defect in defect_types])

# Create labels
y = np.concatenate([np.full(len(features[defect]), i) for i, defect in enumerate(defect_types)])

# Print original class distribution
print("Original class distribution:")
print(Counter(y))

# Create a resampling pipeline
over = SMOTE(sampling_strategy='auto', random_state=42)
under = RandomUnderSampler(sampling_strategy='auto', random_state=42)
steps = [('o', over), ('u', under)]
pipeline = Pipeline(steps=steps)

# Apply the resampling
X_resampled, y_resampled = pipeline.fit_resample(X, y)

# Print resampled class distribution
print("\nResampled class distribution:")
print(Counter(y_resampled))

# Visualize class distribution before and after resampling
def plot_class_distribution(y_original, y_resampled):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    ax1.bar(range(len(defect_types)), [Counter(y_original)[i] for i in range(len(defect_types))])
    ax1.set_title('Original Class Distribution')
    ax1.set_xticks(range(len(defect_types)))
    ax1.set_xticklabels(defect_types, rotation=45, ha='right')
    ax1.set_ylabel('Number of Samples')
    
    ax2.bar(range(len(defect_types)), [Counter(y_resampled)[i] for i in range(len(defect_types))])
    ax2.set_title('Resampled Class Distribution')
    ax2.set_xticks(range(len(defect_types)))
    ax2.set_xticklabels(defect_types, rotation=45, ha='right')
    ax2.set_ylabel('Number of Samples')
    
    plt.tight_layout()
    plt.show()

plot_class_distribution(y, y_resampled)

# Reshape the resampled data back into the original format
balanced_features = {defect: X_resampled[y_resampled == i] for i, defect in enumerate(defect_types)}

# Print the new shapes of each defect type's feature array
for defect, feat_array in balanced_features.items():
    print(f"{defect}: {feat_array.shape}")

# Visualize feature distributions for each defect type before and after balancing
def plot_feature_distributions(original_features, balanced_features, defect_type):
    fig, axes = plt.subplots(5, 2, figsize=(15, 25))
    fig.suptitle(f'Feature Distributions for {defect_type}', fontsize=16)
    
    for i in range(5):  # For each of the 5 features
        # Original distribution
        axes[i, 0].hist(original_features[defect_type][:, i], bins=50, alpha=0.7)
        axes[i, 0].set_title(f'Original - Feature {i+1}')
        axes[i, 0].set_ylabel('Frequency')
        
        # Balanced distribution
        axes[i, 1].hist(balanced_features[defect_type][:, i], bins=50, alpha=0.7)
        axes[i, 1].set_title(f'Balanced - Feature {i+1}')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()

# Plot feature distributions for each defect type
for defect in defect_types:
    plot_feature_distributions(features, balanced_features, defect)

# Save the balanced features
np.save('balanced_metal_defect_features.npy', balanced_features)