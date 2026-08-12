import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

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

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images_flat, labels, test_size=0.2, random_state=42)

# Train and evaluate a Logistic Regression model
lr = LogisticRegression(random_state=42)
lr.fit(X_train, y_train)
lr_train_acc = accuracy_score(y_train, lr.predict(X_train))
lr_test_acc = accuracy_score(y_test, lr.predict(X_test))
lr_f1 = f1_score(y_test, lr.predict(X_test), average='macro')
print(f"Logistic Regression Accuracy (Train): {lr_train_acc:.2f}")
print(f"Logistic Regression Accuracy (Test): {lr_test_acc:.2f}")
print(f"Logistic Regression F1-score (Test): {lr_f1:.2f}")