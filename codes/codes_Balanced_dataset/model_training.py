import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load training and testing datasets
train_data = pd.read_csv(r"D:\Capston Project\glcm_features_train.csv")
test_data = pd.read_csv(r"D:\Capston Project\glcm_features_test.csv")

# Split features and labels
X_train = train_data.drop(columns=['Filename', 'class'])  # Drop 'Filename' and 'class' columns
y_train = train_data['class']  # 'class' column as the label
X_test = test_data.drop(columns=['Filename', 'class'])  # Drop 'Filename' and 'class' columns
y_test = test_data['class']  # 'class' column as the label

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),  # Increased max_iter for convergence
    "Decision Tree": DecisionTreeClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Support Vector Classifier": SVC(kernel='linear', probability=True),  # Linear kernel
	"Naive Bayes": GaussianNB()
}

# Train and evaluate each model
for model_name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict on test data
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    
    print(f"\nModel: {model_name}")
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", matrix)