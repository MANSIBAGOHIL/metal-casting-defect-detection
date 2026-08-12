import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# Load training and testing datasets
train_data = pd.read_csv(r"D:\Capston Project\Augmentation\glcm_features_train.csv")
test_data = pd.read_csv(r"D:\Capston Project\Augmentation\glcm_features_test.csv")

# Split features and labels
X_train = train_data.drop(columns=['Filename', 'class'])  # Drop 'Filename' and 'class' columns
y_train = train_data['class']  # 'class' column as the label
X_test = test_data.drop(columns=['Filename', 'class'])  # Drop 'Filename' and 'class' columns
y_test = test_data['class']  # 'class' column as the label

# Feature Scaling
scaler = StandardScaler()

# Remove constant or near-constant features
variance_filter = VarianceThreshold(threshold=1e-4)

# Feature selection
feature_selector = SelectKBest(score_func=f_classif, k='all')  # Use all features since we have only 6

# Define parameter grids for each model
param_grids = {
    "Random Forest": {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10, 20, 37],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 2]
    },
    "Support Vector Classifier": {
        'classifier__kernel': ['linear', 'rbf'],
        'classifier__C': [0.1, 1.0, 10],
        'classifier__gamma': ['scale', 'auto']
    },
    "Gradient Boosting": {
        'classifier__n_estimators': [100, 200],
        'classifier__learning_rate': [0.01, 0.1],
        'classifier__max_depth': [3, 5]
    },
    "K-Nearest Neighbors": {
        'classifier__n_neighbors': [3, 5, 7],
        'classifier__weights': ['uniform', 'distance']
    },
    "Logistic Regression": {
        'classifier__C': [0.01, 0.1, 1.0, 10],
        'classifier__solver': ['lbfgs', 'liblinear']
    },
    "Decision Tree": {
        'classifier__max_depth': [5, 10, None],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 2]
    }
}

# Define models
base_models = {
    "Random Forest": RandomForestClassifier(),
    "Support Vector Classifier": SVC(probability=True),
    "Gradient Boosting": GradientBoostingClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier()
}

# Train and evaluate each model
for model_name, model in base_models.items():
    print(f"\nTuning hyperparameters for {model_name}...")
    
    if model_name in param_grids:
        # Create pipeline
        pipeline = Pipeline([
            ('variance_filter', variance_filter),
            ('scaler', scaler),
            ('feature_selection', feature_selector),
            ('classifier', model)
        ])
        
        # Perform Grid Search for models with hyperparameter grids
        grid = GridSearchCV(pipeline, 
                            param_grids[model_name], 
                            cv=5, 
                            scoring='accuracy', 
                            n_jobs=-1)
        grid.fit(X_train, y_train)
        
        # Best parameters and cross-validation accuracy
        print(f"Best parameters for {model_name}: {grid.best_params_}")
        print(f"Best cross-validation accuracy for {model_name}: {grid.best_score_:.2f}")
        
        # Test set evaluation
        best_model = grid.best_estimator_
    else:
        # For models without hyperparameter tuning
        pipeline = Pipeline([
            ('variance_filter', variance_filter),
            ('scaler', scaler),
            ('feature_selection', feature_selector),
            ('classifier', model)
        ])
        pipeline.fit(X_train, y_train)
        best_model = pipeline
    
    # Predict on test set
    y_pred = best_model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    
    print(f"\nModel: {model_name}")
    print(f"Accuracy on test data: {accuracy:.2f}")
    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", matrix)
