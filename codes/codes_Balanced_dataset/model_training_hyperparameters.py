import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.pipeline import Pipeline

# Load training and testing datasets
train_data = pd.read_csv(r"D:\Capston Project\glcm_features_train.csv")
test_data = pd.read_csv(r"D:\Capston Project\glcm_features_test.csv")

# Split features and labels
X_train = train_data.drop(columns=['Filename', 'class'])  # Drop 'Filename' and 'class' columns
y_train = train_data['class']  # 'class' column as the label
X_test = test_data.drop(columns=['Filename', 'class'])  # Drop 'Filename' and 'class' columns
y_test = test_data['class']  # 'class' column as the label

# Feature Scaling
scaler = StandardScaler()

# Adjust k in SelectKBest to match the number of features or use 'all'
feature_selector = SelectKBest(score_func=f_classif, k='all')  # Use all features since we have only 6

# Correct Parameter Grids
param_grids = {
    "Random Forest": {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [10, 20, None],
        'classifier__min_samples_split': [2, 10],
        'classifier__min_samples_leaf': [1, 5],
        'classifier__bootstrap': [True, False]
    },
    "Support Vector Classifier": {
        'classifier__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'classifier__C': [0.1, 1.0, 10],
        'classifier__gamma': ['scale', 'auto'],
        'classifier__degree': [2, 3, 4]  # Polynomial degree for poly kernel
    },
    "Gradient Boosting": {
        'classifier__n_estimators': [100, 200],
        'classifier__learning_rate': [0.01, 0.1, 0.2],
        'classifier__max_depth': [3, 5, 7]
    }
}

# Base Models
base_models = {
    "Random Forest": RandomForestClassifier(),
    "Support Vector Classifier": SVC(probability=True),
    "Gradient Boosting": GradientBoostingClassifier()
}

# Pipeline and Grid Search
for model_name, model in base_models.items():
    print(f"\nTuning hyperparameters for {model_name}...")
    
    if model_name in param_grids:
        # Create pipeline
        pipeline = Pipeline([
            ('scaler', scaler),
            ('feature_selection', feature_selector),
            ('classifier', model)
        ])
        
        # Perform Grid Search
        grid = GridSearchCV(pipeline, 
                            param_grids[model_name], 
                            cv=5, 
                            scoring='accuracy', 
                            n_jobs=-1)
        grid.fit(X_train, y_train)
        
        # Best parameters and accuracy
        print(f"Best parameters for {model_name}: {grid.best_params_}")
        print(f"Best cross-validation accuracy for {model_name}: {grid.best_score_:.2f}")
        
        # Test set evaluation
        y_pred = grid.best_estimator_.predict(X_test)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    
    print(f"\nModel: {model_name}")
    print(f"Accuracy on test data: {accuracy:.2f}")
    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", matrix)
