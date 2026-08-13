## Running the Project

The repository currently contains the classical machine-learning and image-processing code, extracted GLCM feature files, experiment notebooks, and saved YOLOv8/X-Net result images. 

### Prerequisites

- Python 3.10 or 3.11
- Git
- Jupyter Notebook for the `.ipynb` experiments

### 1. Clone the repository

```bash
git clone https://github.com/MANSIBAGOHIL/metal-casting-defect-detection.git
cd metal-casting-defect-detection
```

### 2. Create and activate a virtual environment

**Windows PowerShell**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS or Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Run the classical model comparison

The repository includes the prepared feature files:

```text
models-and-outputs/glcm_features_train.csv
models-and-outputs/glcm_features_test.csv
```

Before running the comparison, open:

```text
codes/codes_Balanced_dataset/model_training.py
```

Replace its two original `D:\Capston Project\...` CSV paths with repository-relative paths. A portable replacement is:

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
train_data = pd.read_csv(ROOT / "models-and-outputs" / "glcm_features_train.csv")
test_data = pd.read_csv(ROOT / "models-and-outputs" / "glcm_features_test.csv")
```

Remove or replace the two existing `pd.read_csv(...)` lines, then run from the repository root:

```bash
python "codes/codes_Balanced_dataset/model_training.py"
```

This trains and evaluates:

- Logistic Regression
- Decision Tree
- K-Nearest Neighbors
- Random Forest
- Support Vector Classifier
- Gaussian Naive Bayes

The script prints accuracy, a classification report, and a confusion matrix for each model.

### Run hyperparameter tuning

Apply the same two relative CSV paths to:

```text
codes/codes_Balanced_dataset/model_training_hyperparameters.py
```

Then run:

```bash
python "codes/codes_Balanced_dataset/model_training_hyperparameters.py"
```

This performs grid search for Random Forest, SVC, and Gradient Boosting. It may take longer because it evaluates multiple parameter combinations using five-fold cross-validation.

### Explore the notebooks

Start Jupyter from the repository root:

```bash
jupyter notebook
```

Useful notebooks include:

- `codes/SVM_RF_LR.ipynb` - Logistic Regression, Random Forest, and SVM experiments
- `codes/codes_Balanced_dataset/model_training_5.ipynb` - six-model comparison
- `codes/codes_Balanced_dataset/model_training_parameters.ipynb` - model parameter experiments
- `codes/rotation/model.ipynb` - evaluation after rotation-based augmentation
- `codes/rotation/rot_PCA.ipynb` - PCA-based experiments
- `codes/polynomial_regression.ipynb` - polynomial regression experiment

Several notebooks contain absolute paths from the original Windows computer. Replace each `D:\Capston Project\...` path with the corresponding dataset or CSV location on your computer before executing those cells.

### Extract features from raw images

Raw industrial images are not included in the repository. If you have authorized access to the dataset, arrange it as:

```text
dataset/
├── Cutting_Marks/
├── Hot_tears_cracks/
├── Inclusion/
├── Porosity/
├── Scabs/
├── Shrink/
├── Surface_Roughness/
├── Veining/
└── Wrinkles_Folds_Coldshuts/
```

Update `dataset_path` in `codes/Feature_Extraction.py` to that directory, then run:

```bash
python codes/Feature_Extraction.py
```

The script extracts HOG, LBP, GLCM, Haralick, and entropy features, creates feature-distribution plots, and saves the extracted feature dictionary as `metal_defect_features.npy`.

### YOLOv8 and X-Net results

The saved evaluation images can be viewed under:

```text
images-and-results/yolov8-results/
images-and-results/xnet_results/
```