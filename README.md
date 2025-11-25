# XGBoost

# Obesity Risk Classification with XGBoost  
Kaggle Playground Series – S4E2

This project builds a multi-class classifier to predict obesity levels using behavioral, dietary, and physiological features. The workflow includes EDA, feature engineering, XGBoost modeling, ROC analysis, and hyperparameter tuning.

---

## 🔧 Workflow Summary

### 1. Data Loading & Cleaning
- Combined Kaggle S4E2 dataset with the original *ObesityDataSet.csv*
- Removed duplicates
- Dropped `Height` (low correlation, inconsistent scaling)
- Target: 7-class obesity category (`NObeyesdad`)

### 2. Feature Engineering
- Ordinal encoding for: Gender, CAEC, FAVC, CALC, SMOKE, SCC, family history  
- One-hot encoding for: MTRANS  
- Numerical features standardized through exploration

### 3. Correlation Insights
Top correlated features with the target:
- **Weight (0.896)**
- **family_history_with_overweight (0.508)**
- **Age, CAEC, FCVC, CH2O, FAF**

### 4. XGBoost Model
Key parameters:
```python
learning_rate=0.01
n_estimators=1000
max_depth=5
min_child_weight=3
subsample=0.8
colsample_bytree=0.8
objective="multi:softprob"
eval_metric="mlogloss"
```





## Installation & Deployment

You can run all models (Logistic Regression, XGBoost, CatBoost, Random Forest) either in **Google Colab** or on a **local Python environment**.

---

### 1. Requirements

- Python 3.8+
- JupyterNotebook/vscode/Google Colab
- Python packages:
  - `numpy`, `pandas`
  - `scikit-learn`
  - `xgboost`
  - `catboost`
  - `matplotlib`, `seaborn`

The notebooks assume the Kaggle obesity dataset files are available:

- `train.csv`
- `test.csv`
- `ObesityDataSet.csv` (merged with the Kaggle data in some models)

Place these in a folder called `sample_data/` at the root of the repo, or update the `train_path` / `test_path` variables at the top of each notebook.


### 2. Deployment

1. Open Google Colab/vscode/JupyterNotebook
2. Paste the repository URL and open the notebook you want, e.g.:
   - `Logistic Regression.ipynb`
   - `xgboost-on-obesity-risk (1).ipynb`
   - `CatBoost_v1.ipynb`
   - `ECS_171_Random_forest (2).ipynb`
3. Upload the dataset files into a `sample_data/` folder in the Colab runtime:
   - In the file browser, create `/content/sample_data/` and upload `train.csv`, `test.csv`, and `ObesityDataSet.csv`.
4. Run all cells (**Runtime → Run all**).  
   Each notebook will:
   - Load and clean the data
   - Apply feature engineering (BMI, rounding, encoding)
   - Train the model with stratified cross-validation
   - Print metrics (accuracy, macro-F1, confusion matrix)
   - (For tree models) optionally save a Kaggle-style `submission.csv`
