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
