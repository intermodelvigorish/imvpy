# SHAP-IMV Analysis Examples

This directory contains scripts to replicate the SHAP-IMV analysis from the paper using different datasets.

## Scripts

- `shap_imv_titanic.py` - Analysis on Titanic dataset
- `shap_imv_breastcancer.py` - Analysis on Breast Cancer Wisconsin dataset
- `shap_imv_winequality.py` - Analysis on Wine Quality dataset
- `shap_imv_adultincome.py` - Analysis on Adult Income dataset

## Features

### Automatic Result Caching

Each script automatically saves results for each model (Logistic Regression, XGBoost, LightGBM) after completion. This means:

- If the script is interrupted, you won't lose completed models
- Re-running the script will use cached results (much faster!)
- Results are saved in `results/` directory

### Usage

**Basic usage:**
```bash
python shap_imv_titanic.py
python shap_imv_breastcancer.py
python shap_imv_winequality.py
python shap_imv_adultincome.py
```

**Force re-run all models (ignore cache):**
```bash
python shap_imv_titanic.py --force
python shap_imv_breastcancer.py --force
python shap_imv_winequality.py --force
```

**Clear all cached results:**
```bash
python shap_imv_titanic.py --clear-cache
python shap_imv_breastcancer.py --clear-cache
python shap_imv_winequality.py --clear-cache
```

## Output

Each script generates:

1. **Results Files** (in `results/` directory):
   - `titanic_logistic_regression_results.pkl`
   - `titanic_xgboost_results.pkl`
   - `titanic_lightgbm_results.pkl`
   - `breast_cancer_logistic_regression_results.pkl`
   - `breast_cancer_xgboost_results.pkl`
   - `breast_cancer_lightgbm_results.pkl`
   - `wine_quality_logistic_regression_results.pkl`
   - `wine_quality_xgboost_results.pkl`
   - `wine_quality_lightgbm_results.pkl`
   - Contains SHAP-IMV values and performance metrics for each model

2. **Figures** (in `figures/` directory):
   - Comparison plots showing SHAP-IMV values across models
   - Bar charts replicating the paper's visualizations

3. **Console Output**:
   - Performance comparison table (accuracy, precision)
   - Detailed SHAP-IMV values for all features
   - Top 5 features for each model

## Requirements

- Python 3.7+
- scikit-learn
- numpy
- pandas
- matplotlib
- xgboost (optional, for XGBoost models)
- lightgbm (optional, for LightGBM models)

Install optional dependencies:
```bash
pip install xgboost lightgbm
```

## Titanic Dataset

The Titanic analysis uses the following features as described in the paper:

- **Class** (3-class): Ticket class (1st, 2nd, 3rd)
- **Sex** (binary): Passenger gender
- **Age** (continuous): Passenger age
- **Age*Class** (continuous): Interaction term between Age and Class
- **Alone** (binary): Whether passenger was alone
- **Fare** (continuous): Ticket fare
- **Embarked** (3-class): Embarkation port (C, Q, S)

Missing values are handled with:
- Mean imputation for continuous variables (Age, Fare)
- Mode imputation for categorical variables (Embarked)

## Breast Cancer Dataset

The Breast Cancer Wisconsin (Original) analysis uses 9 features measuring tumor characteristics:

- **Clump Thickness**: Thickness of clump
- **Uniformity of Cell Size**: Uniformity of cell size
- **Uniformity of Cell Shape**: Uniformity of cell shape
- **Marginal Adhesion**: Marginal adhesion
- **Single Epithelial Cell Size**: Single epithelial cell size
- **Bare Nuclei**: Bare nuclei
- **Bland Chromatin**: Bland chromatin
- **Normal Nucleoli**: Normal nucleoli
- **Mitoses**: Mitoses

Missing values (< 1%) are handled with mean imputation for all numerical features.

## Wine Quality Dataset

The Wine Quality analysis uses 11 chemical features:

- **Fixed Acidity**: Fixed acidity
- **Volatile Acidity**: Volatile acidity
- **Citric Acid**: Citric acid
- **Residual Sugar**: Residual sugar
- **Chlorides**: Chlorides
- **Free Sulfur Dioxide**: Free sulfur dioxide
- **Total Sulfur Dioxide**: Total sulfur dioxide
- **Density**: Density
- **pH**: pH value
- **Sulphates**: Sulphates
- **Alcohol**: Alcohol content

Target is dichotomized: 0 for quality ≤ 5, 1 for quality > 5. No missing values.

## Notes

- Each model runs with 3 different random seeds (42, 123, 456) as mentioned in the paper
- Results are averaged across seeds
- 10-fold cross-validation is used for evaluation
- The analysis can take several minutes per model depending on your hardware
