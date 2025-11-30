# SHAP-IMV Tests

This directory contains tests for the SHAP-IMV package that replicate the functionality from the `SHAP_IMV_AdultIncome (2).ipynb` notebook.

## Setup

First, install the required dependencies:

```bash
# Core dependencies
pip install numpy pandas scipy scikit-learn matplotlib seaborn joblib tqdm

# Dataset dependency
pip install ucimlrepo

# Optional: for better progress bars with joblib
pip install tqdm-joblib
```

## Running Tests

The test file `test_shap_imv.py` contains several test functions that replicate different aspects of the notebook:

### Quick Test (Recommended for first run)
```bash
cd tests
python test_shap_imv.py
```

This runs a quick test with a subset of variables (5 variables instead of 11) and uses 5-fold cross-validation.

### Full Test (Replicates Notebook Exactly)
```bash
cd tests
python test_shap_imv.py --full
```

This runs the complete test with all 11 variables and 10-fold cross-validation, exactly as in the notebook. **Warning: This may take several minutes to complete.**

### Run Specific Tests
```bash
# Test basic functionality only
python test_shap_imv.py --test basic

# Test full functionality (all variables)
python test_shap_imv.py --test full

# Test with train-test split instead of k-fold
python test_shap_imv.py --test split

# Test core calculation methods only
python test_shap_imv.py --test calc
```

## Test Descriptions

### 1. Core IMV Calculations (`test_imv_calculations`)
Tests the fundamental IMV calculation methods:
- `ll()` - Log-likelihood calculation
- `get_w()` - Weight calculation using optimization
- `calculate_imv()` - IMV score calculation
- `calculate_weight()` - Shapley weight calculation

### 2. Basic SHAP-IMV Test (`test_shap_imv_basic`)
- Uses 5 variables: age, education, marital-status, sex, hours-per-week
- 5-fold cross-validation
- Generates SHAP-IMV bar plot
- Quick execution (~1-2 minutes)

### 3. Full SHAP-IMV Test (`test_shap_imv_full`)
- Uses all 11 variables from the notebook
- 10-fold cross-validation
- Generates both SHAP-IMV bar plot and violin plot
- Longer execution time (~5-15 minutes depending on CPU)
- Evaluates 2^11 = 2048 variable combinations

### 4. Train-Test Split Test (`test_shap_imv_train_test_split`)
- Tests alternative splitting method
- Uses train-test split instead of k-fold cross-validation
- 4 variables for quick execution

## Expected Outputs

The tests will generate the following files:

1. **test_shap_imv_basic.png** - SHAP-IMV values for subset of variables
2. **test_shap_imv_full.png** - SHAP-IMV values for all variables (only with --full)
3. **test_single_var_violin.png** - Violin plot showing distribution of single variable IMV scores

## Comparison with Notebook

The test suite is designed to replicate the notebook's functionality:

| Notebook Cell | Test Function | Description |
|--------------|---------------|-------------|
| Cell 1 | `load_and_prepare_adult_income_data()` | Data loading and preprocessing |
| Cell 2 | `IMVEvaluator` class | Main class implementation |
| Cell 7 | `test_shap_imv_full()` | Logistic Regression SHAP-IMV |

## Key Features Tested

✅ Data loading and preprocessing from UCI ML Repository  
✅ Label encoding for categorical variables  
✅ IMV calculation with k-fold cross-validation  
✅ Parallel processing with joblib  
✅ SHAP value calculation for IMV  
✅ Visualization generation (bar plots and violin plots)  
✅ Both classification and regression model types  
✅ Both k-fold and train-test split methods  

## Troubleshooting

### ImportError: No module named 'ucimlrepo'
Install the package:
```bash
pip install ucimlrepo
```

### ImportError: No module named 'tqdm_joblib'
This is optional. Install it for better progress bars:
```bash
pip install tqdm-joblib
```
Or the code will use a fallback implementation.

### Tests running slowly
- Use the default quick mode instead of --full
- Reduce the number of variables being tested
- Reduce n_splits (e.g., from 10 to 5)

### Memory issues
If you encounter memory issues with the full test:
- Close other applications
- Try the basic test first
- Consider using fewer variables

## Development Notes

The `shap_imv.py` module includes improvements over the notebook:
1. Better error handling
2. Fallback for optional dependencies
3. More flexible visualization options
4. Consistent API across different split methods
