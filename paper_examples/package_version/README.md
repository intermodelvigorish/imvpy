# SHAP-IMV Adult Income Multi-Model Comparison

This example compares three machine learning models (Logistic Regression, XGBoost, and LightGBM) using SHAP-IMV analysis on the Adult Income dataset.

## Features

- **Multi-Model Comparison**: Trains and evaluates 3 different models
- **Multiple Seeds**: Runs each model with 3 different random seeds (42, 43, 44) for robust results
- **10-Fold Cross-Validation**: Uses k-fold CV for reliable estimates
- **SHAP-IMV Analysis**: Calculates Shapley values using the IMV framework
- **Automated Visualizations**: Generates publication-quality figures
- **Performance Tables**: Creates comparison tables with accuracy and precision

## Requirements

```bash
pip install ucimlrepo xgboost lightgbm scikit-learn pandas numpy matplotlib seaborn
```

Or install the core requirements plus optional ML libraries:
```bash
pip install -r ../../requirements.txt
pip install xgboost lightgbm
```

## Usage

### Quick Run (with all models)

```bash
cd examples/package_version
python shap_imv_adultincome.py
```

### Expected Output

The script will:

1. **Load/Download Data**: Fetches Adult Income dataset from UCI (or loads cached version)
2. **Train Models**: For each of LR, XGBoost, LightGBM:
   - Trains with seeds 42, 43, 44
   - Performs 10-fold cross-validation
   - Calculates SHAP-IMV values for all features
3. **Generate Outputs**:
   - `figures/shap_imv_model_comparison.png` - Bar plot comparing top 5 features across models
   - `figures/performance_comparison.csv` - Performance metrics table

### Output Files

```
examples/package_version/
├── data/
│   └── adult_income_processed.csv        # Cached processed dataset
├── figures/
│   ├── shap_imv_model_comparison.png    # Main comparison plot
│   └── performance_comparison.csv        # Performance table
└── shap_imv_adultincome.py              # This script
```

## Understanding the Results

### SHAP-IMV Bar Plot

Shows the top 5 most important features for each model:
- **Feature Selection**: Top 5 features are selected based on AVERAGED SHAP-IMV values across all 3 seeds
- **Bar height**: Mean SHAP-IMV value (averaged across seeds 42, 43, 44)
- **Error bars**: Standard deviation across the 3 seeds
- **Color gradient**: Darker green = more important (relative ranking)
- **Important**: Features are NOT selected per-seed then averaged; rather, all seeds are averaged first, THEN top 5 are selected

### Performance Table

Compares models on:
- **Accuracy**: Overall classification accuracy (%)
- **Precision**: Precision for positive class (%)
- Values are averaged across the 3 seeds

## Customization

### Change Number of Seeds

Edit the `seeds` list in `main()`:
```python
seeds = [42, 43, 44, 45, 46]  # Use 5 seeds instead of 3
```

### Modify Cross-Validation Folds

In `run_shap_imv_analysis()`:
```python
n_splits=10,  # Change to 5 for faster runs
```

### Select Different Features

Modify `all_variables` in `main()`:
```python
all_variables = ['age', 'education', 'sex', 'hours-per-week']  # Subset
```

### Change Top N Features in Plot

In `main()`:
```python
plot_shap_imv_comparison(aggregated_results, top_n=8)  # Show top 8
```

## Runtime

Approximate runtime per model (depends on hardware):
- **Logistic Regression**: ~5-10 minutes per seed
- **XGBoost**: ~15-20 minutes per seed  
- **LightGBM**: ~10-15 minutes per seed

**Total runtime**: ~1-2 hours for all 3 models with 3 seeds each

To speed up for testing:
- Reduce `n_splits` to 5
- Use fewer variables
- Test with single seed first

## Troubleshooting

### "XGBoost not available"
```bash
pip install xgboost
```

### "LightGBM not available"
```bash
pip install lightgbm
```

### "ucimlrepo not found"
```bash
pip install ucimlrepo
```

### Memory Issues
- Reduce number of features
- Use smaller `n_splits` (e.g., 5 instead of 10)
- Run models sequentially instead of all at once

## Notes

- The script automatically detects available models (LR always available, XGB/LGBM optional)
- Results are cached in `data/` folder for faster subsequent runs
- Figures are saved in high resolution (300 DPI) suitable for publication
- The performance metrics are calculated from the cross-validation folds
