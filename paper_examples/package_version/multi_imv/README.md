# Multi-IMV Analysis Examples

This directory contains scripts for Multi-class IMV (Multi-IMV) analysis on multi-class classification datasets.

## Scripts

- `multi_imv_nursery.py` - Multi-class analysis on Nursery dataset (3 classes)
- `multi_imv_car_evaluation.py` - Multi-class analysis on Car Evaluation dataset (4 classes)
- `multi_imv_dry_bean.py` - Multi-class analysis on Dry Bean dataset (7 classes)
- `create_combined_figure.py` - Generate combined figure (Figure 3) showing all datasets

## Overview

Multi-IMV extends the IMV framework to handle multi-class classification problems (3+ classes). It provides:

1. **Pairwise IMV Confusion Matrix**: Measures information gain between each pair of classes
2. **One-vs-All IMV**: Measures information gain for each class vs all others
3. **Performance Metrics**: Accuracy, Precision, Recall, and Brier score

## Features

### Automatic Result Caching

Each script automatically saves results for each model (Logistic Regression, XGBoost, LightGBM) after completion:

- If interrupted, completed models are preserved
- Re-running uses cached results (much faster!)
- Results saved in `results/` directory

### Usage

**Basic usage:**
```bash
python multi_imv_nursery.py
python multi_imv_car_evaluation.py
python multi_imv_dry_bean.py
```

**Create combined figure (Figure 3 from paper):**
```bash
# First run all three analyses, then:
python create_combined_figure.py
```

**Force re-run all models (ignore cache):**
```bash
python multi_imv_nursery.py --force
python multi_imv_car_evaluation.py --force
python multi_imv_dry_bean.py --force
```

**Clear all cached results:**
```bash
python multi_imv_nursery.py --clear-cache
python multi_imv_car_evaluation.py --clear-cache
python multi_imv_dry_bean.py --clear-cache
```

## Output

Each script generates:

1. **Results Files** (in `results/` directory):
   - `nursery_logistic_regression_results.pkl`
   - `nursery_xgboost_results.pkl`
   - `nursery_lightgbm_results.pkl`
   - `car_evaluation_logistic_regression_results.pkl`
   - `car_evaluation_xgboost_results.pkl`
   - `car_evaluation_lightgbm_results.pkl`
   - `dry_bean_logistic_regression_results.pkl`
   - `dry_bean_xgboost_results.pkl`
   - `dry_bean_lightgbm_results.pkl`
   - Contains Multi-IMV matrices and performance metrics

2. **Figures** (in `figures/` directory):
   - Individual IMV confusion matrix heatmaps for each model and dataset
   - `multi_imv_combined_xgboost.png` - Combined figure (Figure 3 style)
   - `multi_imv_combined_xgboost.pdf` - Publication-quality PDF version
   - Replicates Figure 3 style from the paper

3. **Console Output**:
   - Performance table with Multi-IMV, Brier, Accuracy, Precision, Recall
   - Detailed metrics summary with mean ± std
   - Class-specific IMV values

## Datasets

### Nursery Dataset

The Nursery analysis uses:

- **Instances**: 12,630 (after filtering from 13,000)
  - 4,320 "Not recommend"
  - 4,266 "Priority acceptance"
  - 4,044 "Special priority"

- **Features**: 8 categorical variables
  - Parents' occupation
  - Nursery characteristics
  - Family structure
  - Financial standing
  - Social status
  - Health status

- **Target**: 3-class classification
  - Originally 5 classes, filtered to 3 (removed 'recommend' and 'very_recommend' due to limited instances)

- **Missing Values**: None

### Car Evaluation Dataset

The Car Evaluation analysis uses:

- **Instances**: 1,728
  - 1,210 "Unacceptable"
  - 384 "Acceptable"
  - 69 "Good"
  - 65 "Very good"

- **Features**: 6 categorical variables
  - Buying price
  - Maintenance cost
  - Number of doors
  - Capacity (persons)
  - Luggage boot size
  - Safety rating

- **Target**: 4-class classification

- **Missing Values**: None

### Dry Bean Dataset

The Dry Bean analysis uses:

- **Instances**: 13,611
  - 2,027 Seker
  - 1,322 Barbunya
  - 522 Bombay
  - 1,630 Cali
  - 3,546 Dermosan
  - 1,928 Horoz
  - 2,636 Sira

- **Features**: 16 continuous variables
  - Area, Perimeter, Major/Minor Axis Length
  - Aspect Ratio, Eccentricity
  - Convex Area, Extent
  - Solidity, Roundness
  - Compactness, ShapeFactor1-4

- **Target**: 7-class classification (bean varieties)

- **Missing Values**: None

## Methodology

- **Cross-validation**: 10-fold
- **Random Seeds**: 3 seeds (42, 43, 44) averaged across folds
  - Total: 30 runs per model (3 seeds × 10 folds)
- **Models**: Multinomial Logistic Regression, XGBoost, LightGBM
- **Metrics**:
  - Multi-IMV: Pairwise information gain matrix
  - Brier Score: Probability calibration metric
  - Accuracy: Overall classification accuracy
  - Precision: Macro-averaged precision
  - Recall: Macro-averaged recall

## Multi-IMV Confusion Matrix

The Multi-IMV confusion matrix shows information gain for discriminating between each pair of classes:

- **Element (i,j)**: IMV for distinguishing class i from class j
- **Diagonal**: Always 0 (no self-discrimination)
- **Interpretation**: Higher values indicate better class separation
- **Asymmetric**: Generally IMV(i,j) ≠ IMV(j,i)

Example:
```
              Not recommend  Priority  Special
Not recommend     0.000      0.987     0.942
Priority          0.987      0.000     0.946
Special           0.942      0.946     0.000
```

## Requirements

- Python 3.7+
- numpy, pandas, scikit-learn, matplotlib, seaborn
- ucimlrepo (for dataset loading)
- xgboost (optional)
- lightgbm (optional)
- imv package (local)

Install optional dependencies:
```bash
pip install xgboost lightgbm ucimlrepo
```

## Notes

- Analysis takes ~15-45 minutes per model depending on hardware
- Progress automatically saved after each model
- Results averaged across multiple seeds for robustness
- Heatmap visualizations match paper's style (Figure 3)
