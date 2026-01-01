# IMV: Information Model Vigor

<div align="center">

*A unified framework for measuring information content and feature importance in machine learning models*

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>

---

## Overview

**IMV (Information Model Vigor)** is a model-agnostic framework for quantifying how much information a model captures about its predictions. Unlike traditional metrics that focus on accuracy or error rates, IMV measures the **information content** using principles from information theory and game theory.

### Key Features

**Three Powerful Modules:**
- **Binary IMV**: Binary classification with SHAP value attribution
- **Multi-class IMV**: Multi-class problems (3+ classes) with confusion matrix analysis
- **Ablation IMV**: Deep learning ablation studies with GPU support

**Performance:**
- Parallel processing for traditional ML
- GPU acceleration for deep learning (CUDA + Apple Silicon MPS)
- Automatic device detection

**Visualizations:**
- Confusion matrix heatmaps
- Performance comparison plots
- IMV distribution analysis

**Research-Grade:**
- Reproducible results with seed management
- K-fold cross-validation
- Statistical stability across runs

---

## Installation

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/intermodelvigorish/imv_ml_package.git
cd imv_ml_package

# Install core dependencies
pip install -r requirements.txt
```

### Optional: Deep Learning Support

For ablation studies with transformers:

```bash
# GPU support (CUDA or Apple Silicon MPS)
pip install torch>=1.12.0 transformers>=4.20.0 datasets>=2.0.0

# Note: PyTorch 1.12+ required for Apple Silicon MPS support
```

### Development Installation

```bash
# Install in editable mode
pip install -e .

# Install with dev dependencies
pip install -e ".[dev]"
```

---

## Quick Start

### 1. Binary IMV: Binary Classification

Compute IMV-based feature importance for binary classification:

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from imv import BinaryIMV

# Load data
data = pd.read_csv('adult_income.csv')

# Create evaluator
evaluator = BinaryIMV(
    data=data,
    outcome_variable='income_>50K',
    optional_explanatory_variables=['age', 'education_years', 'hours_per_week'],
    model_creator=lambda: LogisticRegression(max_iter=1000),
    split_method='kfold',
    n_splits=5,
    prop_test=0.2,
    model_type='classification',
    random_seed=42
)

# Run evaluation
evaluator.run_evaluation()

# Visualize results
fig, ax = evaluator.evaluate_imvshapley(figsize=(12, 4))
```

**Output:**
```
Best explanatory variables' combination: ('age', 'education_years'), IMV: 0.234
SHAP-IMV value for variable age: 0.145
SHAP-IMV value for variable education_years: 0.112
SHAP-IMV value for variable hours_per_week: 0.068
```

### 2. Multi-class IMV: Multi-class Classification

Analyze multi-class classification problems:

```python
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from imv import MulticlassIMV

# Load multi-class data
data = pd.read_csv('nursery.csv')

# Create evaluator
evaluator = MulticlassIMV(
    data=data,
    outcome_variable='outcome',
    model_creator=lambda: GradientBoostingClassifier(n_estimators=100, random_state=42),
    n_splits=10,
    optional_explanatory_variables=['parents', 'has_nurs', 'form', 'children'],
    random_state=42
)

# Calculate IMV confusion matrix
imv_matrix = evaluator.calculate_imv_matrix()
print(imv_matrix)

# Visualize
fig, ax = evaluator.plot_imv_heatmap(imv_matrix, figsize=(8, 8))
```

**Interpretation:**
- Diagonal values are always 0 (a class compared to itself)
- Off-diagonal values show information gain when predicting one class vs another
- Higher values indicate better class separation

### 3. Ablation IMV: Deep Learning

Quantify the importance of model components in deep learning:

```python
import torch
from transformers import DistilBertForSequenceClassification
from imv import AblationIMV

# Initialize (automatically detects GPU)
ablator = AblationIMV(random_seed=42)
# Output: Using device: Apple Silicon GPU (MPS)

# Create model variants with different layer counts
model_6layer = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)
model_4layer = ablator.reduce_bert_layers(model_6layer, num_layers_to_keep=4)
model_2layer = ablator.reduce_bert_layers(model_6layer, num_layers_to_keep=2)

# Train each variant and compare
results = {}
for name, model in [('6-layer', model_6layer), ('4-layer', model_4layer), ('2-layer', model_2layer)]:
    result = ablator.train_and_evaluate(
        model=model,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        num_epochs=3,
        lr=2e-5,
        seed=42
    )
    results[name] = result['test_predictions']

# Compute IMV matrix
imv_matrix = ablator.calculate_imv_matrix(results)
print(imv_matrix)
```

---

## Project Structure

```
imv_ml_package/
├── imv/                        # Main package
│   ├── __init__.py
│   ├── core.py                 # Core IMV calculation
│   ├── binary.py               # BinaryIMV class
│   ├── multiclass.py           # MulticlassIMV class
│   ├── ablation.py             # AblationIMV class
│   └── utils.py                # Utility functions
├── paper_examples/                   # Example scripts
│   └── package_version/
│       ├── shap_imv/          # Binary classification examples
│       │   ├── shap_imv_titanic.py
│       │   ├── shap_imv_breastcancer.py
│       │   └── shap_imv_winequality.py
│       └── multi_imv/         # Multi-class examples
│           ├── multi_imv_nursery.py
│           ├── multi_imv_car_evaluation.py
│           ├── multi_imv_dry_bean.py
│           └── create_combined_figure.py
├── tests/                      # Test suite
│   ├── test_ablate_imv.py
│   ├── test_multi_imv.py
│   └── test_shap_imv.py
├── requirements.txt            # Dependencies
├── requirements-dev.txt        # Dev dependencies
├── setup.py                    # Package setup
└── README.md                   # This file
```

---

## Documentation

| Document | Description |
|----------|-------------|
| **[TECHNICAL.md](docs/TECHNICAL.md)** | Technical documentation and algorithms |
| **[Paper Examples](paper_examples/)** | Complete example scripts with real datasets to replicate paper results|
| **[Tests README](tests/README.md)** | Testing guide and test descriptions |

---

## Examples to replicate the paper results

### Binary Classification Examples

The `examples/package_version/shap_imv/` directory contains:

1. **Titanic Dataset** (`shap_imv_titanic.py`)
   - Binary survival prediction
   - 7 features: Class, Sex, Age, Age×Class, Alone, Fare, Embarked
   - Models: Logistic Regression, XGBoost, LightGBM

2. **Breast Cancer Dataset** (`shap_imv_breastcancer.py`)
   - Binary diagnosis (malignant/benign)
   - 9 tumor characteristics
   - Models: Logistic Regression, XGBoost, LightGBM

3. **Wine Quality Dataset** (`shap_imv_winequality.py`)
   - Binary quality classification (high/low)
   - 11 chemical properties
   - Models: Logistic Regression, XGBoost, LightGBM

### Multi-class Classification Examples

The `examples/package_version/multi_imv/` directory contains:

1. **Nursery Dataset** (`multi_imv_nursery.py`)
   - 3 classes: Not recommend, Priority, Special priority
   - 8 categorical features
   - Models: Logistic Regression, XGBoost, LightGBM

2. **Car Evaluation Dataset** (`multi_imv_car_evaluation.py`)
   - 4 classes: Unacceptable, Acceptable, Good, Very good
   - 6 categorical features
   - Models: Logistic Regression, XGBoost, LightGBM

3. **Dry Bean Dataset** (`multi_imv_dry_bean.py`)
   - 7 bean varieties
   - 16 continuous structural features
   - Models: Logistic Regression, XGBoost, LightGBM

4. **Combined Figure** (`create_combined_figure.py`)
   - Generates publication-quality combined visualization
   - Side-by-side IMV confusion matrices for all three datasets

### Running Examples

```bash
# Binary classification examples
cd examples/package_version/shap_imv
python shap_imv_titanic.py
python shap_imv_breastcancer.py
python shap_imv_winequality.py

# Multi-class examples
cd ../multi_imv
python multi_imv_nursery.py
python multi_imv_car_evaluation.py
python multi_imv_dry_bean.py

# Generate combined figure
python create_combined_figure.py
```

All scripts support caching:
- `--force`: Ignore cache and re-run analysis
- `--clear-cache`: Delete all cached results

---

## Testing

```bash
# Run all tests
cd tests
python test_shap_imv.py
python test_multi_imv.py
python test_ablate_imv.py

# Run specific test
python test_shap_imv.py --test basic
```

---

## API Reference

### BinaryIMV Class

```python
from imv import BinaryIMV

evaluator = BinaryIMV(
    data: pd.DataFrame,
    outcome_variable: str,
    optional_explanatory_variables: List[str],
    model_creator: Callable,
    split_method: str,  # 'kfold' or 'train_test_split'
    n_splits: int,
    prop_test: float,
    model_type: str,  # 'classification' or 'regression'
    random_seed: int = 42
)
```

**Key Methods:**
- `run_evaluation()`: Compute IMV for all feature combinations
- `evaluate_imvshapley(ax=None, figsize=(12, 4))`: Calculate and visualize SHAP-IMV values
- `calculate_imvshapley_value(variable)`: SHAP-IMV for one feature

### MulticlassIMV Class

```python
from imv import MulticlassIMV

evaluator = MulticlassIMV(
    data: pd.DataFrame,
    outcome_variable: str,
    model_creator: Callable,
    n_splits: int = 10,
    optional_explanatory_variables: Optional[List[str]] = None,
    random_state: Optional[int] = None
)
```

**Key Methods:**
- `calculate_imv_matrix()`: Pairwise IMV confusion matrix
- `plot_imv_heatmap(imv_matrix, ax=None, figsize=(8, 8))`: Heatmap visualization
- `calculate_performance_metrics()`: Accuracy, precision, recall, Brier score

### AblationIMV Class

```python
from imv import AblationIMV

ablator = AblationIMV(random_seed: int = 42)
```

**Key Methods:**
- `train_and_evaluate(model, train_dataloader, test_dataloader, ...)`: Train and evaluate model
- `calculate_imv_matrix(predictions_dict)`: Pairwise IMV for multiple models
- `reduce_bert_layers(model, num_layers_to_keep)`: Reduce transformer layers

---

## Performance

| Configuration | Combinations | Estimated Time | CPU Usage |
|--------------|--------------|----------------|-----------|
| 5 variables, 5-fold | 32 | 1-2 minutes | High |
| 10 variables, 10-fold | 1,024 | 3-8 minutes | Very High |
| 11 variables, 10-fold | 2,048 | 5-15 minutes | Very High |

*Performance scales with number of CPU cores (uses all cores by default)*

---

## Hardware Requirements

### Minimum Requirements
- **CPU:** Dual-core processor
- **RAM:** 4GB
- **Python:** 3.8+

### Recommended for Deep Learning
- **GPU:** 
  - NVIDIA GPU with 8GB+ VRAM (CUDA)
  - Apple M1/M2/M3 (8GB+ unified memory)
- **RAM:** 16GB+
- **Storage:** 10GB+ for model checkpoints


## Troubleshooting

### Slow Performance
- Reduce number of variables
- Reduce `n_splits` parameter
- Use `split_method='train_test_split'` instead of `'kfold'`

### Memory Issues
- Test with fewer variables first
- Close other applications
- Consider using a subset of data for testing

### Import Errors
```bash
# Install missing packages
pip install -r requirements.txt
```



## Contact

- **Issues:** [GitHub Issues](https://github.com/intermodelvigorish/imv_ml_package/issues)

---

<div align="center">

[Back to Top](#imv-information-model-vigor)

</div>
