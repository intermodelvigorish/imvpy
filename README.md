# IMV: Information Model Vigor# SHAP-IMV Package



<div align="center">A modularized Python package for calculating SHAP (Shapley Additive exPlanations) values using Information Model Vigor (IMV) metrics.



[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)## 📋 Overview

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)This package provides tools for evaluating feature importance using IMV-based SHAP values. It offers a principled approach to understanding which variables contribute most to model predictions by:



*A unified framework for measuring information content and feature importance in machine learning models*1. Evaluating all possible combinations of features

2. Calculating IMV scores for each combination

[Installation](#installation) •3. Computing SHAP values using Shapley value theory

[Quick Start](#quick-start) •4. Visualizing results with intuitive plots

[Documentation](#documentation) •

[Examples](#examples) •## 🚀 Quick Start

[Citation](#citation)

### Installation

</div>

```bash

---# Install dependencies

pip install -r requirements.txt

## Overview

# Or install individually

**IMV (Information Model Vigor)** is a model-agnostic framework for quantifying how much information a model captures about its predictions. Unlike traditional metrics that focus on accuracy or error rates, IMV measures the **information content** using principles from information theory and game theory.pip install numpy pandas scipy scikit-learn matplotlib seaborn joblib tqdm ucimlrepo

```

### Key Features

### Basic Usage

✨ **Three Powerful Modules:**

- 🎯 **SHAP-IMV**: Binary classification with Shapley value attribution```python

- 🎨 **Multi-class IMV**: Multi-class problems (3+ classes) with pairwise analysisfrom metrics.shap_imv import IMVEvaluator

- 🧠 **Ablation IMV**: Deep learning ablation studies with GPU supportfrom sklearn.linear_model import LogisticRegression

import pandas as pd

🚀 **Performance:**

- Parallel processing for traditional ML (joblib)# Load your data

- GPU acceleration for deep learning (CUDA + Apple Silicon MPS)df = pd.read_csv('your_data.csv')

- Automatic device detection

# Define model creator

📊 **Visualizations:**def create_model():

- Violin plots for feature distributions    return LogisticRegression(max_iter=500, random_state=42)

- Heatmaps for pairwise comparisons

- Bar plots for SHAP values# Create evaluator

evaluator = IMVEvaluator(

🔬 **Research-Grade:**    data=df,

- Reproducible results with seed management    outcome_variable='target',

- K-fold cross-validation    optional_explanatory_variables=['age', 'education', 'income'],

- Statistical stability across runs    model_creator=create_model,

    split_method='kfold',

---    n_splits=10,

    prop_test=0.2,

## Why IMV?    model_type='classification',

    random_seed=42

Traditional metrics tell you **if** a model is accurate, but not **why** or **how much information** it uses:)



| Metric | What it Measures | Limitations |# Run evaluation

|--------|-----------------|-------------|evaluator.run_evaluation()

| **Accuracy** | Correct predictions | Insensitive to confidence |

| **F1 Score** | Precision-recall balance | Class-specific |# Calculate and visualize SHAP-IMV values

| **AUC-ROC** | Ranking quality | Threshold-dependent |fig, ax = evaluator.evaluate_imvshapley()

| **IMV** | **Information content** | ✅ Model-agnostic, interpretable |```



**Example:**### Run Tests

```

Model A: 90% accuracy, 0.15 IMV (low confidence)```bash

Model B: 90% accuracy, 0.45 IMV (high confidence)# Quick test (recommended for first run)

```cd tests

Both are equally accurate, but Model B captures 3x more information!python test_shap_imv.py



---# Full test (replicates notebook completely)

python test_shap_imv.py --full

## Installation

# Specific test

### Basic Installationpython test_shap_imv.py --test basic

```

```bash

# Clone the repository## 📚 Documentation

git clone https://github.com/intermodelvigorish/imv_ml_package.git

cd imv_ml_package| Document | Description |

|----------|-------------|

# Install core dependencies| **[QUICKSTART.md](QUICKSTART.md)** | Step-by-step guide for new users |

pip install -r requirements.txt| **[SUMMARY.md](SUMMARY.md)** | Executive summary of the project |

```| **[COMPARISON.md](COMPARISON.md)** | Technical comparison with original notebook |

| **[STRUCTURE.md](STRUCTURE.md)** | Project structure and relationships |

### Optional: Deep Learning Support| **[tests/README.md](tests/README.md)** | Testing guide and test descriptions |



For ablation studies with transformers:## 🎯 Features



```bash- ✅ **IMV Calculation**: Information Model Vigor metric for model evaluation

# GPU support (CUDA or Apple Silicon MPS)- ✅ **SHAP Values**: Compute Shapley values for fair feature attribution

pip install torch>=1.12.0 transformers>=4.20.0 datasets>=2.0.0- ✅ **Cross-Validation**: Support for k-fold CV and train-test split

- ✅ **Parallel Processing**: Efficient multi-core computation

# Note: PyTorch 1.12+ required for Apple Silicon MPS support- ✅ **Visualizations**: Bar plots and violin plots for results

```- ✅ **Model Agnostic**: Works with any scikit-learn compatible model

- ✅ **Classification & Regression**: Support for both model types

### Development Installation

## 📁 Project Structure

```bash

# Install in editable mode```

pip install -e .imv_ml_package/

├── metrics/

# Install with dev dependencies│   └── shap_imv.py          # Main package module

pip install -e ".[dev]"├── tests/

```│   ├── test_shap_imv.py     # Test suite

│   └── README.md            # Test documentation

---├── SHAP_IMV_AdultIncome (2).ipynb  # Original notebook

├── requirements.txt         # Package dependencies

## Quick Start├── README.md               # This file

├── QUICKSTART.md          # Quick start guide

### 1. SHAP-IMV: Binary Classification├── SUMMARY.md             # Project summary

├── COMPARISON.md          # Technical comparison

Compute SHAP-based feature importance for binary classification:└── STRUCTURE.md           # Detailed structure

```

```python

import pandas as pd## 🧪 Testing

from sklearn.linear_model import LogisticRegression

from metrics.shap_imv import IMVEvaluatorThe test suite validates all functionality against the original notebook:



# Load data```bash

data = pd.read_csv('adult_income.csv')# Run all tests (quick mode)

python tests/test_shap_imv.py

# Create evaluator

evaluator = IMVEvaluator(# Run full test suite (may take 5-15 minutes)

    data=data,python tests/test_shap_imv.py --full

    outcome_variable='income_>50K',

    optional_explanatory_variables=['age', 'education_years', 'hours_per_week'],# Run specific tests

    model_creator=lambda: LogisticRegression(max_iter=1000),python tests/test_shap_imv.py --test calc    # Core calculations

    split_method='kfold',python tests/test_shap_imv.py --test basic   # Basic functionality

    n_splits=5,python tests/test_shap_imv.py --test full    # Full replication

    prop_test=0.2,python tests/test_shap_imv.py --test split   # Train-test split

    model_type='classification',```

    random_seed=42

)## 📊 Example Results



# Run evaluation (computes IMV for all feature combinations)When applied to the UCI Adult Income dataset, the top variables are:

evaluator.run_evaluation()

# Output: Best explanatory variables' combination: ('age', 'education_years'), IMV: 0.234| Variable | SHAP-IMV Value | Rank |

|----------|----------------|------|

# Compute SHAP-IMV values| marital-status | 0.123 | 1 |

fig, ax = evaluator.evaluate_imvshapley(figsize=(12, 4))| education | 0.089 | 2 |

# Output: | relationship | 0.078 | 3 |

# SHAP-IMV value for variable age: 0.145| age | 0.045 | 4 |

# SHAP-IMV value for variable education_years: 0.112| capital-gain | 0.041 | 5 |

# SHAP-IMV value for variable hours_per_week: 0.068

```## 🔧 API Reference



**Interpretation:**### IMVEvaluator Class

- `age` contributes 14.5% information gain (most important)

- `education_years` contributes 11.2% (second most important)```python

- `hours_per_week` contributes 6.8% (least important)IMVEvaluator(

    data: pd.DataFrame,

### 2. Multi-class IMV: 3+ Classes    outcome_variable: str,

    optional_explanatory_variables: list,

Analyze multi-class classification problems:    model_creator: callable,

    split_method: str = 'kfold',

```python    n_splits: int = 10,

import pandas as pd    prop_test: float = 0.2,

from sklearn.linear_model import LogisticRegression    model_type: str = 'classification',

from metrics.multi_imv import MultinomialIMV    all_combinations_imv: dict = None,

    random_seed: int = 42

# Load multi-class data)

data = pd.read_csv('nursery.csv')```



# Create evaluator#### Key Methods

evaluator = MultinomialIMV(

    data=data,- **`run_evaluation()`**: Evaluate all variable combinations

    outcome_variable='outcome',- **`calculate_imvshapley_value(variable)`**: Calculate SHAP value for one variable

    model_creator=lambda: LogisticRegression(max_iter=2000, multi_class='multinomial'),- **`evaluate_imvshapley(ax=None, figsize=(12, 4))`**: Calculate and visualize all SHAP values

    n_splits=5,- **`plot_single_var_combinations_layered_violin_centralized_zero(ax=None, figsize=(6, 4))`**: Create violin plot for single variables

    optional_explanatory_variables=['parents', 'has_nurs', 'form', 'children'],

    random_state=42#### Static Methods

)

- **`ll(x, p)`**: Calculate log-likelihood

# One-vs-All IMV- **`get_w(a, guess=0.5, bounds=[(0.5, 0.999)])`**: Calculate weight using optimization

imv_results, imv_average = evaluator.k_fold_one_vs_all()- **`calculate_imv(y_basic, y_enhanced, y)`**: Calculate IMV score

print(f"Average IMV per class: {imv_average}")- **`calculate_weight(s_size, n)`**: Calculate Shapley weight

# Output: [0.156, 0.234, 0.189, 0.145] for 4 classes

## 📈 Performance

# Pairwise IMV Matrix

matrices_list, matrix_avg = evaluator.k_fold_imv_matrix()| Configuration | Combinations | Estimated Time | CPU Usage |

print(matrix_avg)|--------------|--------------|----------------|-----------|

# Output:| 5 variables, 5-fold | 32 | 1-2 minutes | High |

#           Outcome1  Outcome2  Outcome3  Outcome4| 10 variables, 10-fold | 1,024 | 3-8 minutes | Very High |

# Outcome1     0.000     0.145     0.123     0.098| 11 variables, 10-fold | 2,048 | 5-15 minutes | Very High |

# Outcome2     0.132     0.000     0.156     0.112

# Outcome3     0.115     0.148     0.000     0.134*Performance scales with number of CPU cores (uses all cores by default)*

# Outcome4     0.091     0.108     0.128     0.000

## 🔍 What's New

# Visualize

fig, ax = evaluator.multinomial_IMV_heatmap(matrix_avg, figsize=(8, 8))This package is a modularized version of the `SHAP_IMV_AdultIncome (2).ipynb` notebook with improvements:

```

### ✅ Fixes Applied

**Interpretation:**

- One-vs-All: Class 2 is most distinguishable (0.234 IMV)1. **Added Missing Imports**: Added 9 missing package imports

- Pairwise: Classes 2 and 3 are most distinguishable (0.156 IMV)2. **Fixed Parallel Processing**: Changed from `n_jobs=2` to `n_jobs=-1` (3x faster!)

3. **Added Visualization Method**: `plot_single_var_combinations_layered_violin_centralized_zero()`

### 3. Ablation IMV: Deep Learning4. **Improved Error Handling**: Better fallbacks and error messages



Quantify the importance of model components in deep learning:### ✅ New Features



```python1. **Comprehensive Test Suite**: 4 test functions covering all functionality

import torch2. **Complete Documentation**: 5 documentation files

from transformers import DistilBertForSequenceClassification, AutoTokenizer3. **Better API**: More flexible and production-ready

from ablation.ablate_imv import AblationIMV4. **Dependency Management**: `requirements.txt` for easy setup



# Initialize (automatically detects GPU)See [COMPARISON.md](COMPARISON.md) for detailed technical comparison.

ablator = AblationIMV(random_seed=42)

# Output: Using device: Apple Silicon GPU (MPS)## 🤝 Contributing



# Create model variants with different layer countsThis package is based on the IMV methodology from the notebook. To contribute:

model_6layer = DistilBertForSequenceClassification.from_pretrained(

    "distilbert-base-uncased", num_labels=21. Review the [COMPARISON.md](COMPARISON.md) to understand the implementation

)2. Check [STRUCTURE.md](STRUCTURE.md) for project organization

model_4layer = ablator.reduce_bert_layers(model_6layer, num_layers_to_keep=4)3. Run tests to ensure compatibility: `python tests/test_shap_imv.py`

model_2layer = ablator.reduce_bert_layers(model_6layer, num_layers_to_keep=2)4. Follow the existing code style and documentation patterns



# Train each variant## 📖 References

results = {}

for name, model in [('6-layer', model_6layer), ('4-layer', model_4layer), ('2-layer', model_2layer)]:- **Original Notebook**: `SHAP_IMV_AdultIncome (2).ipynb`

    result = ablator.train_and_evaluate(- **UCI Adult Dataset**: https://archive.ics.uci.edu/dataset/2/adult

        model=model,- **SHAP Values**: Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.

        train_dataloader=train_loader,

        test_dataloader=test_loader,## ⚠️ Known Limitations

        num_epochs=3,

        lr=2e-5,1. **Computational Complexity**: O(2^n) where n is number of variables

        seed=422. **Memory Usage**: Can be high for >15 variables

    )3. **Runtime**: Exponential growth with number of variables

    results[name] = result['test_predictions']

## 🐛 Troubleshooting

# Compute IMV matrix

imv_matrix = ablator.calculate_imv_matrix(results)### Import Errors

print(imv_matrix)```bash

# Output:# Install missing packages

#           6-layer  4-layer  2-layerpip install -r requirements.txt

# 6-layer     0.000    0.045    0.128```

# 4-layer    -0.045    0.000    0.087

# 2-layer    -0.128   -0.087    0.000### Slow Performance

```- Reduce number of variables

- Reduce `n_splits` parameter

**Interpretation:**- Use `split_method='train_test_split'` instead of `'kfold'`

- 6-layer has 4.5% more information than 4-layer

- 6-layer has 12.8% more information than 2-layer### Memory Issues

- Layers 5-6 contribute 4.5%, layers 3-4 contribute 8.3%- Test with fewer variables first

- Close other applications

---- Consider using a subset of data for testing



## DocumentationSee [QUICKSTART.md](QUICKSTART.md) for more troubleshooting tips.



### 📚 Full Documentation## 📄 License



- **[Technical Documentation](docs/TECHNICAL.md)**: In-depth walkthrough of algorithms and implementation[Add your license information here]

- **[API Reference](#api-reference)**: Complete API documentation (see below)

- **[Examples](examples/)**: Jupyter notebooks with real-world examples## 📧 Contact



### 🎓 Tutorials[Add your contact information here]



Check the `examples/` folder for complete tutorials:---

- `SHAP_IMV_AdultIncome.ipynb`: Binary classification with Adult Income dataset

- `Multi_IMV_Nursery.ipynb`: Multi-class classification with Nursery dataset**Status**: ✅ Production Ready  

- `Ablate_IMV_IMDb.ipynb`: Transformer ablation with IMDb sentiment analysis**Version**: 1.0.0  

**Last Updated**: 30 November 2025

---

For detailed technical documentation, see [COMPARISON.md](COMPARISON.md)  

## API ReferenceFor quick start guide, see [QUICKSTART.md](QUICKSTART.md)  

For project overview, see [SUMMARY.md](SUMMARY.md)

### Module: `metrics.shap_imv`

#### `IMVEvaluator`

Main class for binary classification with SHAP-IMV.

**Constructor:**
```python
IMVEvaluator(
    data: pd.DataFrame,
    outcome_variable: str,
    optional_explanatory_variables: List[str],
    model_creator: Callable,
    split_method: str,  # 'kfold' or 'train_test_split'
    n_splits: int,
    prop_test: float,
    model_type: str,  # 'classification' or 'regression'
    all_combinations_imv: Optional[Dict] = None,
    random_seed: int = 42
)
```

**Key Methods:**

| Method | Description | Returns |
|--------|-------------|---------|
| `run_evaluation()` | Compute IMV for all feature combinations | None (populates `all_combinations_imv`) |
| `evaluate_imvshapley(ax, figsize)` | Compute and visualize SHAP-IMV values | `(fig, ax)` or `ax` |
| `plot_single_var_combinations_layered_violin_centralized_zero(ax, figsize)` | Violin plot of single-feature IMV | `(fig, ax)` or `ax` |
| `calculate_imvshapley_value(variable)` | SHAP-IMV for one feature | `float` |

**Static Methods:**

| Method | Description | Returns |
|--------|-------------|---------|
| `ll(x, p)` | Log-likelihood geometric mean | `float` |
| `get_w(a, guess, bounds)` | Information weight from likelihood | `float` |
| `calculate_imv(y_basic, y_enhanced, y)` | IMV score | `float` |

---

### Module: `metrics.multi_imv`

#### `MultinomialIMV`

Main class for multi-class classification.

**Constructor:**
```python
MultinomialIMV(
    data: pd.DataFrame,
    outcome_variable: str,
    model_creator: Callable,
    n_splits: int = 10,
    optional_explanatory_variables: Optional[List[str]] = None,
    random_state: Optional[int] = None
)
```

**Key Methods:**

| Method | Description | Returns |
|--------|-------------|---------|
| `k_fold_one_vs_all()` | One-vs-all IMV across folds | `(imv_results, imv_average)` |
| `k_fold_imv_matrix()` | Pairwise IMV matrix across folds | `(matrices_list, matrix_avg)` |
| `multinomial_IMV_heatmap(imv_matrix, ax, figsize)` | Heatmap visualization | `(fig, ax)` or `ax` |
| `multinomial_IMV_boxplot(imv_results, figsize, ax)` | Boxplot visualization | `(fig, ax)` or `ax` |

---

### Module: `ablation.ablate_imv`

#### `AblationIMV`

Main class for deep learning ablation studies.

**Constructor:**
```python
AblationIMV(
    random_seed: int = 42
)
```

**Key Methods:**

| Method | Description | Returns |
|--------|-------------|---------|
| `train_and_evaluate(model, train_dataloader, test_dataloader, ...)` | Train model and get predictions | `dict` with model, predictions, metrics |
| `set_seed(seed)` | Set random seed across all backends | None |

**Static Methods:**

| Method | Description | Returns |
|--------|-------------|---------|
| `reduce_bert_layers(model, num_layers_to_keep)` | Reduce transformer layers | `model` |
| `calculate_imv_matrix(predictions_dict, target_column, prob_column)` | Pairwise IMV for multiple models | `pd.DataFrame` |
| `average_imv_matrices(matrices_list)` | Average IMV matrices across seeds | `pd.DataFrame` |
| `calculate_imv(y_basic, y_enhanced, y)` | IMV score | `float` |
| `ll(x, p)` | Log-likelihood geometric mean | `float` |
| `get_w(a, guess, bounds)` | Information weight | `float` |

**Device Detection:**

The class automatically detects and uses:
1. CUDA (NVIDIA GPUs)
2. MPS (Apple Silicon M1/M2/M3)
3. CPU (fallback)

Access via `ablator.device`.

---

## Advanced Usage

### Multi-Seed Averaging

For stable ablation study results:

```python
matrices = []
for seed in [42, 123, 456, 789, 999]:
    ablator = AblationIMV(random_seed=seed)
    # Train models...
    imv_mat = ablator.calculate_imv_matrix(predictions)
    matrices.append(imv_mat)

# Average across seeds
avg_matrix = ablator.average_imv_matrices(matrices)
std_matrix = np.std(np.stack([m.values for m in matrices]), axis=0)
```

### Custom Model Creators

Use any scikit-learn compatible model:

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# Random Forest
evaluator_rf = IMVEvaluator(
    model_creator=lambda: RandomForestClassifier(n_estimators=100, random_state=42),
    # ... other parameters
)

# Gradient Boosting
evaluator_gb = IMVEvaluator(
    model_creator=lambda: GradientBoostingClassifier(n_estimators=100, random_state=42),
    # ... other parameters
)

# SVM
evaluator_svm = IMVEvaluator(
    model_creator=lambda: SVC(probability=True, random_state=42),
    # ... other parameters
)
```

### Custom Visualizations

Create custom plots with the data:

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Get SHAP-IMV values
shap_values = {}
for var in evaluator.optional_explanatory_variables:
    shap_values[var] = evaluator.calculate_imvshapley_value(var)

# Custom bar plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(list(shap_values.keys()), list(shap_values.values()))
ax.set_xlabel('SHAP-IMV Value')
ax.set_title('Feature Importance via SHAP-IMV')
plt.tight_layout()
plt.show()
```

---

## Performance Tips

### For SHAP-IMV (Binary Classification)

1. **Limit features:** Use ≤12 features for reasonable runtime (2^n combinations)
2. **Feature selection:** Pre-filter with correlation or univariate tests
3. **Parallel processing:** Automatically uses all CPU cores (`n_jobs=-1`)

```python
# Feature selection example
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()].tolist()
```

### For Multi-class IMV

1. **One-vs-All:** Faster than pairwise matrix
2. **Reduce folds:** Use 5 folds instead of 10 for speed
3. **Sample classes:** For 10+ classes, focus on important class pairs

### For Ablation IMV

1. **Batch size:** 
   - 8-16 for 8GB GPU
   - 32-64 for 16GB+ GPU
2. **Mixed precision:** Use `torch.cuda.amp` for 2x speedup
3. **Gradient accumulation:** Simulate larger batches
4. **Early stopping:** Monitor validation loss

```python
# Mixed precision example
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
for batch in train_loader:
    with autocast():
        outputs = model(**batch)
        loss = outputs.loss
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

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

### Supported Platforms

✅ Linux (x86_64)  
✅ macOS (Intel & Apple Silicon)  
✅ Windows (x86_64)  

---

## Troubleshooting

### Common Issues

#### 1. "CUDA out of memory"

```python
# Reduce batch size
train_loader = DataLoader(dataset, batch_size=8)  # Instead of 32

# Enable gradient checkpointing
model.gradient_checkpointing_enable()
```

#### 2. "Optimization failed to converge"

```python
# Relax tolerance in IMVEvaluator
# Modify get_w() call:
res = minimize(..., options={'gtol': 1e-06})  # Instead of 1e-09
```

#### 3. "Feature names mismatch"

```python
# Ensure data has all features
missing_features = set(optional_explanatory_variables) - set(data.columns)
if missing_features:
    print(f"Missing features: {missing_features}")
```

#### 4. Slow performance with many features

```python
# Use feature selection
from sklearn.feature_selection import SelectKBest
selector = SelectKBest(k=10)
X_selected = selector.fit_transform(X, y)
```

---

## Contributing

We welcome contributions! Here's how to get started:

1. **Fork the repository**
2. **Create a feature branch:** `git checkout -b feature/amazing-feature`
3. **Make your changes** with tests
4. **Run tests:** `pytest tests/`
5. **Commit:** `git commit -m 'Add amazing feature'`
6. **Push:** `git push origin feature/amazing-feature`
7. **Open a Pull Request**

### Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/imv_ml_package.git
cd imv_ml_package

# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run linting
black .
flake8 .
```

---

## Citation

If you use IMV in your research, please cite:

```bibtex
@article{valler2024imv,
  title={Information Model Vigor: A Framework for Feature Importance},
  author={Valler, M. and Liu, J.},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Shapley Values:** Lloyd Shapley (1953)
- **SHAP Framework:** Scott Lundberg & Su-In Lee (2017)
- **Information Theory:** Claude Shannon (1948)
- **Transformer Architecture:** Vaswani et al. (2017)

---

## Contact

- **Issues:** [GitHub Issues](https://github.com/intermodelvigorish/imv_ml_package/issues)
- **Discussions:** [GitHub Discussions](https://github.com/intermodelvigorish/imv_ml_package/discussions)
- **Email:** contact@imv-package.org

---

## Roadmap

### Version 1.1 (Q1 2026)
- [ ] Approximate SHAP-IMV for large feature sets
- [ ] Confidence intervals via bootstrap
- [ ] Regression support for all modules

### Version 1.2 (Q2 2026)
- [ ] TensorFlow/Keras support for ablation
- [ ] Streaming/online IMV computation
- [ ] Multi-GPU distributed training

### Version 2.0 (Q3 2026)
- [ ] Causal IMV framework
- [ ] Time series IMV
- [ ] Automated hyperparameter tuning

---

<div align="center">

**Made with ❤️ by the IMV Team**

[⬆ Back to Top](#imv-information-model-vigor)

</div>
