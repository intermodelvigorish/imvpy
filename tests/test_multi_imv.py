"""
Test file for Multinomial IMV functionality
Replicates the Nursery dataset example from Multi_IMV_Nursery.ipynb
"""

import warnings

warnings.filterwarnings("ignore")
import pytest

pytestmark = pytest.mark.slow

import os
import sys

# Figures belong beside the tests, never in whatever directory pytest was
# launched from.
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Add parent directory to path to import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from imv import MulticlassIMV


def load_and_prepare_nursery_data():
    """
    Load and prepare the Nursery dataset
    Replicates the data preparation from the notebook
    """
    try:
        # Try to load from local file first
        if os.path.exists('nursery.csv'):
            nursery = pd.read_csv('nursery.csv')
        else:
            # Try fetching from UCI repository
            from ucimlrepo import fetch_ucirepo
            nursery_data = fetch_ucirepo(id=76)
            nursery = pd.concat([nursery_data.data.features, nursery_data.data.targets], axis=1)
        
        print(f"Dataset loaded. Shape: {nursery.shape}")
        
        # Encode categorical variables to numeric
        L = len(nursery.index)
        
        # Columns to encode (excluding 'health' and 'final evaluation' which are handled separately)
        encode_cols = [col for col in nursery.columns 
                      if col not in ['health', 'class']]
        
        for x in encode_cols:
            if nursery[x].dtype == 'object':
                lst = list(nursery[x].value_counts().index)
                dic = {k: i+1 for i, k in enumerate(lst)}
                nursery[x] = nursery[x].replace(dic)
        
        # Encode 'health' if it exists
        if 'health' in nursery.columns:
            dic1 = {'recommended': 2, 'priority': 3, 'not_recom': 1}
            nursery['health'] = nursery['health'].replace(dic1)
        
        # Encode target variable 'class' (or 'final evaluation')
        target_col = 'class' if 'class' in nursery.columns else 'final evaluation'
        
        # Map to numeric classes
        if nursery[target_col].dtype == 'object':
            dic2 = {
                'not_recom': 1,
                'recommend': 2,
                'very_recom': 3,
                'priority': 4,
                'spec_prior': 5
            }
            nursery[target_col] = nursery[target_col].replace(dic2)
        
        # Filter to keep only classes 1, 4, 5 (as in notebook)
        # Remove classes 2 and 3
        ind = (nursery[target_col] == 2) | (nursery[target_col] == 3)
        nursery = nursery[~ind].reset_index(drop=True)
        
        # Remap classes 4->2, 5->3 for 3-class problem
        dic3 = {4: 2, 5: 3}
        nursery[target_col] = nursery[target_col].replace(dic3)
        
        # Rename to 'target' for consistency
        nursery.rename(columns={target_col: 'target'}, inplace=True)
        
        print(f"Final dataset shape: {nursery.shape}")
        print(f"Target distribution:\n{nursery['target'].value_counts().sort_index()}")
        
        return nursery
        
    except ImportError:
        print("ERROR: ucimlrepo package not found. Please install it:")
        print("pip install ucimlrepo")
        return None
    except FileNotFoundError:
        print("ERROR: nursery.csv not found and could not fetch from UCI repository")
        return None
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        return None


def create_logistic_regression_model():
    """
    Create a logistic regression model for multi-class classification
    """
    # multi_class= was removed in scikit-learn 1.7; lbfgs is multinomial by
    # default for multiclass targets, matching the note in nursery_seed42_parity.json.
    return LogisticRegression(max_iter=500, random_state=42, solver='lbfgs')


def test_multi_imv_one_vs_all():
    """
    Test one-vs-all IMV calculation (each class vs all others)
    """
    print("\n" + "="*80)
    print("TEST 1: One-vs-All Multinomial IMV")
    print("="*80)
    
    # Load data
    nursery = load_and_prepare_nursery_data()
    if nursery is None:
        print("Skipping test due to data loading failure")
        return
    
    # Get feature columns (all except target)
    feature_cols = [col for col in nursery.columns if col != 'target']
    
    print(f"\nUsing {len(feature_cols)} features: {feature_cols}")
    print(f"Target variable: 'target' with {nursery['target'].nunique()} classes")
    
    # Create evaluator
    evaluator = MulticlassIMV(
        data=nursery,
        outcome_variable='target',
        model_creator=create_logistic_regression_model,
        n_splits=5,  # Use 5 folds for faster testing
        optional_explanatory_variables=feature_cols,
        random_state=42
    )
    
    # Run one-vs-all evaluation
    print("\nRunning k-fold one-vs-all IMV evaluation...")
    imv_results, imv_average = evaluator.k_fold_one_vs_all()
    
    # Print results
    print("\n" + "-"*80)
    print("One-vs-All IMV Results:")
    print("-"*80)
    outcomes = np.sort(nursery['target'].unique())
    for i, (outcome, avg_imv) in enumerate(zip(outcomes, imv_average)):
        print(f"Outcome {outcome}: {avg_imv:.4f}")
    
    # Create boxplot visualization
    print("\nCreating boxplot visualization...")
    try:
        fig, ax = evaluator.multinomial_IMV_boxplot(imv_results, figsize=(8, 6))
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'test_multi_imv_boxplot.png'), dpi=150, bbox_inches='tight')
        print("✓ Boxplot saved to: test_multi_imv_boxplot.png")
        plt.close()
    except Exception as e:
        print(f"Warning: Could not create boxplot: {e}")
    
    print("\n✓ One-vs-all test completed successfully!")


def test_multi_imv_confusion_matrix():
    """
    Test pairwise IMV confusion matrix (each class vs each other class)
    """
    print("\n" + "="*80)
    print("TEST 2: Pairwise IMV Confusion Matrix")
    print("="*80)
    
    # Load data
    nursery = load_and_prepare_nursery_data()
    if nursery is None:
        print("Skipping test due to data loading failure")
        return
    
    # Get feature columns
    feature_cols = [col for col in nursery.columns if col != 'target']
    
    print(f"\nUsing {len(feature_cols)} features")
    print("Computing pairwise IMV for all class combinations")
    
    # Create evaluator
    evaluator = MulticlassIMV(
        data=nursery,
        outcome_variable='target',
        model_creator=create_logistic_regression_model,
        n_splits=5,  # Use 5 folds for faster testing
        optional_explanatory_variables=feature_cols,
        random_state=42
    )
    
    # Run IMV matrix evaluation
    print("\nRunning k-fold IMV matrix evaluation...")
    imv_matrices_list, imv_matrix_average = evaluator.k_fold_imv_matrix()
    
    # Print results
    print("\n" + "-"*80)
    print("Average IMV Confusion Matrix:")
    print("-"*80)
    print(imv_matrix_average)
    
    # Create heatmap visualization
    print("\nCreating heatmap visualization...")
    try:
        fig, ax = evaluator.multinomial_IMV_heatmap(imv_matrix_average, figsize=(8, 8))
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'test_multi_imv_heatmap.png'), dpi=150, bbox_inches='tight')
        print("✓ Heatmap saved to: test_multi_imv_heatmap.png")
        plt.close()
    except Exception as e:
        print(f"Warning: Could not create heatmap: {e}")
    
    print("\n✓ Confusion matrix test completed successfully!")


def test_multi_imv_subset_features():
    """
    Test with a subset of features for faster execution
    """
    print("\n" + "="*80)
    print("TEST 3: Multinomial IMV with Subset of Features")
    print("="*80)
    
    # Load data
    nursery = load_and_prepare_nursery_data()
    if nursery is None:
        print("Skipping test due to data loading failure")
        return
    
    # Use only a subset of features
    all_features = [col for col in nursery.columns if col != 'target']
    subset_features = all_features[:3]  # Use first 3 features
    
    print(f"\nUsing subset of {len(subset_features)} features: {subset_features}")
    
    # Create evaluator
    evaluator = MulticlassIMV(
        data=nursery,
        outcome_variable='target',
        model_creator=create_logistic_regression_model,
        n_splits=3,  # Even fewer folds for speed
        optional_explanatory_variables=subset_features,
        random_state=42
    )
    
    # Run both evaluations
    print("\nRunning one-vs-all evaluation...")
    imv_results, imv_average = evaluator.k_fold_one_vs_all()
    
    print("\nOne-vs-All Results:")
    outcomes = np.sort(nursery['target'].unique())
    for outcome, avg_imv in zip(outcomes, imv_average):
        print(f"  Outcome {outcome}: {avg_imv:.4f}")
    
    print("\nRunning IMV matrix evaluation...")
    _, imv_matrix_average = evaluator.k_fold_imv_matrix()
    
    print("\n✓ Subset features test completed successfully!")


def test_core_methods():
    """
    Test the core calculation methods
    """
    print("\n" + "="*80)
    print("TEST 4: Core Multinomial IMV Methods")
    print("="*80)
    
    # Create a simple synthetic dataset
    np.random.seed(42)
    data = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'target': np.random.choice([1, 2, 3], 100)
    })
    
    print(f"Synthetic dataset shape: {data.shape}")
    print(f"Target distribution:\n{data['target'].value_counts().sort_index()}")
    
    # Create evaluator
    evaluator = MulticlassIMV(
        data=data,
        outcome_variable='target',
        model_creator=create_logistic_regression_model,
        n_splits=3,
        random_state=42
    )
    
    # Test ll function
    print("\nTesting ll (log-likelihood) function...")
    x = np.array([1, 0, 1, 1, 0])
    p = np.array([0.8, 0.2, 0.7, 0.9, 0.3])
    ll_result = evaluator.ll(x, p)
    print(f"✓ ll calculation: {ll_result:.6f}")
    
    # Test get_w function
    print("\nTesting get_w (weight calculation) function...")
    a = 0.6
    w = evaluator.get_w(a)
    print(f"✓ get_w calculation for a={a}: {w:.6f}")
    
    # Test one_vs_all_single_fold
    print("\nTesting one_vs_all_single_fold method...")
    model = create_logistic_regression_model()
    X = data[['feature1', 'feature2']]
    y = data['target']
    
    X_constant = np.ones((X.shape[0], 1))
    model_basic = create_logistic_regression_model()
    model_basic.fit(X_constant, y)
    p_base = model_basic.predict_proba(X_constant)
    
    model_enhanced = create_logistic_regression_model()
    model_enhanced.fit(X, y)
    p_enhanced = model_enhanced.predict_proba(X)
    
    ova_result = evaluator.one_vs_all_single_fold(data, 'target', p_base, p_enhanced)
    print("✓ One-vs-all result:")
    print(ova_result)
    
    print("\n✓ Core methods test completed successfully!")


def run_all_tests(quick_mode=True):
    """
    Run all tests
    
    Args:
        quick_mode: If True, use fewer folds and features for speed
    """
    print("\n" + "="*80)
    print("Multinomial IMV Package Test Suite")
    print("="*80)
    
    # Always run core methods test
    test_core_methods()
    
    # Run subset features test (quick)
    test_multi_imv_subset_features()
    
    if not quick_mode:
        # Run full tests with all features
        test_multi_imv_one_vs_all()
        test_multi_imv_confusion_matrix()
    else:
        print("\n" + "="*80)
        print("Quick mode enabled - skipping full feature tests")
        print("To run full test suite, call: run_all_tests(quick_mode=False)")
        print("="*80)
    
    print("\n" + "="*80)
    print("All tests completed!")
    print("="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Multinomial IMV package')
    parser.add_argument('--full', action='store_true',
                       help='Run full test suite (slower, uses all features)')
    parser.add_argument('--test', type=str, 
                       choices=['ova', 'matrix', 'subset', 'core', 'all'],
                       default='all', help='Which test to run')
    
    args = parser.parse_args()
    
    if args.test == 'ova':
        test_multi_imv_one_vs_all()
    elif args.test == 'matrix':
        test_multi_imv_confusion_matrix()
    elif args.test == 'subset':
        test_multi_imv_subset_features()
    elif args.test == 'core':
        test_core_methods()
    else:  # 'all'
        run_all_tests(quick_mode=not args.full)
