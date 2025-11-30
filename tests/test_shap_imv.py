"""
Test file for SHAP-IMV functionality
Replicates the Adult Income dataset example from SHAP_IMV_AdultIncome (2).ipynb
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Add parent directory to path to import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from imv import BinaryIMV, ll, get_w, calculate_imv  # Updated import for reorganized package structure


def load_and_prepare_adult_income_data():
    """
    Load and prepare the Adult Income dataset
    Replicates the data preparation from the notebook
    """
    try:
        from ucimlrepo import fetch_ucirepo
        
        # Fetch dataset
        print("Fetching Adult Income dataset...")
        adult = fetch_ucirepo(id=2)
        
        # Get features and targets
        X = adult.data.features
        y = adult.data.targets
        
        # Drop missing values
        X = X.dropna()
        print(f"Dataset shape after dropping NA: {X.shape}")
        
        # Encode categorical variables
        categorical = ['workclass', 'education', 'marital-status', 'occupation',
                      'relationship', 'race', 'sex', 'native-country']
        label_encoder = LabelEncoder()
        for col in categorical:
            if col in X.columns:
                X[col] = label_encoder.fit_transform(X[col])
        
        # Prepare target variable
        y = adult.data.targets['income']
        y = pd.Series(y, name="target")
        y = y.replace({'<=50K': 0, '<=50K.': 0, '>50K': 1, '>50K.': 1})
        
        # Combine features and target
        df_combined = pd.concat([X, y], axis=1)
        df_combined = df_combined.dropna()
        
        print(f"Final dataset shape: {df_combined.shape}")
        print(f"Target distribution:\n{df_combined['target'].value_counts()}")
        
        return df_combined
        
    except ImportError:
        print("ERROR: ucimlrepo package not found. Please install it:")
        print("pip install ucimlrepo")
        return None
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        return None


def create_logistic_regression_model():
    """
    Create a logistic regression model with the same parameters as the notebook
    """
    return LogisticRegression(max_iter=500, random_state=42)


def test_shap_imv_basic():
    """
    Test basic SHAP-IMV functionality with a subset of variables
    This is a quick test to ensure the package works
    """
    print("\n" + "="*80)
    print("TEST 1: Basic SHAP-IMV Test with Subset of Variables")
    print("="*80)
    
    # Load data
    df_combined = load_and_prepare_adult_income_data()
    if df_combined is None:
        print("Skipping test due to data loading failure")
        return
    
    # Use only a subset of variables for faster testing
    test_variables = ['age', 'education', 'marital-status', 'sex', 'hours-per-week']
    
    print(f"\nTesting with variables: {test_variables}")
    
    # Create evaluator
    evaluator = BinaryIMV(
        data=df_combined,
        outcome_variable='target',
        optional_explanatory_variables=test_variables,
        model_creator=create_logistic_regression_model,
        split_method='kfold',
        n_splits=5,  # Use 5 folds for faster testing
        prop_test=0.2,
        model_type='classification',
        random_seed=42
    )
    
    # Run evaluation
    print("\nRunning IMV evaluation...")
    evaluator.run_evaluation()
    
    # Calculate SHAP-IMV values
    print("\nCalculating SHAP-IMV values...")
    imvshapley_values = {}
    for variable in test_variables:
        imvshapley_values[variable] = evaluator.calculate_imvshapley_value(variable)
    
    # Print results
    print("\n" + "-"*80)
    print("SHAP-IMV Results:")
    print("-"*80)
    sorted_vars = sorted(imvshapley_values.items(), key=lambda x: x[1], reverse=True)
    for var, value in sorted_vars:
        print(f"{var:20s}: {value:.4f}")
    
    # Create visualization
    print("\nCreating SHAP-IMV visualization...")
    try:
        fig, ax = evaluator.evaluate_imvshapley(figsize=(12, 4))
        plt.tight_layout()
        plt.savefig('test_shap_imv_basic.png', dpi=150, bbox_inches='tight')
        print("✓ Visualization saved to: test_shap_imv_basic.png")
        plt.close()
    except Exception as e:
        print(f"Warning: Could not create visualization: {e}")
    
    print("\n✓ Basic test completed successfully!")


def test_shap_imv_full():
    """
    Test SHAP-IMV with all variables (replicates the notebook exactly)
    This test takes longer to run
    """
    print("\n" + "="*80)
    print("TEST 2: Full SHAP-IMV Test with All Variables (Replicates Notebook)")
    print("="*80)
    
    # Load data
    df_combined = load_and_prepare_adult_income_data()
    if df_combined is None:
        print("Skipping test due to data loading failure")
        return
    
    # Use all variables as in the notebook
    all_variables = ['age', 'workclass', 'education', 'marital-status', 
                     'occupation', 'relationship', 'race', 'sex', 
                     'capital-gain', 'capital-loss', 'hours-per-week']
    
    print(f"\nTesting with all variables: {all_variables}")
    print(f"Total combinations to evaluate: {2**len(all_variables)}")
    
    # Create evaluator (matching notebook parameters exactly)
    evaluator = BinaryIMV(
        data=df_combined,
        outcome_variable='target',
        optional_explanatory_variables=all_variables,
        model_creator=create_logistic_regression_model,
        split_method='kfold',
        n_splits=10,  # Same as notebook
        prop_test=0.2,
        model_type='classification',
        random_seed=42
    )
    
    # Run evaluation
    print("\nRunning IMV evaluation (this may take several minutes)...")
    evaluator.run_evaluation()
    
    # Calculate SHAP-IMV values
    print("\nCalculating SHAP-IMV values...")
    imvshapley_values = {}
    for variable in all_variables:
        imvshapley_values[variable] = evaluator.calculate_imvshapley_value(variable)
    
    # Print results
    print("\n" + "-"*80)
    print("SHAP-IMV Results (All Variables):")
    print("-"*80)
    sorted_vars = sorted(imvshapley_values.items(), key=lambda x: x[1], reverse=True)
    for var, value in sorted_vars:
        print(f"{var:20s}: {value:.4f}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    try:
        # SHAP-IMV bar plot
        fig, ax = evaluator.evaluate_imvshapley(figsize=(12, 6))
        plt.tight_layout()
        plt.savefig('test_shap_imv_full.png', dpi=150, bbox_inches='tight')
        print("✓ SHAP-IMV visualization saved to: test_shap_imv_full.png")
        plt.close()
        
        # Single variable violin plot
        fig, ax = plt.subplots(figsize=(8, 6))
        evaluator.plot_single_var_combinations_layered_violin_centralized_zero(ax=ax)
        plt.tight_layout()
        plt.savefig('test_single_var_violin.png', dpi=150, bbox_inches='tight')
        print("✓ Single variable violin plot saved to: test_single_var_violin.png")
        plt.close()
    except Exception as e:
        print(f"Warning: Could not create visualization: {e}")
    
    print("\n✓ Full test completed successfully!")


def test_shap_imv_train_test_split():
    """
    Test SHAP-IMV with train_test_split instead of kfold
    """
    print("\n" + "="*80)
    print("TEST 3: SHAP-IMV with Train-Test Split")
    print("="*80)
    
    # Load data
    df_combined = load_and_prepare_adult_income_data()
    if df_combined is None:
        print("Skipping test due to data loading failure")
        return
    
    # Use subset of variables
    test_variables = ['age', 'education', 'sex', 'hours-per-week']
    
    print(f"\nTesting with variables: {test_variables}")
    
    # Create evaluator with train_test_split
    evaluator = BinaryIMV(
        data=df_combined,
        outcome_variable='target',
        optional_explanatory_variables=test_variables,
        model_creator=create_logistic_regression_model,
        split_method='train_test_split',  # Different split method
        n_splits=5,
        prop_test=0.2,
        model_type='classification',
        random_seed=42
    )
    
    # Run evaluation
    print("\nRunning IMV evaluation with train-test split...")
    evaluator.run_evaluation()
    
    # Calculate SHAP-IMV values
    print("\nCalculating SHAP-IMV values...")
    for variable in test_variables:
        evaluator.calculate_imvshapley_value(variable)
    
    print("\n✓ Train-test split test completed successfully!")


def test_imv_calculations():
    """
    Test the core IMV calculation methods
    """
    print("\n" + "="*80)
    print("TEST 4: Core IMV Calculation Methods")
    print("="*80)
    
    # Test the core IMV functions (now imported from imv.core)
    print("\nTesting core IMV functions...")
    
    # Test ll function
    x = np.array([1, 0, 1, 1, 0])
    p = np.array([0.8, 0.2, 0.7, 0.9, 0.3])
    ll_result = ll(x, p)
    print(f"✓ ll calculation: {ll_result:.6f}")
    
    # Test get_w function
    a = 0.6
    w = get_w(a)
    print(f"✓ get_w calculation for a={a}: {w:.6f}")
    
    # Test calculate_imv function
    y = np.array([1, 0, 1, 1, 0, 1])
    y_basic = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    y_enhanced = np.array([0.8, 0.2, 0.7, 0.9, 0.3, 0.85])
    imv_result = calculate_imv(y_basic, y_enhanced, y)
    print(f"✓ calculate_imv: {imv_result:.6f}")
    
    # Test calculate_weight function (specific to BinaryIMV class)
    s_size = 3
    n = 10
    weight = BinaryIMV.calculate_weight(s_size, n)
    print(f"✓ calculate_weight (s_size={s_size}, n={n}): {weight:.6f}")
    
    print("\n✓ Core calculation tests completed successfully!")


def run_all_tests(quick_mode=True):
    """
    Run all tests
    
    Args:
        quick_mode: If True, only run quick tests. If False, run all tests including slow ones.
    """
    print("\n" + "="*80)
    print("SHAP-IMV Package Test Suite")
    print("="*80)
    
    # Always run core calculation tests
    test_imv_calculations()
    
    # Always run basic test
    test_shap_imv_basic()
    
    # Run train-test split test
    test_shap_imv_train_test_split()
    
    if not quick_mode:
        # Run full test only if not in quick mode (takes longer)
        test_shap_imv_full()
    else:
        print("\n" + "="*80)
        print("Quick mode enabled - skipping full test with all variables")
        print("To run full test suite, call: run_all_tests(quick_mode=False)")
        print("="*80)
    
    print("\n" + "="*80)
    print("All tests completed!")
    print("="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test SHAP-IMV package')
    parser.add_argument('--full', action='store_true', 
                       help='Run full test suite (slower, includes all variables)')
    parser.add_argument('--test', type=str, choices=['basic', 'full', 'split', 'calc', 'all'],
                       default='all', help='Which test to run')
    
    args = parser.parse_args()
    
    if args.test == 'basic':
        test_shap_imv_basic()
    elif args.test == 'full':
        test_shap_imv_full()
    elif args.test == 'split':
        test_shap_imv_train_test_split()
    elif args.test == 'calc':
        test_imv_calculations()
    else:  # 'all'
        run_all_tests(quick_mode=not args.full)
