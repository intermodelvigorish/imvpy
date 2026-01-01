"""
SHAP-IMV Adult Income Analysis
Compares Logistic Regression, XGBoost, and LightGBM using SHAP-IMV
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from imv import BinaryIMV

# Try to import XGBoost and LightGBM
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip3 install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not available. Install with: pip3 install lightgbm")

# Setup directories
EXAMPLE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(EXAMPLE_DIR, 'data')
FIGURES_DIR = os.path.join(EXAMPLE_DIR, 'figures')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_and_prepare_adult_income_data():
    """
    Load and prepare the Adult Income dataset
    """
    data_file = os.path.join(DATA_DIR, 'adult_income_processed.csv')

    # Try to load cached data
    if os.path.exists(data_file):
        try:
            df_combined = pd.read_csv(data_file)
            print(f"Loaded processed dataset from: {data_file}")
            return df_combined
        except Exception as e:
            print(f"Warning: failed to read cached dataset: {e}")

    try:
        from ucimlrepo import fetch_ucirepo

        print("Fetching Adult Income dataset...")
        adult = fetch_ucirepo(id=2)

        X = adult.data.features
        y = adult.data.targets

        X = X.dropna()
        print(f"Dataset shape after dropping NA: {X.shape}")

        # Encode categorical variables
        categorical = ['workclass', 'education', 'marital-status', 'occupation',
                      'relationship', 'race', 'sex', 'native-country']
        label_encoder = LabelEncoder()
        for col in categorical:
            if col in X.columns:
                X[col] = label_encoder.fit_transform(X[col])

        # Prepare target
        y = adult.data.targets['income']
        y = pd.Series(y, name="target")
        y = y.replace({'<=50K': 0, '<=50K.': 0, '>50K': 1, '>50K.': 1})

        df_combined = pd.concat([X, y], axis=1)
        df_combined = df_combined.dropna()

        print(f"Final dataset shape: {df_combined.shape}")

        # Save for future use
        try:
            df_combined.to_csv(data_file, index=False)
            print(f"Saved processed dataset to: {data_file}")
        except Exception as e:
            print(f"Warning: could not save dataset: {e}")

        return df_combined

    except ImportError:
        print("ERROR: ucimlrepo not found. Install with: pip install ucimlrepo")
        return None
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        return None


def create_logistic_regression_model():
    """Create Logistic Regression model"""
    return LogisticRegression(max_iter=500, random_state=42)


def create_xgboost_model():
    """Create XGBoost model"""
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost not available")
    return xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )


def create_lightgbm_model():
    """Create LightGBM model"""
    if not LIGHTGBM_AVAILABLE:
        raise ImportError("LightGBM not available")
    return lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        verbose=-1
    )


def run_shap_imv_analysis(df_combined, model_name, model_creator, seed, all_variables):
    """
    Run SHAP-IMV analysis for a given model and seed
    
    Returns:
        dict: Dictionary with SHAP-IMV values for each feature
    """
    print(f"\n{'='*80}")
    print(f"Running {model_name} with seed {seed}")
    print(f"{'='*80}")
    
    # Create evaluator
    evaluator = BinaryIMV(
        data=df_combined,
        outcome_variable='target',
        optional_explanatory_variables=all_variables,
        model_creator=model_creator,
        split_method='kfold',
        n_splits=10,
        prop_test=0.2,
        model_type='classification',
        random_seed=seed
    )
    
    # Run evaluation
    print("Running IMV evaluation...")
    evaluator.run_evaluation()
    
    # Calculate SHAP-IMV values
    print("Calculating SHAP-IMV values...")
    imvshapley_values = {}
    for variable in all_variables:
        imvshapley_values[variable] = evaluator.calculate_imvshapley_value(variable)
    
    return imvshapley_values, evaluator


def calculate_performance_metrics(model_creator, df_combined, all_variables, seed):
    """
    Calculate accuracy and precision using train-test split
    
    Returns:
        tuple: (accuracy, precision)
    """
    from sklearn.model_selection import train_test_split
    
    # Prepare data
    X = df_combined[all_variables]
    y = df_combined['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    
    # Train model
    model = model_creator()
    model.fit(X_train, y_train)
    
    # Get predictions
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)
    else:
        y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred) * 100
    precision = precision_score(y_test, y_pred, zero_division=0) * 100
    
    return accuracy, precision


def aggregate_shap_values(results_dict):
    """
    Aggregate SHAP-IMV values across multiple seeds
    
    Args:
        results_dict: Dict of {seed: {feature: shap_value}}
    
    Returns:
        tuple: (mean_values, std_values) as dictionaries
    """
    all_features = list(next(iter(results_dict.values())).keys())
    
    mean_values = {}
    std_values = {}
    
    for feature in all_features:
        values = [results_dict[seed][feature] for seed in results_dict.keys()]
        mean_values[feature] = np.mean(values)
        std_values[feature] = np.std(values)
    
    return mean_values, std_values


def plot_shap_imv_comparison(results, top_n=5):
    """
    Create bar plot comparing SHAP-IMV values across models
    
    Args:
        results: Dict of {model_name: (mean_values, std_values)}
        top_n: Number of top features to display per model
    
    Note:
        Features are selected based on AVERAGED SHAP-IMV values across all seeds.
        The top_n features with highest mean SHAP-IMV are displayed.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    model_names = ['Logistic Regression', 'XGBoost', 'LightGBM']
    
    for idx, (ax, model_name) in enumerate(zip(axes, model_names)):
        if model_name not in results:
            ax.set_title(f'{model_name}\n(Not Available)')
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        
        mean_values, std_values = results[model_name]
        
        # Get top N features based on AVERAGED values across all seeds
        # Sort by mean SHAP-IMV (descending) and take top_n
        sorted_features = sorted(mean_values.items(), key=lambda x: x[1], reverse=True)[:top_n]
        features = [f[0] for f in sorted_features]
        means = [f[1] for f in sorted_features]
        stds = [std_values[f[0]] for f in sorted_features]
        
        # Create bar plot
        y_pos = np.arange(len(features))
        bars = ax.barh(y_pos, means, xerr=stds, 
                       color=plt.cm.Greens(np.linspace(0.4, 0.8, len(features))),
                       capsize=5, alpha=0.8, edgecolor='black', linewidth=1.2)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()
        ax.set_xlabel('Average SHAP-IMV Value', fontsize=11)
        ax.set_title(f'Adult Income ({model_name})', fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax.text(mean + std + 0.005, i, f'{mean:.3f}', 
                   va='center', fontsize=9)
    
    plt.tight_layout()
    output_path = os.path.join(FIGURES_DIR, 'shap_imv_model_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Comparison plot saved to: {output_path}")
    plt.close()


def create_performance_table(performance_results):
    """
    Create performance comparison table like the image
    
    Args:
        performance_results: Dict of {model_name: {'accuracy': [], 'precision': []}}
    
    Returns:
        pd.DataFrame: Formatted performance table
    """
    rows = []
    
    model_display_names = {
        'Logistic Regression': 'LR',
        'XGBoost': 'XGB',
        'LightGBM': 'LGBM'
    }
    
    for model_name, metrics in performance_results.items():
        if metrics['accuracy']:  # Check if we have data
            display_name = model_display_names.get(model_name, model_name)
            
            # Calculate averages
            avg_acc = np.mean(metrics['accuracy'])
            avg_prec = np.mean(metrics['precision'])
            
            rows.append({
                'Methods': display_name,
                'Adult Income Accuracy': f"{avg_acc:.2f}",
                'Adult Income Precision': f"{avg_prec:.2f}",
                'Average Accuracy': f"{avg_acc:.2f}",
                'Average Precision': f"{avg_prec:.2f}"
            })
    
    df = pd.DataFrame(rows)
    
    # Highlight best values
    if not df.empty:
        # Find max values for highlighting
        acc_col = 'Adult Income Accuracy'
        prec_col = 'Adult Income Precision'
        
        df[acc_col] = df[acc_col].astype(float)
        df[prec_col] = df[prec_col].astype(float)
        
        max_acc_idx = df[acc_col].idxmax()
        max_prec_idx = df[prec_col].idxmax()
        
        # Convert back to formatted strings and underline best
        for col in [acc_col, prec_col, 'Average Accuracy', 'Average Precision']:
            df[col] = df[col].apply(lambda x: f"{float(x):.2f}")
        
        print("\n" + "="*80)
        print("PERFORMANCE COMPARISON TABLE")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80)
        
        # Save to CSV
        csv_path = os.path.join(FIGURES_DIR, 'performance_comparison.csv')
        df.to_csv(csv_path, index=False)
        print(f"✓ Performance table saved to: {csv_path}")
    
    return df


def main():
    """Main execution function"""
    print("="*80)
    print("SHAP-IMV Multi-Model Comparison: Adult Income Dataset")
    print("="*80)
    
    # Load data
    df_combined = load_and_prepare_adult_income_data()
    if df_combined is None:
        print("ERROR: Could not load dataset")
        return
    
    # Define variables (same as full test)
    all_variables = ['age', 'workclass', 'education', 'marital-status', 
                     'occupation', 'relationship', 'race', 'sex', 
                     'capital-gain', 'capital-loss', 'hours-per-week']
    
    # Seeds to use
    seeds = [42, 43, 44]
    
    # Models to test
    models = {
        'Logistic Regression': create_logistic_regression_model,
    }
    
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = create_xgboost_model
    else:
        print("\nWarning: XGBoost not available, skipping...")
    
    if LIGHTGBM_AVAILABLE:
        models['LightGBM'] = create_lightgbm_model
    else:
        print("\nWarning: LightGBM not available, skipping...")
    
    # Store results
    all_results = {}  # {model_name: {seed: {feature: shap_value}}}
    performance_results = {}  # {model_name: {'accuracy': [], 'precision': []}}
    
    # Run analysis for each model and seed
    for model_name, model_creator in models.items():
        all_results[model_name] = {}
        performance_results[model_name] = {'accuracy': [], 'precision': []}
        
        for seed in seeds:
            try:
                shap_values, evaluator = run_shap_imv_analysis(
                    df_combined, model_name, model_creator, seed, all_variables
                )
                all_results[model_name][seed] = shap_values
                
                # Calculate performance metrics with separate train-test evaluation
                acc, prec = calculate_performance_metrics(
                    model_creator, df_combined, all_variables, seed
                )
                performance_results[model_name]['accuracy'].append(acc)
                performance_results[model_name]['precision'].append(prec)
                
                print(f"Performance: Accuracy={acc:.2f}%, Precision={prec:.2f}%")
                
                # Print top 5 for this individual seed (for reference)
                print(f"\nTop 5 features for {model_name} (seed {seed} only):")
                sorted_features = sorted(shap_values.items(), key=lambda x: x[1], reverse=True)[:5]
                for feat, val in sorted_features:
                    print(f"  {feat:20s}: {val:.4f}")
                
            except Exception as e:
                print(f"ERROR running {model_name} with seed {seed}: {e}")
                import traceback
                traceback.print_exc()
    
    # Aggregate results across seeds
    aggregated_results = {}
    for model_name, seed_results in all_results.items():
        if seed_results:  # Check if we have results
            mean_vals, std_vals = aggregate_shap_values(seed_results)
            aggregated_results[model_name] = (mean_vals, std_vals)
    
    # Print final averaged top 5 features for each model
    if aggregated_results:
        print("\n" + "="*80)
        print("FINAL TOP 5 FEATURES (Averaged across all seeds)")
        print("="*80)
        for model_name, (mean_vals, std_vals) in aggregated_results.items():
            print(f"\n{model_name}:")
            sorted_features = sorted(mean_vals.items(), key=lambda x: x[1], reverse=True)[:5]
            for feat, mean_val in sorted_features:
                std_val = std_vals[feat]
                print(f"  {feat:20s}: {mean_val:.4f} ± {std_val:.4f}")
    
    # Create visualizations
    if aggregated_results:
        plot_shap_imv_comparison(aggregated_results, top_n=5)
    
    # Create performance table
    if performance_results:
        create_performance_table(performance_results)
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()


