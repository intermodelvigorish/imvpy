"""
Create Combined Multi-IMV Confusion Matrix Figure
Shows XGBoost results for Nursery, Car Evaluation, and Dry Bean datasets side-by-side
Replicates Figure 3 from the paper
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Setup directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
FIGURES_DIR = os.path.join(SCRIPT_DIR, 'figures')


def load_xgboost_imv_matrix(dataset_name):
    """Load XGBoost IMV matrix results for a dataset"""
    filename = os.path.join(RESULTS_DIR, f'{dataset_name}_xgboost_results.pkl')
    
    if not os.path.exists(filename):
        print(f"ERROR: {filename} not found!")
        print(f"Please run multi_imv_{dataset_name}.py first to generate results.")
        return None, None
    
    with open(filename, 'rb') as f:
        results = pickle.load(f)
    
    imv_results = results['imv_results']
    
    # Aggregate across seeds
    matrices = [imv_results[seed]['imv_matrix_avg'].values for seed in imv_results.keys()]
    matrices_stack = np.stack(matrices)
    mean_matrix = np.mean(matrices_stack, axis=0)
    
    # Get class labels from first seed's matrix
    first_matrix = imv_results[list(imv_results.keys())[0]]['imv_matrix_avg']
    mean_df = pd.DataFrame(mean_matrix, index=first_matrix.index, columns=first_matrix.columns)
    
    return mean_df, first_matrix.index.tolist()


def create_combined_figure():
    """Create side-by-side IMV confusion matrices for all three datasets"""
    
    print("="*80)
    print("Creating Combined Multi-IMV Confusion Matrix Figure (XGBoost)")
    print("="*80)
    
    # Load data for all three datasets
    print("\nLoading results...")
    nursery_matrix, nursery_classes = load_xgboost_imv_matrix('nursery')
    car_matrix, car_classes = load_xgboost_imv_matrix('car_evaluation')
    bean_matrix, bean_classes = load_xgboost_imv_matrix('dry_bean')
    
    # Check if all loaded successfully
    if nursery_matrix is None or car_matrix is None or bean_matrix is None:
        print("\nERROR: Could not load all required results!")
        print("Please run the following scripts first:")
        print("  1. python multi_imv_nursery.py")
        print("  2. python multi_imv_car_evaluation.py")
        print("  3. python multi_imv_dry_bean.py")
        return
    
    # Define class names for display
    nursery_names = ['Not recommend', 'Priority acceptance', 'Special priority']
    car_names = ['Unacceptable', 'Acceptable', 'Good', 'Very good']
    bean_names = ['Seker', 'Barbunya', 'Bombay', 'Cali', 'Dermosan', 'Horoz', 'Sira']
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Configure color palette to match the paper's style
    # Using 'crest' colormap from seaborn
    cmap = 'crest'
    
    # Plot 1: Nursery
    print("  Creating Nursery heatmap...")
    sns.heatmap(nursery_matrix, annot=True, fmt='.3f', cmap=cmap,
                xticklabels=nursery_names, yticklabels=nursery_names,
                ax=axes[0], vmin=0, vmax=1, cbar=False,
                linewidths=0.5, linecolor='white',
                annot_kws={'size': 10})
    axes[0].set_title('Nursery', fontsize=14, fontweight='bold', pad=10)
    axes[0].set_xlabel('')
    axes[0].set_ylabel('')
    axes[0].tick_params(axis='x', rotation=45, labelsize=10)
    axes[0].tick_params(axis='y', rotation=0, labelsize=10)
    
    # Plot 2: Car Evaluation
    print("  Creating Car Evaluation heatmap...")
    sns.heatmap(car_matrix, annot=True, fmt='.3f', cmap=cmap,
                xticklabels=car_names, yticklabels=car_names,
                ax=axes[1], vmin=0, vmax=1, cbar=False,
                linewidths=0.5, linecolor='white',
                annot_kws={'size': 10})
    axes[1].set_title('Car Evaluation', fontsize=14, fontweight='bold', pad=10)
    axes[1].set_xlabel('')
    axes[1].set_ylabel('')
    axes[1].tick_params(axis='x', rotation=45, labelsize=10)
    axes[1].tick_params(axis='y', rotation=0, labelsize=10)
    
    # Plot 3: Dry Bean
    print("  Creating Dry Bean heatmap...")
    im = sns.heatmap(bean_matrix, annot=True, fmt='.3f', cmap=cmap,
                     xticklabels=bean_names, yticklabels=bean_names,
                     ax=axes[2], vmin=0, vmax=1, cbar=False,
                     linewidths=0.5, linecolor='white',
                     annot_kws={'size': 9})
    axes[2].set_title('Dry Bean', fontsize=14, fontweight='bold', pad=10)
    axes[2].set_xlabel('')
    axes[2].set_ylabel('')
    axes[2].tick_params(axis='x', rotation=45, labelsize=9)
    axes[2].tick_params(axis='y', rotation=0, labelsize=9)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(FIGURES_DIR, 'multi_imv_combined_xgboost.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Combined figure saved to: {output_path}")
    
    # Also save as PDF for publication quality
    output_pdf = os.path.join(FIGURES_DIR, 'multi_imv_combined_xgboost.pdf')
    plt.savefig(output_pdf, dpi=300, bbox_inches='tight', format='pdf')
    print(f"✓ PDF version saved to: {output_pdf}")
    
    plt.close()
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS (XGBoost Model)")
    print("="*80)
    
    print("\nNursery Dataset:")
    print(f"  Matrix shape: {nursery_matrix.shape}")
    print(f"  Mean IMV (off-diagonal): {np.mean(nursery_matrix.values[np.eye(len(nursery_matrix), dtype=bool) == False]):.4f}")
    print(f"  Max IMV: {np.max(nursery_matrix.values[np.eye(len(nursery_matrix), dtype=bool) == False]):.4f}")
    
    print("\nCar Evaluation Dataset:")
    print(f"  Matrix shape: {car_matrix.shape}")
    print(f"  Mean IMV (off-diagonal): {np.mean(car_matrix.values[np.eye(len(car_matrix), dtype=bool) == False]):.4f}")
    print(f"  Max IMV: {np.max(car_matrix.values[np.eye(len(car_matrix), dtype=bool) == False]):.4f}")
    
    print("\nDry Bean Dataset:")
    print(f"  Matrix shape: {bean_matrix.shape}")
    print(f"  Mean IMV (off-diagonal): {np.mean(bean_matrix.values[np.eye(len(bean_matrix), dtype=bool) == False]):.4f}")
    print(f"  Max IMV: {np.max(bean_matrix.values[np.eye(len(bean_matrix), dtype=bool) == False]):.4f}")
    
    print("\n" + "="*80)
    print("✓ Figure generation complete!")
    print("="*80)


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create Combined Multi-IMV Figure')
    args = parser.parse_args()
    
    create_combined_figure()


if __name__ == '__main__':
    main()
