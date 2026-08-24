"""
Test file for Ablation IMV functionality
Replicates the transformer ablation study from Ablate_IMV_Latest.ipynb

NOTE: This test requires PyTorch and transformers library.
      It will automatically use GPU if available, otherwise CPU (slower).
"""

import warnings

warnings.filterwarnings("ignore")
import os
import sys

import pytest

# Figures belong beside the tests, never in whatever directory pytest was
# launched from.
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import torch
    from datasets import load_dataset
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, DistilBertForSequenceClassification, get_scheduler
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print("ERROR: Missing dependencies for ablation IMV tests")
    print("Please install: pip install torch transformers datasets")
    print(f"Import error: {e}")
    DEPENDENCIES_AVAILABLE = False

from imv import AblationIMV  # Updated import for reorganized package structure
from imv.utils import save_figure

pytestmark = [
    pytest.mark.slow,
    pytest.mark.deep_learning,
    pytest.mark.skipif(not DEPENDENCIES_AVAILABLE, reason="deep-learning extras not installed"),
]


def check_dependencies():
    """Check if all required dependencies are available"""
    if not DEPENDENCIES_AVAILABLE:
        print("\n" + "="*80)
        print("DEPENDENCIES NOT AVAILABLE")
        print("="*80)
        print("\nTo run ablation IMV tests, install:")
        print("  pip install torch transformers datasets")
        print("\nThese tests will be skipped.")
        return False
    return True


def prepare_imdb_data(sample_size=None):
    """
    Load and prepare IMDb dataset for sentiment analysis.
    
    Parameters
    ----------
    sample_size : int, optional
        If specified, use only this many samples for faster testing
        
    Returns
    -------
    tuple
        (train_dataset, test_dataset, tokenizer)
    """
    print("Loading IMDb dataset...")
    imdb = load_dataset('imdb')
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased', lower=True)
    
    def clean_text(text):
        """Remove HTML tags and extra whitespace"""
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def preprocess_function(examples):
        """Tokenize and clean text"""
        examples['text'] = [clean_text(text) for text in examples['text']]
        return tokenizer(examples['text'], padding='max_length', truncation=True)
    
    print("Preprocessing dataset...")
    encoded_imdb = imdb.map(preprocess_function, batched=True)
    encoded_imdb = encoded_imdb.remove_columns(["text"])
    # Rename 'label' to 'labels' to match model expectations
    encoded_imdb = encoded_imdb.rename_column("label", "labels")
    encoded_imdb.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    train_dataset = encoded_imdb['train']
    test_dataset = encoded_imdb['test']
    
    # Sample if requested (for faster testing)
    if sample_size is not None:
        train_dataset = train_dataset.select(range(min(sample_size, len(train_dataset))))
        test_dataset = test_dataset.select(range(min(sample_size // 5, len(test_dataset))))
        print(f"Using sample: {len(train_dataset)} train, {len(test_dataset)} test")
    else:
        print(f"Using full dataset: {len(train_dataset)} train, {len(test_dataset)} test")
    
    return train_dataset, test_dataset, tokenizer


def test_core_imv_functions():
    """Test the core IMV calculation functions"""
    print("\n" + "="*80)
    print("TEST 1: Core IMV Calculation Functions")
    print("="*80)
    
    # Test ll function
    print("\nTesting ll (log-likelihood)...")
    x = np.array([1, 0, 1, 1, 0])
    p = np.array([0.8, 0.2, 0.7, 0.9, 0.3])
    ll_result = AblationIMV.ll(x, p)
    print(f"✓ ll result: {ll_result:.6f}")
    
    # Test get_w function
    print("\nTesting get_w (weight calculation)...")
    a = 0.6
    w = AblationIMV.get_w(a)
    print(f"✓ get_w for a={a}: {w:.6f}")
    
    # Test calculate_imv
    print("\nTesting calculate_imv...")
    y = np.array([1, 0, 1, 1, 0, 1])
    y_basic = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    y_enhanced = np.array([0.8, 0.2, 0.7, 0.9, 0.3, 0.85])
    imv = AblationIMV.calculate_imv(y_basic, y_enhanced, y)
    print(f"✓ IMV score: {imv:.6f}")
    
    print("\n✓ Core function tests passed!")


def test_device_detection():
    """Test automatic GPU/CPU detection"""
    print("\n" + "="*80)
    print("TEST 2: Automatic Device Detection")
    print("="*80)
    
    evaluator = AblationIMV(random_seed=42)
    
    print(f"\nDevice detected: {evaluator.device}")
    print(f"Device type: {evaluator.device.type}")
    
    # Check all possible backends
    print("\nAvailable backends:")
    print(f"  CUDA (NVIDIA GPU): {torch.cuda.is_available()}")
    if hasattr(torch.backends, 'mps'):
        print(f"  MPS (Apple Silicon): {torch.backends.mps.is_available()}")
    else:
        print("  MPS (Apple Silicon): False (PyTorch version too old)")
    print("  CPU: True (always available)")
    
    if evaluator.device.type == "cuda":
        print(f"\n✓ Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
        print("  Training will use CUDA acceleration")
    elif evaluator.device.type == "mps":
        print("\n✓ Using Apple Silicon GPU (MPS)")
        print("  Training will use Metal Performance Shaders acceleration")
        print("  Note: MPS support requires PyTorch >= 1.12")
    else:
        print("\n⚠ Using CPU")
        print("  Training will be slower but functional")
    
    print("\n✓ Device detection test passed!")


def test_layer_reduction():
    """Test reducing layers in DistilBERT"""
    print("\n" + "="*80)
    print("TEST 3: Layer Reduction")
    print("="*80)
    
    if not check_dependencies():
        return
    
    print("\nLoading DistilBERT model...")
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', 
        num_labels=2
    )
    
    original_layers = len(model.distilbert.transformer.layer)
    print(f"Original model has {original_layers} layers")
    
    # Test reduction
    num_layers_to_keep = 3
    model_reduced = AblationIMV.reduce_bert_layers(model, num_layers_to_keep)
    
    reduced_layers = len(model_reduced.distilbert.transformer.layer)
    print(f"Reduced model has {reduced_layers} layers")
    
    assert reduced_layers == num_layers_to_keep, "Layer reduction failed"
    print(f"\n✓ Successfully reduced from {original_layers} to {reduced_layers} layers!")


def test_small_training_run(quick=True):
    """
    Test training with a very small dataset sample.
    This is just to verify the training loop works.
    """
    print("\n" + "="*80)
    print("TEST 4: Small Training Run (Verification Only)")
    print("="*80)
    
    if not check_dependencies():
        return
    
    # Use very small sample for testing
    sample_size = 100 if quick else 1000
    print(f"\nUsing tiny sample ({sample_size} examples) to verify training works...")
    print("Note: This is NOT for actual research, just testing the code!")
    
    # Load small sample
    train_dataset, test_dataset, tokenizer = prepare_imdb_data(sample_size=sample_size)
    
    # Create dataloaders with small batch size
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8)
    
    # Create evaluator
    evaluator = AblationIMV(random_seed=42)
    
    # Create model
    print("\nCreating model...")
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=2
    )
    
    # Train for just 1 epoch (very fast)
    print("\nTraining for 1 epoch (just to test)...")
    
    def create_scheduler(optimizer, num_training_steps):
        return get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps
        )
    
    results = evaluator.train_and_evaluate(
        model=model,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        num_epochs=1,
        lr=2e-5,
        optimizer_class=AdamW,
        scheduler_fn=create_scheduler,
        seed=42,
        verbose=True
    )
    
    print("\n✓ Training completed successfully!")
    print(f"  Test accuracy: {results['test_accuracy']:.4f}")
    print(f"  Predictions shape: {results['test_predictions'].shape}")


def test_imv_matrix_calculation():
    """Test IMV matrix calculation with mock predictions"""
    print("\n" + "="*80)
    print("TEST 5: IMV Matrix Calculation")
    print("="*80)
    
    # Create mock predictions for different model variants
    n_samples = 1000
    np.random.seed(42)
    
    # Simulate predictions from different models
    predictions_dict = {
        'Original': pd.DataFrame({
            'Negative Probability': np.random.random(n_samples) * 0.3,
            'Positive Probability': np.random.random(n_samples) * 0.7 + 0.3,
            'True Label': np.random.choice([0, 1], n_samples),
            'Predicted Label': np.random.choice([0, 1], n_samples)
        }),
        '3Layers': pd.DataFrame({
            'Negative Probability': np.random.random(n_samples) * 0.4,
            'Positive Probability': np.random.random(n_samples) * 0.6 + 0.4,
            'True Label': np.random.choice([0, 1], n_samples),
            'Predicted Label': np.random.choice([0, 1], n_samples)
        }),
        'NoAttention': pd.DataFrame({
            'Negative Probability': np.random.random(n_samples) * 0.5,
            'Positive Probability': np.random.random(n_samples) * 0.5 + 0.5,
            'True Label': np.random.choice([0, 1], n_samples),
            'Predicted Label': np.random.choice([0, 1], n_samples)
        })
    }
    
    # Make sure all have same labels
    true_labels = predictions_dict['Original']['True Label'].values
    for model_name in predictions_dict:
        predictions_dict[model_name]['True Label'] = true_labels
    
    print("\nCalculating IMV matrix for 3 model variants...")
    imv_matrix = AblationIMV.calculate_imv_matrix(predictions_dict)
    
    print("\nIMV Matrix:")
    print(imv_matrix)
    
    # Create heatmap
    print("\nCreating heatmap visualization...")
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(imv_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                   center=0, ax=ax)
        ax.set_title('Ablation IMV Matrix\n(row model vs column model)')
        plt.tight_layout()
        save_figure(fig, os.path.join(FIGURES_DIR, 'test_ablation_imv_matrix'))
        print("✓ Heatmap saved to: test_ablation_imv_matrix.png")
        plt.close()
    except Exception as e:
        print(f"Warning: Could not create heatmap: {e}")
    
    print("\n✓ IMV matrix calculation test passed!")


def test_average_matrices():
    """Test averaging IMV matrices from multiple seeds"""
    print("\n" + "="*80)
    print("TEST 6: Averaging IMV Matrices")
    print("="*80)
    
    # Create mock IMV matrices from different seeds
    model_names = ['Original', '3Layers', 'NoAttention']
    matrices = []
    
    for seed in [42, 43, 44]:
        np.random.seed(seed)
        matrix = pd.DataFrame(
            np.random.randn(3, 3) * 0.1,
            index=model_names,
            columns=model_names
        )
        # Zero out diagonal
        for i in range(3):
            matrix.iloc[i, i] = 0
        matrices.append(matrix)
        print(f"\nMatrix for seed {seed}:")
        print(matrix)
    
    # Average
    print("\nAveraging matrices...")
    avg_matrix = AblationIMV.average_imv_matrices(matrices)
    
    print("\nAveraged IMV Matrix:")
    print(avg_matrix)
    
    print("\n✓ Matrix averaging test passed!")


def run_all_tests(quick_mode=True):
    """
    Run all tests
    
    Parameters
    ----------
    quick_mode : bool
        If True, use minimal data for fast testing
    """
    print("\n" + "="*80)
    print("Ablation IMV Package Test Suite")
    print("="*80)
    
    if not check_dependencies():
        print("\nSkipping tests due to missing dependencies")
        return
    
    # Always run these
    test_core_imv_functions()
    test_device_detection()
    test_layer_reduction()
    test_imv_matrix_calculation()
    test_average_matrices()
    
    # Optional: small training run
    if quick_mode:
        print("\n" + "="*80)
        print("Skipping full training test in quick mode")
        print("To run small training test, use: python test_ablate_imv.py --train")
        print("="*80)
    else:
        test_small_training_run(quick=True)
    
    print("\n" + "="*80)
    print("All tests completed!")
    print("="*80)
    print("\nNOTE: For actual ablation studies:")
    print("  1. Use full datasets (not samples)")
    print("  2. Train for multiple epochs (3-5)")
    print("  3. Use multiple seeds for robustness")
    print("  4. GPU highly recommended for reasonable training time")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Ablation IMV package')
    parser.add_argument('--train', action='store_true',
                       help='Include small training test (slower)')
    parser.add_argument('--test', type=str,
                       choices=['core', 'device', 'layers', 'matrix', 'average', 'train', 'all'],
                       default='all', help='Which test to run')
    
    args = parser.parse_args()
    
    if not check_dependencies():
        sys.exit(1)
    
    if args.test == 'core':
        test_core_imv_functions()
    elif args.test == 'device':
        test_device_detection()
    elif args.test == 'layers':
        test_layer_reduction()
    elif args.test == 'matrix':
        test_imv_matrix_calculation()
    elif args.test == 'average':
        test_average_matrices()
    elif args.test == 'train':
        test_small_training_run(quick=True)
    else:  # 'all'
        run_all_tests(quick_mode=not args.train)
