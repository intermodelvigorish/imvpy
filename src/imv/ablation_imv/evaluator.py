"""
Ablation IMV: InterModel Vigorish for Deep Learning Ablation Studies

This module applies IMV to deep learning models, specifically transformers,
to quantify the information contribution of architectural components through
ablation studies. Common ablations include:
- Layer reduction (removing transformer layers)
- Component removal (attention heads, feedforward layers)
- Architecture simplification

Supports automatic GPU detection: CUDA (NVIDIA) > MPS (Apple Silicon) > CPU
"""

import numpy as np
import pandas as pd
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tqdm.auto import tqdm
torch = None


def _require_torch():
    """Import PyTorch only when training functionality is actually requested."""
    global torch
    if torch is None:
        try:
            import torch as torch_module
        except ImportError as exc:
            raise ImportError(
                "Training with AblationIMV requires PyTorch; install imv[deep-learning]"
            ) from exc
        torch = torch_module
    return torch

# Import shared IMV functions from core module
from ..utils.core import ll, get_w, calculate_imv


class AblationIMV:
    ll = staticmethod(ll)
    get_w = staticmethod(get_w)
    calculate_imv = staticmethod(calculate_imv)
    """
    Ablation IMV for deep learning models (especially NLP transformers).
    
    This class performs ablation studies by training models with different
    architectural modifications (reduced layers, removed components) and
    calculating IMV to measure the impact of each modification.
    
    The class automatically detects and uses GPU if available, otherwise uses CPU.
    
    Parameters
    ----------
    random_seed : int, default=42
        Random seed for reproducibility
    """
    
    def __init__(self, random_seed=42):
        _require_torch()
        self.random_seed = random_seed
        
        # Automatic device detection: CUDA > MPS > CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"Using device: CUDA GPU")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print(f"Using device: Apple Silicon GPU (MPS)")
            print("Note: MPS provides GPU acceleration on M1/M2/M3 Macs")
        else:
            self.device = torch.device("cpu")
            print("Using device: CPU")
            print("Note: No GPU available, training will be slower")
        
        print(f"PyTorch version: {torch.__version__}")
    
    def set_seed(self, seed=None):
        """
        Set random seed for reproducibility.
        
        Parameters
        ----------
        seed : int, optional
            Random seed. If None, uses self.random_seed
        """
        if seed is None:
            seed = self.random_seed
            
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Set seeds for all available GPU backends
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # MPS uses the same manual_seed as CPU
            torch.mps.manual_seed(seed)
    
    # Note: ll(), get_w(), and calculate_imv() are now imported from imv.core
    # No need to redefine them here - this eliminates code duplication!
    
    @staticmethod
    def reduce_bert_layers(model, num_layers_to_keep):
        """
        Reduce the number of transformer layers in a DistilBERT model.
        
        Performs layer ablation by removing transformer layers from the end of the
        network. This is a common ablation technique to measure the importance of
        model depth.
        
        Parameters
        ----------
        model : transformers.DistilBertForSequenceClassification
            The DistilBERT model to modify (or similar architecture)
        num_layers_to_keep : int
            Number of layers to keep counting from the beginning.
            Must be >= 1 and <= original number of layers.
            
        Returns
        -------
        model
            Modified model with reduced layers (in-place modification)
            
        Example:
            >>> from transformers import DistilBertForSequenceClassification
            >>> model = DistilBertForSequenceClassification.from_pretrained(
            ...     "distilbert-base-uncased", num_labels=2
            ... )
            >>> # DistilBERT has 6 layers by default, reduce to 3
            >>> model = AblationIMV.reduce_bert_layers(model, num_layers_to_keep=3)
            >>> print(len(model.distilbert.transformer.layer))  # Output: 3
            
        Note:
            - Modifies model in-place but also returns it for convenience
            - Works with DistilBERT; may need adaptation for BERT, RoBERTa, etc.
            - Keep at least 1 layer for meaningful model function
            - Earlier layers capture more basic features; later layers capture complex patterns
        """
        torch_module = _require_torch()
        if not isinstance(num_layers_to_keep, int) or num_layers_to_keep < 1:
            raise ValueError("num_layers_to_keep must be a positive integer")
        available = len(model.distilbert.transformer.layer)
        if num_layers_to_keep > available:
            raise ValueError(f"cannot keep {num_layers_to_keep} layers; model has {available}")
        model.distilbert.transformer.layer = torch_module.nn.ModuleList(
            model.distilbert.transformer.layer[:num_layers_to_keep]
        )
        return model
    
    def train_and_evaluate(self, model, train_dataloader, test_dataloader, 
                          num_epochs=3, lr=2e-5, optimizer_class=None, 
                          scheduler_fn=None, seed=None, verbose=True):
        """
        Train and evaluate a model with automatic GPU/CPU detection.
        
        Parameters
        ----------
        model : torch.nn.Module
            PyTorch model to train
        train_dataloader : DataLoader
            Training data loader
        test_dataloader : DataLoader
            Test data loader
        num_epochs : int, default=3
            Number of training epochs
        lr : float, default=2e-5
            Learning rate
        optimizer_class : class, optional
            Optimizer class (e.g., AdamW). If None, uses torch.optim.Adam
        scheduler_fn : callable, optional
            Function to create learning rate scheduler
        seed : int, optional
            Random seed for this run
        verbose : bool, default=True
            Print training progress
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'model': trained model
            - 'test_predictions': DataFrame with predictions and labels
            - 'test_accuracy': float
            - 'test_precision': float
            - 'test_recall': float
        """
        self.set_seed(seed)
        
        # Setup optimizer
        if optimizer_class is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        else:
            optimizer = optimizer_class(model.parameters(), lr=lr)
        
        # Setup scheduler
        if scheduler_fn is not None:
            num_training_steps = num_epochs * len(train_dataloader)
            lr_scheduler = scheduler_fn(optimizer=optimizer, num_training_steps=num_training_steps)
        else:
            lr_scheduler = None
        
        # Move model to device (GPU or CPU)
        model.to(self.device)
        
        # Training loop
        model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            all_labels = []
            all_preds = []
            
            iterator = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}") if verbose else train_dataloader
            
            for batch in iterator:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = model(**batch)
                loss = outputs.loss
                logits = outputs.logits
                
                total_loss += loss.item()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if lr_scheduler is not None:
                    lr_scheduler.step()
                
                preds = logits.argmax(dim=-1).detach().cpu().numpy()
                labels = batch['labels'].cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels)
            
            if verbose:
                train_acc = accuracy_score(all_labels, all_preds)
                train_precision = precision_score(all_labels, all_preds, average='binary')
                train_recall = recall_score(all_labels, all_preds, average='binary')
                avg_loss = total_loss / len(train_dataloader)
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, "
                      f"Accuracy: {train_acc:.4f}, Precision: {train_precision:.4f}, "
                      f"Recall: {train_recall:.4f}")
        
        # Evaluation
        model.eval()
        all_test_labels = []
        all_test_preds = []
        all_test_logits = []
        
        with torch.no_grad():
            iterator = tqdm(test_dataloader, desc="Evaluating") if verbose else test_dataloader
            for batch in iterator:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = model(**batch)
                logits = outputs.logits
                preds = logits.argmax(dim=-1)
                
                all_test_preds.extend(preds.cpu().numpy())
                all_test_labels.extend(batch['labels'].cpu().numpy())
                all_test_logits.extend(torch.softmax(logits, dim=-1).cpu().numpy())
        
        test_acc = accuracy_score(all_test_labels, all_test_preds)
        test_precision = precision_score(all_test_labels, all_test_preds, average='binary')
        test_recall = recall_score(all_test_labels, all_test_preds, average='binary')
        
        if verbose:
            print(f"Test Accuracy: {test_acc:.4f}, Precision: {test_precision:.4f}, "
                  f"Recall: {test_recall:.4f}")
        
        # Create predictions DataFrame
        df = pd.DataFrame(all_test_logits, columns=['Negative Probability', 'Positive Probability'])
        df['True Label'] = all_test_labels
        df['Predicted Label'] = all_test_preds
        
        return {
            'model': model,
            'test_predictions': df,
            'test_accuracy': test_acc,
            'test_precision': test_precision,
            'test_recall': test_recall
        }
    
    @staticmethod
    def calculate_imv_matrix(predictions_dict, target_column='True Label', 
                            prob_column='Positive Probability'):
        """
        Calculate pairwise IMV comparison matrix for multiple model variants.
        
        Creates a matrix where element (i,j) represents the IMV of model i compared
        to model j (how much better model i is than model j). Useful for comparing
        multiple ablation variants simultaneously.
        
        Parameters
        ----------
        predictions_dict : dict of {str: pd.DataFrame}
            Dictionary mapping model variant names to their prediction DataFrames.
            Each DataFrame must contain target_column and prob_column.
            
            Example structure:
                {
                    '6-layer': df_6layer,
                    '4-layer': df_4layer,
                    '2-layer': df_2layer
                }
                
        target_column : str, default='True Label'
            Name of the column containing true binary labels
        prob_column : str, default='Positive Probability'
            Name of the column containing predicted probabilities for positive class
            
        Returns
        -------
        pd.DataFrame, shape (n_models, n_models)
            Pairwise IMV comparison matrix where:
            - Rows represent "enhanced" models
            - Columns represent "basic" models
            - Element (i,j) = IMV comparing model i to model j
            - Diagonal elements are 0 (model vs itself)
            
        Interpretation:
            - IMV(i,j) > 0: Model i has more information than model j
            - IMV(i,j) = 0: Models are equivalent
            - IMV(i,j) < 0: Model j is better than model i
            - The matrix is directional and generally not antisymmetric
            
        Example:
            >>> predictions = {
            ...     'Full': df_full,
            ...     'Ablated-Layer': df_ablated,
            ...     'Baseline': df_baseline
            ... }
            >>> imv_matrix = AblationIMV.calculate_imv_matrix(predictions)
            >>> print(imv_matrix)
            #                   Full  Ablated-Layer  Baseline
            # Full             0.000          0.045     0.120
            # Ablated-Layer   -0.045          0.000     0.068
            # Baseline        -0.120         -0.068     0.000
        """
        model_names = list(predictions_dict.keys())
        if not model_names:
            raise ValueError("predictions_dict cannot be empty")
        n_models = len(model_names)
        
        # Initialize IMV matrix
        imv_matrix = pd.DataFrame(
            np.zeros((n_models, n_models)),
            columns=model_names,
            index=model_names
        )
        
        # Get true labels (same for all models)
        first = predictions_dict[model_names[0]]
        required = {target_column, prob_column}
        if not required.issubset(first.columns):
            raise ValueError(f"prediction frames must contain {sorted(required)}")
        y = first[target_column].to_numpy()
        for name in model_names:
            frame = predictions_dict[name]
            if not required.issubset(frame.columns):
                raise ValueError(f"prediction frame {name!r} must contain {sorted(required)}")
            if len(frame) != len(y) or not np.array_equal(frame[target_column].to_numpy(), y):
                raise ValueError("all prediction frames must have identical aligned labels")
        
        # Calculate pairwise IMV
        for i, model_i in enumerate(model_names):
            for j, model_j in enumerate(model_names):
                if i == j:
                    continue
                
                y_enhanced = predictions_dict[model_i][prob_column].values
                y_basic = predictions_dict[model_j][prob_column].values
                
                # Use shared calculate_imv() from core module
                imv_value = calculate_imv(y_basic, y_enhanced, y)
                imv_matrix.iloc[i, j] = imv_value
        
        return imv_matrix
    
    @staticmethod
    def average_imv_matrices(matrices_list):
        """
        Average multiple IMV matrices across random seeds or folds.
        
        Combines IMV matrices from multiple runs to get stable estimates
        and reduce variance from random initialization. Useful for getting
        reliable ablation study results.
        
        Parameters
        ----------
        matrices_list : list of pd.DataFrame
            List of IMV matrices to average. All matrices must have the same
            shape, index, and columns (same model variant names).
            
        Returns
        -------
        pd.DataFrame
            Averaged IMV matrix with same structure as input matrices
            
        Raises:
            ValueError: If matrices_list is empty
            
        Example:
            >>> # Run ablation study with multiple seeds
            >>> matrices = []
            >>> for seed in [42, 123, 456, 789, 999]:
            ...     # Train models with different seeds
            ...     predictions = run_ablation_study(seed=seed)
            ...     imv_mat = AblationIMV.calculate_imv_matrix(predictions)
            ...     matrices.append(imv_mat)
            >>> 
            >>> # Get stable averaged results
            >>> avg_matrix = AblationIMV.average_imv_matrices(matrices)
            >>> print(avg_matrix)
            
        Note:
            - Element-wise averaging (not matrix algebra)
            - Preserves index and column labels from first matrix
            - Recommended: Use at least 3-5 seeds for stable estimates
            - Standard deviation can be computed separately with np.std()
        """
        if not matrices_list:
            raise ValueError("matrices_list cannot be empty")
        first = matrices_list[0]
        for matrix in matrices_list:
            if not isinstance(matrix, pd.DataFrame):
                raise TypeError("all matrices must be pandas DataFrames")
            if not matrix.index.equals(first.index) or not matrix.columns.equals(first.columns):
                raise ValueError("all matrices must have identical index and columns")
        
        # Stack and average
        stacked = np.stack([m.values for m in matrices_list])
        averaged = np.mean(stacked, axis=0)
        
        # Create DataFrame with same structure
        result = pd.DataFrame(
            averaged,
            index=matrices_list[0].index,
            columns=matrices_list[0].columns
        )
        
        return result
