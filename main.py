"""
Main entry point for running EdgeMask-DG* experiments.

This script handles command-line argument parsing, environment setup (seed, device),
and initiates the appropriate experiment runner based on the target dataset.
"""

import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path

from src import config
from src.utils import set_seed
from src.runner import run_experiments_artificial_shift

try:
    import faiss
except ImportError:
    faiss = None

def main():
    """Parses arguments and runs the selected experiment."""
    parser = argparse.ArgumentParser(description='EdgeMask-DG* (GAT Backbone) OOD Benchmarking')
    
    # --- Experiment Setup ---
    parser.add_argument('--experiment_target', type=str, default='cora', choices=['cora', 'photo'], help='OOD benchmark dataset to run.')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory for datasets.')
    parser.add_argument('--results_dir', type=str, default=config.RESULTS_DIR_DEFAULT, help='Directory to save results.')
    parser.add_argument('--cache_dir', type=str, default=config.CACHE_DIR_DEFAULT, help='Directory for precomputed cache.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--disable_amp', action='store_true', help='Disable Automatic Mixed Precision (AMP).')
    parser.add_argument('--force_recompute', action='store_true', help='Force recomputation of cached items.')

    # --- Training Hyperparameters ---
    parser.add_argument('--epochs', type=int, default=config.EPOCHS, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=config.LR, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=config.WEIGHT_DECAY, help='Weight decay for TaskNet.')
    parser.add_argument('--early_stopping_patience', type=int, default=config.EARLY_STOPPING_PATIENCE, help='Patience for early stopping. 0 to disable.')
    
    # --- Model Hyperparameters ---
    parser.add_argument('--hidden_dim', type=int, default=config.HIDDEN_DIM, help='Hidden dimension size for GNN.')
    parser.add_argument('--num_layers', type=int, default=config.NUM_LAYERS, help='Number of GNN layers.')
    parser.add_argument('--gat_heads', type=int, default=config.GAT_HEADS, help='Number of attention heads for GATConv.')
    parser.add_argument('--gat_dropout', type=float, default=config.GAT_DROPOUT, help='Dropout for GATConv attention.')

    # --- EdgeMask-DG* Specific Hyperparameters ---
    parser.add_argument('--mask_proj_dim', type=int, default=config.MASK_PROJ_DIM, help='Projection dim for MaskNet input.')
    parser.add_argument('--lambda_sparsity', type=float, default=config.LAMBDA_SPARSITY, help='Sparsity regularization for MaskNet.')
    parser.add_argument('--descent_steps', type=int, default=config.DESCENT_STEPS, help='TaskNet steps per epoch.')
    parser.add_argument('--ascent_steps', type=int, default=config.ASCENT_STEPS, help='MaskNet steps per epoch.')

    # --- Feature Engineering Hyperparameters ---
    parser.add_argument('--spectral_k', type=int, default=config.SPECTRAL_K_DEFAULT, help='Num clusters for spectral edges.')
    parser.add_argument('--spectral_add_ratio', type=float, default=config.SPECTRAL_ADD_RATIO_DEFAULT, help='Ratio of spectral edges.')
    parser.add_argument('--emdg_star_spectral_sample_ratio', type=float, default=config.EMDG_STAR_SPECTRAL_SAMPLE_RATIO, help='Ratio of precomputed spectral edges to sample.')
    parser.add_argument('--knn_k', type=int, default=config.KNN_K, help='Num neighbors for kNN edges.')
    parser.add_argument('--knn_metric', type=str, default=config.KNN_METRIC, choices=['cosine', 'euclidean'], help='Metric for kNN.')
    parser.add_argument('--use_faiss_knn', action='store_true', help='Use FAISS for kNN.')
    parser.add_argument('--emdg_star_knn_sample_ratio', type=float, default=config.EMDG_STAR_KNN_SAMPLE_RATIO, help='Ratio of precomputed kNN edges to sample.')

    args = parser.parse_args()

    set_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = (device.type == 'cuda' and not args.disable_amp)

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)

    if args.use_faiss_knn and faiss is None:
        print("Warning: FAISS requested (--use_faiss_knn) but package not found. Falling back to sklearn.")
        args.use_faiss_knn = False

    print(f"Running with Settings (Seed: {args.seed}):")
    print(f"  Experiment Target: {args.experiment_target.upper()}")
    print(f"  Device: {device}, AMP Enabled: {use_amp}")
    print(f"  Data Dir: {args.data_dir}, Results Dir: {args.results_dir}, Cache Dir: {args.cache_dir}")
    print("-" * 30)

    dataset_target_name = args.experiment_target.lower()

    if dataset_target_name in ['cora', 'photo']:
        summary = run_experiments_artificial_shift(
            dataset_name=dataset_target_name.capitalize(),
            args=args,
            device=device,
            use_amp=use_amp
        )
        if summary:
            print("\nSingle seed run completed. Summary already printed above.")
        else:
            print(f"No results from {dataset_target_name.upper()} OOD run.")
    else:
        print(f"Experiment target '{args.experiment_target}' not implemented.")

    print("\nExperiment run finished.")

if __name__ == '__main__':
    main()