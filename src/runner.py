"""High-level experiment runner for the artificial shift OOD benchmarks."""

import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from .datasets import get_artificial_spurious_data_for_ood
from .feature_engineering import precompute_cluster_edges, precompute_knn_edges
from .models import TaskNet, Projection, MaskNet
from .training import train_edgemask
from .utils import evaluate

def run_experiments_artificial_shift(dataset_name, args, device, use_amp):
    """
    Manages the full experimental pipeline for a given artificial shift dataset.

    This includes data generation, feature engineering, training, and evaluation
    across multiple OOD scenarios.

    Args:
        dataset_name (str): The name of the dataset (e.g., 'Cora').
        args: Parsed command-line arguments.
        device: The PyTorch device to use.
        use_amp (bool): Whether to use Automatic Mixed Precision.

    Returns:
        A dictionary containing summary statistics of the experiment.
    """
    print(f"Running {dataset_name} OOD (Artificial Shift) experiments on device: {device}, AMP: {use_amp}")
    num_test_scenarios = 8
    dataset_scenarios_data = get_artificial_spurious_data_for_ood(
        dataset_name=dataset_name,
        root_dir=args.data_dir,
        num_test_scenarios=num_test_scenarios
    )

    all_results = []
    target_accuracies, target_micro_f1s, target_macro_f1s = [], [], []

    for scenario_idx in range(num_test_scenarios):
        print(f"\n--- {dataset_name} OOD Scenario {scenario_idx + 1}/{num_test_scenarios} (Seed: {args.seed}) ---")
        
        data_list_cpu, source_keys, val_key, target_key, num_features, num_classes = dataset_scenarios_data[scenario_idx]
        data_list_device = {k: v.to(device) for k, v in data_list_cpu.items()}
        source_data_for_precompute = {k: data_list_device[k] for k in source_keys if k in data_list_device}
        
        current_cache_dir = Path(args.cache_dir)
        precomputed_spectral = precompute_cluster_edges(
            source_data_for_precompute, source_keys, args.spectral_k, args.spectral_add_ratio,
            device, args.seed, args.force_recompute, current_cache_dir
        )
        precomputed_knn = precompute_knn_edges(
            source_data_for_precompute, source_keys, args.knn_k, args.knn_metric,
            args.use_faiss_knn, device, args.force_recompute, current_cache_dir
        )

        task_net = TaskNet(num_features, args.hidden_dim, args.num_layers, num_classes, args.gat_heads, args.gat_dropout)
        mask_proj = Projection(num_features, args.mask_proj_dim)
        mask_net = MaskNet(args.mask_proj_dim, args.hidden_dim)
        
        start_time = time.time()
        val_data = data_list_device.get(val_key)

        trained_task_net, _, _ = train_edgemask(
            task_net, mask_net, mask_proj, data_list_device, source_keys,
            args.epochs, args.lambda_sparsity, args.descent_steps, args.ascent_steps,
            args.lr, args.weight_decay, device, use_amp,
            precomputed_spectral_edges=precomputed_spectral, spectral_edge_sample_ratio=args.emdg_star_spectral_sample_ratio,
            precomputed_knn_edges=precomputed_knn, knn_edge_sample_ratio=args.emdg_star_knn_sample_ratio,
            val_data=val_data, early_stopping_patience=args.early_stopping_patience
        )
        train_time = time.time() - start_time

        acc, micro_f1, macro_f1 = evaluate(trained_task_net, data_list_device[target_key], device, use_amp_eval=use_amp)
        
        method_label = "EdgeMask-DG*(GAT,Spec+kNN)"
        print(f"{dataset_name} Scen{scenario_idx+1} {method_label} -> Acc:{acc:.4f} MiF1:{micro_f1:.4f} MaF1:{macro_f1:.4f} Time:{train_time:.2f}s")
        
        all_results.append({
            'scenario': scenario_idx + 1, 'method': method_label, 'acc': acc,
            'micro_f1': micro_f1, 'macro_f1': macro_f1, 'train_time': train_time
        })
        target_accuracies.append(acc)
        target_micro_f1s.append(micro_f1)
        target_macro_f1s.append(macro_f1)

        if device.type == 'cuda':
            del data_list_device, source_data_for_precompute, val_data
            del task_net, mask_proj, mask_net, trained_task_net
            if precomputed_spectral: del precomputed_spectral
            if precomputed_knn: del precomputed_knn
            torch.cuda.empty_cache()

    print("\n" + "="*30 + f"\n {dataset_name} OOD Experiment Summary (Seed: {args.seed}) \n" + "="*30)
    df_results = pd.DataFrame(all_results)
    
    results_dir_path = Path(args.results_dir)
    results_dir_path.mkdir(parents=True, exist_ok=True)
    results_path = results_dir_path / f'{dataset_name.lower()}_ood_results_seed{args.seed}.csv'
    df_results.to_csv(results_path, index=False, float_format="%.4f")
    print(f"{dataset_name} OOD scenario results saved to {results_path}")

    summary_stats = {}
    if target_accuracies:
        min_acc, avg_acc, std_acc = np.min(target_accuracies), np.mean(target_accuracies), np.std(target_accuracies)
        avg_micro_f1, std_micro_f1 = np.mean(target_micro_f1s), np.std(target_micro_f1s)
        avg_macro_f1, std_macro_f1 = np.mean(target_macro_f1s), np.std(target_macro_f1s)

        print(f"Overall {dataset_name} OOD Performance (across {num_test_scenarios} scenarios, Seed: {args.seed}):")
        print(f"  Accuracy: Min={min_acc:.4f}, Avg={avg_acc:.4f} ± {std_acc:.4f}")
        print(f"  Micro-F1: Avg={avg_micro_f1:.4f} ± {std_micro_f1:.4f}")
        print(f"  Macro-F1: Avg={avg_macro_f1:.4f} ± {std_macro_f1:.4f}")
        
        summary_stats = {
            'seed': args.seed, 'dataset': f'{dataset_name}_OOD', 'min_acc': min_acc,
            'avg_acc': avg_acc, 'std_acc': std_acc, 'avg_micro_f1': avg_micro_f1,
            'std_micro_f1': std_micro_f1, 'avg_macro_f1': avg_macro_f1, 'std_macro_f1': std_macro_f1
        }
    else:
        print(f"No {dataset_name} results to summarize.")
    return summary_stats