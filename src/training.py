"""Contains the core adversarial training loop for EdgeMask-DG*."""

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch_geometric.utils import coalesce

from .utils import evaluate

def train_edgemask(
    task_net, mask_net, mask_proj,
    data_list, source_keys,
    epochs, lambda_sparsity,
    descent_steps, ascent_steps,
    lr, weight_decay, device,
    use_amp_train,
    precomputed_spectral_edges=None,
    spectral_edge_sample_ratio=1.0,
    precomputed_knn_edges=None,
    knn_edge_sample_ratio=1.0,
    val_data=None,
    early_stopping_patience=10
):
    """
    Executes the adversarial training loop for EdgeMask-DG*.

    This involves alternating between descending on the TaskNet loss and
    ascending on the MaskNet objective.
    """
    scaler_task = GradScaler(enabled=use_amp_train)
    scaler_mask = GradScaler(enabled=use_amp_train)
    opt_task = torch.optim.Adam(task_net.parameters(), lr=lr, weight_decay=weight_decay)
    opt_mask = torch.optim.Adam(list(mask_net.parameters()) + list(mask_proj.parameters()), lr=lr)
    
    task_net.to(device)
    mask_net.to(device)
    mask_proj.to(device)
    
    best_val_metric = -1
    epochs_no_improve = 0
    best_task_net_state, best_mask_net_state, best_mask_proj_state = None, None, None
    
    pbar_epochs = tqdm(range(1, epochs + 1), desc="EdgeMask-DG*(GAT) Epochs", leave=False)
    
    for epoch in pbar_epochs:
        task_net.train()
        mask_net.train()
        mask_proj.train()
        epoch_task_loss_sum, epoch_mask_obj_sum, epoch_total_nodes = 0, 0, 0
        
        for _ in range(descent_steps):
            opt_task.zero_grad(set_to_none=True)
            current_batch_task_loss_unscaled, current_batch_nodes = 0, 0
            for key in source_keys:
                if key not in data_list: continue
                data = data_list[key].to(device)
                x_task, original_edge_index, N = data.x, data.edge_index, data.num_nodes
                
                edge_indices_to_combine = [original_edge_index]
                if precomputed_spectral_edges and key in precomputed_spectral_edges:
                    spectral_edges = precomputed_spectral_edges[key]
                    if spectral_edge_sample_ratio < 1.0 and spectral_edges.size(1) > 0:
                        num_sample = int(spectral_edges.size(1) * spectral_edge_sample_ratio)
                        perm = torch.randperm(spectral_edges.size(1), device=device)
                        edge_indices_to_combine.append(spectral_edges[:, perm[:num_sample]])
                    elif spectral_edges.size(1) > 0:
                        edge_indices_to_combine.append(spectral_edges)

                if precomputed_knn_edges and key in precomputed_knn_edges:
                    knn_edges = precomputed_knn_edges[key]
                    if knn_edge_sample_ratio < 1.0 and knn_edges.size(1) > 0:
                        num_sample = int(knn_edges.size(1) * knn_edge_sample_ratio)
                        perm = torch.randperm(knn_edges.size(1), device=device)
                        edge_indices_to_combine.append(knn_edges[:, perm[:num_sample]])
                    elif knn_edges.size(1) > 0:
                        edge_indices_to_combine.append(knn_edges)
                
                combined_edge_index = coalesce(torch.cat(edge_indices_to_combine, dim=1), num_nodes=N)
                
                with torch.no_grad():
                    x_mask_proj = mask_proj(data.x)
                    s = mask_net(x_mask_proj.detach(), combined_edge_index, use_amp_fwd=use_amp_train) if combined_edge_index.numel() > 0 else torch.tensor([], device=device)

                with autocast(enabled=use_amp_train):
                    out = task_net(x_task, combined_edge_index, edge_weight=s if s.numel() > 0 else None)
                    loss = F.cross_entropy(out, data.y)
                
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    scaler_task.scale(loss / len(source_keys)).backward()
                    current_batch_task_loss_unscaled += loss.item() * N
                    current_batch_nodes += N
            
            if current_batch_nodes > 0:
                scaler_task.unscale_(opt_task)
                torch.nn.utils.clip_grad_norm_(task_net.parameters(), 1.0)
                scaler_task.step(opt_task)
                scaler_task.update()
                epoch_task_loss_sum += current_batch_task_loss_unscaled
                epoch_total_nodes += current_batch_nodes
        
        for _ in range(ascent_steps):
            opt_mask.zero_grad(set_to_none=True)
            current_batch_mask_obj_unscaled, current_batch_nodes_mask = 0, 0
            for key in source_keys:
                if key not in data_list: continue
                data = data_list[key].to(device)
                x_task, original_edge_index, N = data.x, data.edge_index, data.num_nodes

                edge_indices_to_combine = [original_edge_index] # Rebuild for this step
                if precomputed_spectral_edges and key in precomputed_spectral_edges:
                    edge_indices_to_combine.append(precomputed_spectral_edges[key])
                if precomputed_knn_edges and key in precomputed_knn_edges:
                    edge_indices_to_combine.append(precomputed_knn_edges[key])
                combined_edge_index = coalesce(torch.cat(edge_indices_to_combine, dim=1), num_nodes=N)

                with autocast(enabled=use_amp_train):
                    x_mask_proj = mask_proj(data.x)
                    s_mask_step = mask_net(x_mask_proj, combined_edge_index, use_amp_fwd=use_amp_train) if combined_edge_index.numel() > 0 else torch.tensor([], device=device)
                    
                    task_net.eval()
                    with torch.no_grad():
                        out_eval = task_net(x_task, combined_edge_index, edge_weight=s_mask_step if s_mask_step.numel() > 0 else None)
                    task_net.train()
                    
                    task_loss_term = F.cross_entropy(out_eval, data.y)
                    sparsity_term = torch.mean(s_mask_step) if s_mask_step.numel() > 0 else torch.tensor(0.0, device=device)
                    mask_objective = -task_loss_term + lambda_sparsity * sparsity_term
                
                if not (torch.isnan(mask_objective) or torch.isinf(mask_objective)):
                    scaler_mask.scale(mask_objective / len(source_keys)).backward()
                    current_batch_mask_obj_unscaled += mask_objective.item() * N
                    current_batch_nodes_mask += N

            if current_batch_nodes_mask > 0:
                scaler_mask.unscale_(opt_mask)
                torch.nn.utils.clip_grad_norm_(list(mask_net.parameters()) + list(mask_proj.parameters()), 1.0)
                scaler_mask.step(opt_mask)
                scaler_mask.update()
                epoch_mask_obj_sum += current_batch_mask_obj_unscaled
        
        avg_task_loss = (epoch_task_loss_sum / epoch_total_nodes) if epoch_total_nodes > 0 else 0
        avg_mask_obj = (epoch_mask_obj_sum / epoch_total_nodes) if epoch_total_nodes > 0 else 0
        pbar_epochs.set_postfix(task_loss=f"{avg_task_loss:.4f}", mask_obj=f"{avg_mask_obj:.4f}")

        if val_data is not None and early_stopping_patience > 0:
            val_acc, _, _ = evaluate(task_net, val_data, device, use_amp_eval=use_amp_train)
            if val_acc > best_val_metric:
                best_val_metric = val_acc
                epochs_no_improve = 0
                best_task_net_state = task_net.state_dict()
                best_mask_net_state = mask_net.state_dict()
                best_mask_proj_state = mask_proj.state_dict()
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch} with best val_acc: {best_val_metric:.4f}")
                break

    if best_task_net_state is not None:
        print(f"Loading best model (val_acc: {best_val_metric:.4f})")
        task_net.load_state_dict(best_task_net_state)
        mask_net.load_state_dict(best_mask_net_state)
        mask_proj.load_state_dict(best_mask_proj_state)

    return task_net, mask_net, mask_proj