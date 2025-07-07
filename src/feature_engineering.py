"""
Functions for creating new graph edges based on node features, including
spectral clustering and k-Nearest Neighbors (kNN). Includes logic for
precomputation and caching.
"""

import time
import hashlib
import numpy as np
import torch
from pathlib import Path
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import kneighbors_graph
from torch_geometric.utils import from_scipy_sparse_matrix, coalesce

from . import config

try:
    import faiss
except ImportError:
    faiss = None
    print("FAISS package not found. kNN will use sklearn.")

def spectral_clustering_edge_adding(data, n_clusters, add_ratio, random_state_seed):
    """
    Generates new edges by connecting nodes within the same spectral cluster.

    Args:
        data: PyG Data object.
        n_clusters (int): The number of clusters for SpectralClustering.
        add_ratio (float): The fraction of potential intra-cluster edges to add.
        random_state_seed (int): Seed for the clustering algorithm.

    Returns:
        A tensor of new edges [2, num_new_edges].
    """
    dev = data.edge_index.device
    N = data.num_nodes
    try:
        features_np = data.x.cpu().numpy()
        features_np = np.nan_to_num(features_np)
        effective_n_clusters = min(n_clusters, N - 1 if N > 1 else 1)
        if effective_n_clusters <= 1:
            return torch.empty((2, 0), dtype=torch.long, device=dev)
        sc = SpectralClustering(
            n_clusters=effective_n_clusters, affinity='rbf',
            assign_labels='discretize', random_state=random_state_seed, n_jobs=-1
        )
        labels = sc.fit_predict(features_np)
        labels = torch.from_numpy(labels).to(torch.long).to(dev)
    except Exception:
        return torch.empty((2, 0), dtype=torch.long, device=dev)

    row, col = [], []
    unique_labels = torch.unique(labels)
    for cluster_id in unique_labels:
        nodes_in_cluster = torch.where(labels == cluster_id)[0]
        if len(nodes_in_cluster) > 1:
            comb = torch.combinations(nodes_in_cluster, r=2)
            row.append(comb[:, 0])
            col.append(comb[:, 1])
            row.append(comb[:, 1])
            col.append(comb[:, 0])
    
    if not row:
        return torch.empty((2, 0), dtype=torch.long, device=dev)
    
    cluster_edge_index = torch.stack([torch.cat(row), torch.cat(col)], dim=0)
    num_cluster_edges = cluster_edge_index.size(1)
    num_edges_to_add = int(num_cluster_edges * add_ratio)
    
    if num_edges_to_add > 0 and num_cluster_edges > 0:
        perm = torch.randperm(num_cluster_edges, device=dev)
        added_edges = cluster_edge_index[:, perm[:num_edges_to_add]]
    else:
        added_edges = torch.empty((2, 0), dtype=torch.long, device=dev)
    
    return added_edges

def precompute_cluster_edges(data_list, source_keys, n_clusters, add_ratio, device, random_state_seed, force_recompute=False, cache_dir_path=None):
    """
    Precomputes and caches spectral clustering edges for all source domain graphs.
    """
    if cache_dir_path is None:
        cache_dir_path = Path(config.CACHE_DIR_DEFAULT)
    else:
        cache_dir_path = Path(cache_dir_path)
    
    keys_str = "_".join(sorted(source_keys))
    cache_filename = f"spectral_edges_k{n_clusters}_r{add_ratio:.2f}_{keys_str}_seed{random_state_seed}.pt"
    if len(cache_filename) > 200:
        keys_hash = hashlib.md5(keys_str.encode()).hexdigest()[:16]
        cache_filename = f"spectral_edges_k{n_clusters}_r{add_ratio:.2f}_hash{keys_hash}_seed{random_state_seed}.pt"
    
    cache_path = cache_dir_path / cache_filename
    if cache_path.exists() and not force_recompute:
        try:
            cluster_edges_map = torch.load(cache_path, map_location='cpu')
            for key_map in cluster_edges_map:
                cluster_edges_map[key_map] = cluster_edges_map[key_map].to(device)
            return cluster_edges_map
        except Exception as e:
            print(f"Warning: Failed to load cached spectral edges ({e}). Recomputing...")

    cluster_edges_map = {}
    for key_iter in source_keys:
        data = data_list[key_iter]
        added_edges = spectral_clustering_edge_adding(data.cpu(), n_clusters, add_ratio, random_state_seed)
        cluster_edges_map[key_iter] = added_edges.to(device)
    
    try:
        cache_dir_path.mkdir(parents=True, exist_ok=True)
        cluster_edges_map_cpu = {k_map: v_map.cpu() for k_map, v_map in cluster_edges_map.items()}
        torch.save(cluster_edges_map_cpu, cache_path)
    except Exception as e:
        print(f"Warning: Failed to save spectral edges to cache ({e}).")
    
    return cluster_edges_map

def knn_feature_edge_adding(data, k, metric, use_faiss_knn_flag, device='cpu'):
    """
    Generates new edges by connecting k-nearest neighbors in the feature space.
    """
    N = data.num_nodes
    features = data.x.cpu().numpy()
    features = np.nan_to_num(features)
    knn_edge_index = torch.empty((2, 0), dtype=torch.long)
    
    try:
        if use_faiss_knn_flag and faiss is not None:
            features_faiss = features.astype('float32')
            d = features_faiss.shape[1]
            index = faiss.IndexFlatIP(d) if metric == 'cosine' else faiss.IndexFlatL2(d)
            if metric == 'cosine':
                faiss.normalize_L2(features_faiss)
            index.add(features_faiss)
            _, indices = index.search(features_faiss, k + 1)
            row_list, col_list = [], []
            for i_node in range(N):
                valid_neighbors = indices[i_node, 1:][indices[i_node, 1:] != -1]
                if len(valid_neighbors) > 0:
                    row_list.extend([i_node] * len(valid_neighbors))
                    col_list.extend(valid_neighbors.tolist())
            if row_list:
                knn_edge_index = torch.stack([torch.tensor(row_list), torch.tensor(col_list)], dim=0).long()

        if not use_faiss_knn_flag or (knn_edge_index.numel() == 0 and N > k):
            if N > 1 and k > 0 and N > k:
                adj_sparse = kneighbors_graph(features, n_neighbors=k, mode='connectivity', metric=metric, include_self=False, n_jobs=-1)
                knn_edge_index_sklearn, _ = from_scipy_sparse_matrix(adj_sparse)
                if knn_edge_index_sklearn.numel() > 0:
                    knn_edge_index = knn_edge_index_sklearn
            elif N <= k and N > 1:
                adj = torch.ones(N, N) - torch.eye(N)
                knn_edge_index = adj.to_sparse().indices().long()

        if knn_edge_index.numel() > 0:
            knn_edge_index = coalesce(
                torch.cat([knn_edge_index, torch.stack([knn_edge_index[1], knn_edge_index[0]], dim=0)], dim=1),
                num_nodes=N
            )
    except Exception as e:
        print(f"kNN edge generation failed: {e}.")
        return torch.empty((2, 0), dtype=torch.long)

    return knn_edge_index.to(device)

def precompute_knn_edges(data_list, source_keys, k_val, metric_val, use_faiss_val, device, force_recompute=False, cache_dir_path=None):
    """
    Precomputes and caches kNN edges for all source domain graphs.
    """
    if cache_dir_path is None:
        cache_dir_path = Path(config.CACHE_DIR_DEFAULT)
    else:
        cache_dir_path = Path(cache_dir_path)
        
    keys_str = "_".join(sorted(source_keys))
    faiss_str = "_faiss" if use_faiss_val and faiss is not None else ""
    cache_filename = f"knn_edges_k{k_val}_m{metric_val}{faiss_str}_{keys_str}.pt"
    if len(cache_filename) > 200:
        keys_hash = hashlib.md5(keys_str.encode()).hexdigest()[:16]
        cache_filename = f"knn_edges_k{k_val}_m{metric_val}{faiss_str}_hash{keys_hash}.pt"
        
    cache_path = cache_dir_path / cache_filename
    if cache_path.exists() and not force_recompute:
        try:
            knn_edges_map = torch.load(cache_path, map_location='cpu')
            for key_map in knn_edges_map:
                knn_edges_map[key_map] = knn_edges_map[key_map].to(device)
            return knn_edges_map
        except Exception as e:
            print(f"Warning: Failed to load cached kNN edges ({e}). Recomputing...")
            
    knn_edges_map = {}
    for key_iter in source_keys:
        data = data_list[key_iter]
        added_edges = knn_feature_edge_adding(data.cpu(), k_val, metric_val, use_faiss_val, device=device)
        knn_edges_map[key_iter] = added_edges
        
    try:
        cache_dir_path.mkdir(parents=True, exist_ok=True)
        knn_edges_map_cpu = {k_map: v_map.cpu() for k_map, v_map in knn_edges_map.items()}
        torch.save(knn_edges_map_cpu, cache_path)
    except Exception as e:
        print(f"Warning: Failed to save kNN edges to cache ({e}).")
        
    return knn_edges_map