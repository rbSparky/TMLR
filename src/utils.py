"""General utility functions for the project."""

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.data import Data
from torch.cuda.amp import autocast

def set_seed(seed: int):
    """
    Sets the random seed for reproducibility across NumPy, PyTorch, and CUDA.

    Args:
        seed (int): The seed to use.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@torch.no_grad()
def evaluate(model: torch.nn.Module, data_loader_or_graph, device, use_amp_eval: bool):
    """
    Evaluates the model on a given dataset.

    Args:
        model (torch.nn.Module): The model to evaluate.
        data_loader_or_graph: A single PyG Data object or a DataLoader.
        device: The device to run evaluation on.
        use_amp_eval (bool): Whether to use Automatic Mixed Precision for evaluation.

    Returns:
        A tuple containing (accuracy, micro_f1, macro_f1).
    """
    model.eval()
    all_preds, all_labels = [], []

    is_single_graph = isinstance(data_loader_or_graph, Data)
    if is_single_graph:
        data = data_loader_or_graph.to(device)
        with autocast(enabled=(use_amp_eval and device.type == 'cuda')):
            out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1).cpu()
        labels = data.y.cpu()
        all_preds.append(pred)
        all_labels.append(labels)
    else:
        for data_batch in data_loader_or_graph:
            data_batch = data_batch.to(device)
            with autocast(enabled=(use_amp_eval and device.type == 'cuda')):
                out = model(data_batch.x, data_batch.edge_index)
            pred = out.argmax(dim=1).cpu()
            labels = data_batch.y.cpu()
            all_preds.append(pred)
            all_labels.append(labels)

    if not all_preds:
        return 0.0, 0.0, 0.0

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    if len(all_labels) == 0:
        return 0.0, 0.0, 0.0

    acc = accuracy_score(all_labels, all_preds)
    micro_f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return acc, micro_f1, macro_f1