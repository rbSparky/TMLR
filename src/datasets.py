"""
Functions for dataset loading and generating artificial spurious features
for Out-of-Distribution (OOD) benchmark experiments.
"""

import torch
from torch_geometric.datasets import Planetoid, Amazon

def _add_spurious_feature_to_graph(base_graph_data_cpu, majority_class_idx: int, spur_ratio: float):
    """
    Adds a single binary spurious feature correlated with a majority class.
    
    P(spur=1 | Y=maj_class) = 1 - spur_ratio
    P(spur=1 | Y!=maj_class) = spur_ratio

    Args:
        base_graph_data_cpu: The base PyG Data object on CPU.
        majority_class_idx (int): The index of the class to correlate with.
        spur_ratio (float): The correlation strength.

    Returns:
        A new PyG Data object with the added spurious feature.
    """
    data = base_graph_data_cpu.clone()
    num_nodes = data.num_nodes
    spurious_feature_values = torch.zeros(num_nodes, 1, dtype=torch.float)

    for i in range(num_nodes):
        is_majority_class = (data.y[i].item() == majority_class_idx)
        prob_spur_is_one = (1.0 - spur_ratio) if is_majority_class else spur_ratio
        if torch.rand(1).item() < prob_spur_is_one:
            spurious_feature_values[i, 0] = 1.0

    data.x = torch.cat([data.x, spurious_feature_values], dim=-1)
    return data

def _generate_artificial_domains_for_scenario(base_data_cpu, dataset_name: str, scenario_idx: int, num_total_scenarios: int):
    """
    Generates train, validation, and test domains for one scenario.

    This follows the EERM paper's setup with 2 train, 1 val, and 1 test graph.
    The spurious correlation shifts between train/val and test.

    Args:
        base_data_cpu: The base PyG Data object on CPU.
        dataset_name (str): Name of the dataset for key naming.
        scenario_idx (int): The index of the current scenario.
        num_total_scenarios (int): Total number of scenarios.

    Returns:
        A tuple containing (domain_graphs, source_keys, val_key, test_key).
    """
    num_classes = base_data_cpu.y.max().item() + 1
    majority_class_for_scenario = scenario_idx % num_classes

    train_spur_ratio = 0.1
    val_spur_ratio = 0.3
    test_spur_ratio = 0.9

    domain_graphs = {}

    for i in range(2):
        key = f"{dataset_name}_scen{scenario_idx}_train{i}"
        domain_graphs[key] = _add_spurious_feature_to_graph(
            base_data_cpu, majority_class_for_scenario, train_spur_ratio
        )
    key_val = f"{dataset_name}_scen{scenario_idx}_val0"
    domain_graphs[key_val] = _add_spurious_feature_to_graph(
        base_data_cpu, majority_class_for_scenario, val_spur_ratio
    )
    key_test = f"{dataset_name}_scen{scenario_idx}_test0"
    domain_graphs[key_test] = _add_spurious_feature_to_graph(
        base_data_cpu, majority_class_for_scenario, test_spur_ratio
    )

    source_keys = [f"{dataset_name}_scen{scenario_idx}_train{i}" for i in range(2)]
    return domain_graphs, source_keys, key_val, key_test

def get_artificial_spurious_data_for_ood(dataset_name: str, root_dir: str = '.', num_test_scenarios: int = 8):
    """
    Prepares a dataset (Cora or Photo) with artificial spurious features for OOD experiments.

    Args:
        dataset_name (str): The name of the dataset ('cora' or 'photo').
        root_dir (str): The directory to download and store the base dataset.
        num_test_scenarios (int): The number of OOD scenarios to generate.

    Returns:
        A list of tuples, one for each scenario. Each tuple contains:
        (data_list_for_scenario, source_keys, val_key, target_key, num_features, num_classes).
        All Data objects are on CPU.
    """
    print(f"Loading base {dataset_name} dataset from: {root_dir}")
    if dataset_name.lower() == 'cora':
        base_pyg_dataset = Planetoid(root=root_dir, name='Cora')
    elif dataset_name.lower() == 'photo':
        base_pyg_dataset = Amazon(root=root_dir, name='Photo')
    else:
        raise ValueError(f"Unsupported dataset for artificial spurious features: {dataset_name}")

    base_data_cpu = base_pyg_dataset[0].cpu()
    original_num_features = base_data_cpu.num_features
    num_classes = base_pyg_dataset.num_classes
    scenarios_data_list = []

    for scenario_idx in range(num_test_scenarios):
        print(f"  Generating data for {dataset_name} OOD Scenario {scenario_idx + 1}/{num_test_scenarios}...")
        current_base_data_cpu = base_data_cpu.clone()

        domain_graphs, source_keys, val_key, target_key = \
            _generate_artificial_domains_for_scenario(
                current_base_data_cpu,
                dataset_name.lower(),
                scenario_idx,
                num_total_scenarios=num_test_scenarios
            )
        
        updated_num_features = original_num_features + 1
        
        scenarios_data_list.append(
            (domain_graphs, source_keys, val_key, target_key, updated_num_features, num_classes)
        )
    
    print(f"Finished generating {dataset_name} OOD data for {num_test_scenarios} scenarios.")
    return scenarios_data_list