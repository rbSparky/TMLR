# EdgeMask-DG*: Learning Domain-Invariant Graph Structures via Adversarial Edge Masking

This repository contains the official source code for the TMLR submission: "EdgeMask-DG*: Learning Domain-Invariant Graph Structures via Adversarial Edge Masking".

The code implements the EdgeMask-DG* method with a GAT backbone and provides the experimental setup for the artificial Out-of-Distribution (OOD) benchmarks on the Cora and Photo datasets.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    This project requires PyTorch and PyG. Please follow the official installation instructions for your specific CUDA version first.

    *   **PyTorch:** [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
    *   **PyG (PyTorch Geometric):** [https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)

    After installing PyTorch and PyG, install the remaining packages from `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Data:**
    The `Planetoid` (Cora) and `Amazon` (Photo) datasets will be automatically downloaded by PyG into the directory specified by the `--data_dir` argument (default: `./data_photo`).

## Running the Code

You can run the experiments using the `main.py` script. All hyperparameters and settings are configurable via command-line arguments.

### Example Command

Here is an example command to run the OOD experiment on the **Photo** dataset:

```bash
python main.py --lr 5e-4 --weight_decay 1e-3 --hidden_dim 128 --num_layers 3 --spectral_k 200 \
    --experiment_target photo \
    --data_dir ./data_photo \
    --results_dir ./results_photo_ood \
    --cache_dir ./cache_photo_ood \
    --seed 42 \
    --epochs 200 \
    --early_stopping_patience 20
```

### Key Arguments

-   `--experiment_target`: The OOD benchmark to run (`cora` or `photo`).
-   `--data_dir`: Directory to store the downloaded datasets.
-   `--results_dir`: Directory to save the final CSV results.
-   `--cache_dir`: Directory to cache precomputed edges (kNN, Spectral).
-   `--seed`: Random seed for reproducibility.
-   `--lr`, `--weight_decay`, `--hidden_dim`, etc.: Hyperparameters for the model and training process.

For a full list of available arguments, run:
```bash
python main.py --help
```
