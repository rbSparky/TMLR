"""Stores default hyperparameters and configuration constants."""

# --- Directory Settings ---
RESULTS_DIR_DEFAULT = 'results_ood_benchmarks'
CACHE_DIR_DEFAULT = 'cache_ood_benchmarks'

# --- Model Hyperparameters ---
MASK_PROJ_DIM = 128
HIDDEN_DIM = 64
NUM_LAYERS = 2
GAT_HEADS = 8
GAT_DROPOUT = 0.6

# --- Training Hyperparameters ---
LR = 1e-3
WEIGHT_DECAY = 5e-4
EPOCHS = 200
LAMBDA_SPARSITY = 1e-3
DESCENT_STEPS = 5
ASCENT_STEPS = 1
EARLY_STOPPING_PATIENCE = 20
USE_AMP = True
MASK_CHUNK_SIZE = 500000

# --- Feature Edge Engineering Hyperparameters ---
KNN_K = 10
KNN_METRIC = 'cosine'
USE_FAISS_KNN = False
SPECTRAL_K_DEFAULT = 100
SPECTRAL_ADD_RATIO_DEFAULT = 0.1
EMDG_STAR_SPECTRAL_SAMPLE_RATIO = 0.1
EMDG_STAR_KNN_SAMPLE_RATIO = 0.1