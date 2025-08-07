# config.py
"""
Central configuration file for the CGU unlearning experiments.
Edit these parameters to change experimental settings.
"""

# --- Path and Device Settings ---
BASE_SAVE_PATH = "./results"  # Base directory to save all output
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Experiment Settings ---
NUM_RUNS = 5              # Number of independent runs for statistical robustness
FORGET_CLASS = 3          # CIFAR-10 'cat' class
NUM_TO_FORGET = 500       # Number of samples to unlearn
SAMPLE_SIZE = 1000        # Size of the retain sample set for Causal Tracing
BATCH_SIZE = 128          # Batch size for DataLoaders
PRETRAIN_EPOCHS = 10      # Epochs for training the original and retrained models
UNLEARN_EPOCHS = 3        # Epochs for fine-tuning based unlearning methods
CGU_REPAIR_STEPS = 200    # Optimization steps for the Generative Repair stage

# --- Tuned Hyperparameters ---
LEARNING_RATE = 5e-5      # Main learning rate for training and CGU repair
K_PERCENTILE = 0.10       # Top % of parameters to identify as critical (10%)
BETA = 0.05               # Unlearning factor in the Generative Repair loss
GAMMA = 1.0               # Balance factor for the Causal Tracing saliency score
ABLATION_FINETUNE_LR = 1e-3 # A specific, higher LR for gradient ascent baselines