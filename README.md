# Causal-Generative Unlearning (CGU)

This repository contains the official implementation for the paper: "Causal-Generative Unlearning (CGU) as a High-Fidelity Approach to Selective Data Removal in Deep Neural Networks".

The code provides a framework to run machine unlearning experiments, comparing our proposed CGU method against standard baselines and ablation variants.

## Project Structure

- `run_experiments.py`: The main script to launch the entire experimental pipeline.
- `config.py`: A centralized file for all hyperparameters and experimental settings. Edit this file to change parameters.
- `requirements.txt`: A list of required Python packages.
- `src/`: A directory containing the core, modularized source code:
  - `model.py`: The `SimpleCNN` model definition.
  - `data_handler.py`: Handles downloading and preparing the CIFAR-10 dataset.
  - `unlearning_methods.py`: Contains the implementations for CGU, baselines, and ablations.
  - `evaluation.py`: Includes all functions for model evaluation, plotting, and saving results.

## How to Run

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Experiment:**
    Execute the main script from the root of the project directory. All results, including CSV files with numerical data and PDF figures, will be saved to a timestamped folder inside the `./results` directory.
    ```bash
    python run_experiments.py
    ```

3.  **Review the Output:**
    Check the newly created `./results/final_run_[timestamp]/` directory for the following outputs:
    - `all_runs_data.csv`: Raw data from every run and every method.
    - `summary_stats.csv`: The aggregated mean and standard deviation for all metrics.
    - `*.pdf`: All generated figures in high-quality, vectorized format.