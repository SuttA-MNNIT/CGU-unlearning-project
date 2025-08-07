import torch
import numpy as np
import pandas as pd
import os
from datetime import datetime
import time
import copy

# Import from our source directory
from src.model import SimpleCNN
from src.data_handler import prepare_data
from src.unlearning_methods import (
    train_model,
    unlearn_retrain,
    unlearn_finetune,
    CausalGenerativeUnlearning,
    unlearn_cgu_no_trace,
    unlearn_cgu_no_repair
)
from src.evaluation import (
    get_predictions,
    evaluate_model,
    save_results,
    plot_main_results,
    plot_tradeoff,
    plot_confusion_matrix,
    plot_ablation_results
)
import config  # Import settings from config.py

def main():
    """Main function to orchestrate the entire experiment."""
    print(f"Using device: {config.DEVICE}")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    SAVE_PATH = os.path.join(config.BASE_SAVE_PATH, f"final_run_{timestamp}")
    os.makedirs(SAVE_PATH, exist_ok=True)
    print(f"Results will be saved to: {SAVE_PATH}")
    
    all_results = []
    
    print("\n--- Training Original Full Model (once) ---")
    torch.manual_seed(42); np.random.seed(42)
    train_loader_full, test_loader, _, _, _, class_names = prepare_data(
        config.FORGET_CLASS, config.NUM_TO_FORGET, config.SAMPLE_SIZE, config.BATCH_SIZE
    )
    original_model = SimpleCNN()
    original_model = train_model(original_model, train_loader_full, config.PRETRAIN_EPOCHS, config.LEARNING_RATE, config.DEVICE)
    
    # --- Run experiments over multiple seeds ---
    for seed in range(config.NUM_RUNS):
        print(f"\n{'='*20} STARTING RUN {seed+1}/{config.NUM_RUNS} {'='*20}")
        torch.manual_seed(seed); np.random.seed(seed)
        _, _, retain_loader, forget_loader, sample_loader, _ = prepare_data(
            config.FORGET_CLASS, config.NUM_TO_FORGET, config.SAMPLE_SIZE, config.BATCH_SIZE
        )
        
        # Evaluate Original model on this specific run's data split
        # ... [Evaluation logic as before] ...
        retain_preds, retain_labels = get_predictions(original_model, retain_loader, config.DEVICE); forget_preds, forget_labels = get_predictions(original_model, forget_loader, config.DEVICE); test_preds, test_labels = get_predictions(original_model, test_loader, config.DEVICE)
        all_results.append({"Method": "Original", "Run": seed, "Execution Time (s)": 0, "Retain Set Accuracy (%)": evaluate_model(retain_preds, retain_labels), "Forget Set Accuracy (%)": evaluate_model(forget_preds, forget_labels), "Test Set Accuracy (%)": evaluate_model(test_preds, test_labels)})

        methods_to_run = {
            "Retrained (Benchmark)": lambda: unlearn_retrain(SimpleCNN, retain_loader, config.PRETRAIN_EPOCHS, config.LEARNING_RATE, config.DEVICE),
            "Fine-tune (Grad-Ascent)": lambda: unlearn_finetune(copy.deepcopy(original_model), forget_loader, config.UNLEARN_EPOCHS, lr=config.ABLATION_FINETUNE_LR, device=config.DEVICE),
            "CGU (No Trace)": lambda: unlearn_cgu_no_trace(copy.deepcopy(original_model), retain_loader, forget_loader, config.BETA, config.LEARNING_RATE, config.CGU_REPAIR_STEPS, config.DEVICE),
            "CGU (No Repair)": lambda: unlearn_cgu_no_repair(copy.deepcopy(original_model), forget_loader, sample_loader, config.K_PERCENTILE, config.GAMMA, config.UNLEARN_EPOCHS, config.ABLATION_FINETUNE_LR, config.DEVICE),
            "CGU (Ours)": lambda: CausalGenerativeUnlearning(copy.deepcopy(original_model), k=config.K_PERCENTILE, gamma=config.GAMMA, beta=config.BETA, device=config.DEVICE).unlearn(forget_loader, sample_loader, retain_loader, learning_rate=config.LEARNING_RATE, repair_steps=config.CGU_REPAIR_STEPS)
        }
        for name, method_func in methods_to_run.items():
            start_time = time.time(); unlearned_model = method_func(); end_time = time.time(); duration = end_time - start_time
            retain_preds, retain_labels = get_predictions(unlearned_model, retain_loader, config.DEVICE); forget_preds, forget_labels = get_predictions(unlearned_model, forget_loader, config.DEVICE); test_preds, test_labels = get_predictions(unlearned_model, test_loader, config.DEVICE)
            all_results.append({"Method": name, "Run": seed, "Execution Time (s)": duration, "Retain Set Accuracy (%)": evaluate_model(retain_preds, retain_labels), "Forget Set Accuracy (%)": evaluate_model(forget_preds, forget_labels), "Test Set Accuracy (%)": evaluate_model(test_preds, test_labels)})
            # Save representative confusion matrix for the final run's CGU model
            if name == "CGU (Ours)" and seed == config.NUM_RUNS - 1:
                 plot_confusion_matrix(unlearned_model, test_loader, class_names, "CGU_Model_Representative", SAVE_PATH, config.DEVICE, highlight_class=config.FORGET_CLASS)
    
    # Process and Save Final Results
    df_all_runs = pd.DataFrame(all_results)
    df_stats = df_all_runs.groupby('Method').agg(['mean', 'std'])
    method_order = ["Original", "Fine-tune (Grad-Ascent)", "CGU (No Repair)", "CGU (No Trace)", "CGU (Ours)", "Retrained (Benchmark)"]
    df_stats = df_stats.reindex(method_order)
    
    save_results(df_all_runs, df_stats, SAVE_PATH)
    
    # Generate and Save Visualizations
    print("\n--- Generating and Saving Visualizations ---")
    plot_main_results(df_all_runs, SAVE_PATH)
    plot_tradeoff(df_stats, SAVE_PATH)
    plot_ablation_results(df_all_runs, SAVE_PATH)
    plot_confusion_matrix(original_model, test_loader, class_names, "Original Model", SAVE_PATH, config.DEVICE, highlight_class=config.FORGET_CLASS)
    
    print(f"\nAll results and charts have been successfully saved to {SAVE_PATH}")

if __name__ == '__main__':
    main()