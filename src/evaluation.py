import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from matplotlib.patches import Rectangle
import torch
import os

def get_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data, targets in loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
    return np.array(all_preds), np.array(all_labels)

def evaluate_model(preds, labels):
    return 100 * (preds == labels).sum() / len(labels)

def save_results(df_all_runs, df_stats, save_path):
    """Saves the numerical results to CSV files."""
    df_all_runs.to_csv(os.path.join(save_path, "all_runs_data.csv"), index=False)
    df_stats.to_csv(os.path.join(save_path, "summary_stats.csv"))
    print("\n--- Summary Statistics (Mean ± Std over 5 Runs) ---")
    print(df_stats.to_string())

def plot_main_results(df_all_runs, save_path):
    # ... [function body from previous script] ...
    df_melted = df_all_runs.melt(id_vars=["Method", "Run"], value_vars=["Retain Set Accuracy (%)", "Forget Set Accuracy (%)", "Test Set Accuracy (%)"], var_name="Metric", value_name="Accuracy")
    plt.figure(figsize=(12, 7)); sns.barplot(data=df_melted, x="Metric", y="Accuracy", hue="Method", palette="viridis", errorbar="sd")
    plt.title("Comparison of Unlearning Methods (5 Runs with Std. Dev.)", fontsize=16); plt.ylabel("Accuracy (%)", fontsize=12); plt.xlabel(""); plt.ylim(0, 100); plt.legend(title="Method"); plt.tight_layout()
    plt.savefig(os.path.join(save_path, "main_results_comparison.pdf"), format='pdf', bbox_inches='tight'); plt.close()

def plot_tradeoff(df_stats, save_path):
    # ... [function body from previous script] ...
    data_mean = df_stats[[('Retain Set Accuracy (%)', 'mean'), ('Forget Set Accuracy (%)', 'mean')]].copy(); data_mean.columns = ['Retain Accuracy', 'Forget Accuracy']; data_mean['Method'] = data_mean.index
    plt.figure(figsize=(8, 8)); sns.scatterplot(data=data_mean, x="Forget Accuracy", y="Retain Accuracy", hue="Method", style="Method", s=250, palette="deep")
    plt.title("Unlearning Trade-off: Fidelity vs. Efficacy (5-Run Avg)", fontsize=16); plt.xlim(0, 100); plt.ylim(0, 100); plt.grid(True); plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.); plt.tight_layout()
    plt.savefig(os.path.join(save_path, "tradeoff_scatter_plot.pdf"), format='pdf', bbox_inches='tight'); plt.close()

def plot_confusion_matrix(model, loader, class_names, title, save_path, device, highlight_class=None):
    # ... [function body from previous script] ...
    preds, labels = get_predictions(model, loader, device); cm = confusion_matrix(labels, preds); plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix: {title}', fontsize=16); plt.xlabel('Predicted Label', fontsize=12); plt.ylabel('True Label', fontsize=12)
    if highlight_class is not None: plt.gca().add_patch(Rectangle((0, highlight_class), 10, 1, edgecolor='red', facecolor='none', lw=3))
    plt.tight_layout(); plt.savefig(os.path.join(save_path, f"confusion_matrix_{title.replace(' ', '_')}.pdf"), format='pdf', bbox_inches='tight'); plt.close()

def plot_ablation_results(df_all_runs, save_path):
    # ... [function body from previous script] ...
    ablation_methods = ["Fine-tune (Grad-Ascent)", "CGU (No Trace)", "CGU (No Repair)", "CGU (Ours)"]
    df_ablation = df_all_runs[df_all_runs['Method'].isin(ablation_methods)].copy()
    df_melted = df_ablation.melt(id_vars=["Method", "Run"], value_vars=["Retain Set Accuracy (%)", "Forget Set Accuracy (%)"], var_name="Metric", value_name="Accuracy")
    plt.figure(figsize=(10, 6)); sns.barplot(data=df_melted, x="Metric", y="Accuracy", hue="Method", palette="magma", errorbar="sd")
    plt.title("Ablation Study of CGU Components (5 Runs)", fontsize=16); plt.ylabel("Accuracy (%)", fontsize=12); plt.xlabel(""); plt.ylim(0, 100); plt.legend(title="Method", loc='upper left', bbox_to_anchor=(1, 1)); plt.tight_layout()
    plt.savefig(os.path.join(save_path, "ablation_comparison.pdf"), format='pdf', bbox_inches='tight'); plt.close()