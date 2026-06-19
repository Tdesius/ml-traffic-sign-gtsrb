import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
import torch

#============================================================
#HISTORY PLOTS (Regression + Classification + Early Stopping)
#============================================================
def plot_history(history, save_path):
    """Plot training history (train/val metrics) for both classification and regression"""
    
    #Determine if it's classification or regression based on available metrics
    is_classification = "train_acc" in history
    
    if is_classification:
        #Classification plot
        plt.figure(figsize=(12, 4))
        
        #1. Plot loss
        plt.subplot(1, 2, 1)
        if "train_loss" in history:
            plt.plot(history["train_loss"], label='Train Loss')
        if "val_loss" in history:
            plt.plot(history["val_loss"], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss over epochs')
        plt.grid(True, alpha=0.3)
        
        #2. Plot accuracy
        plt.subplot(1, 2, 2)
        if "train_acc" in history:
            plt.plot(history["train_acc"], label='Train Acc')
        if "val_acc" in history:
            plt.plot(history["val_acc"], label='Val Acc')
        
        #Optional: Add final test accuracy as a horizontal line
        if "final_test_acc" in history:
            plt.axhline(y=history["final_test_acc"], color='r', linestyle='--', 
                       label=f'Final Test Acc: {history["final_test_acc"]:.4f}')
        
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Accuracy over epochs')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()
    
    else:
        #Regression plot - only loss
        plt.figure(figsize=(10, 6))
        
        if "train_loss" in history:
            plt.plot(history["train_loss"], label='Train Loss')
        if "val_loss" in history:
            plt.plot(history["val_loss"], label='Val Loss')
        
        #Optional: Add final test loss as a horizontal line
        if "final_test_loss" in history:
            plt.axhline(y=history["final_test_loss"], color='r', linestyle='--', 
                       label=f'Final Test Loss: {history["final_test_loss"]:.4f}')
        
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.title('Loss over epochs (Regression)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()


#============================================================
#CONFUSION MATRIX (classification)
#============================================================
def compute_confusion_matrix(y_pred, y_true, num_classes=43):
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()

    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def plot_confusion_matrix(cm, save_path, class_names=None, title="Confusion Matrix"):
    n = cm.shape[0]

    plt.figure(figsize=(12, 10))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.colorbar()

    if class_names is None:
        class_names = [str(i) for i in range(n)]

    plt.xticks(range(n), class_names, rotation=90, fontsize=6)
    plt.yticks(range(n), class_names, fontsize=6)

    #text annotations
    thresh = cm.max() / 2
    font_size = max(4, int(250 / n))

    for i in range(n):
        for j in range(n):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                     fontsize=font_size,
                     color="white" if cm[i, j] > thresh else "black")

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_confusion_matrix_normalized(cm, save_path, class_names=None, title="Normalized Confusion Matrix"):
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    plt.figure(figsize=(12, 10))
    plt.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.title(title)
    plt.colorbar(label="Proportion")

    n = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n)]

    plt.xticks(range(n), class_names, rotation=90, fontsize=6)
    plt.yticks(range(n), class_names, fontsize=6)

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


#============================================================
#MODEL COMPARISON 
#============================================================
def plot_model_comparison(results, save_path, metric="Best Test Acc"):
    """
    results = list of dicts like:
      { "Model": "CNN", "Best Test Acc": 0.92 }
    """

    names = [r["Model"] for r in results]
    vals = [r.get(metric, 0) for r in results]

    plt.figure(figsize=(10, 6))
    plt.bar(names, vals, color="skyblue")
    plt.title(f"Model Comparison ({metric})")
    plt.ylabel(metric)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def plot_fnn_comparison(results, save_path):
    """Plot comparison of FNN models (train/val/test losses)"""
    models = [r["Model"] for r in results]
    
    #Extract losses
    train_losses = [r.get("Final Train Loss", 0) for r in results]
    val_losses = [r.get("Final Val Loss", 0) for r in results]
    test_losses = [r.get("Best Test Loss", 0) for r in results]
    
    x = np.arange(len(models))
    width = 0.25
    
    plt.figure(figsize=(12, 6))
    
    plt.bar(x - width, train_losses, width, label='Train Loss', color='skyblue', alpha=0.8)
    plt.bar(x, val_losses, width, label='Val Loss', color='lightcoral', alpha=0.8)
    plt.bar(x + width, test_losses, width, label='Test Loss', color='lightgreen', alpha=0.8)
    
    plt.xlabel('Model')
    plt.ylabel('MSE Loss')
    plt.title('FNN Model Comparison')
    plt.xticks(x, models)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    #Add value labels
    for i, (train, val, test) in enumerate(zip(train_losses, val_losses, test_losses)):
        plt.text(i - width, train + max(train_losses)*0.01, f'{train:.1f}', ha='center', fontsize=8)
        plt.text(i, val + max(val_losses)*0.01, f'{val:.1f}', ha='center', fontsize=8)
        plt.text(i + width, test + max(test_losses)*0.01, f'{test:.1f}', ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
#============================================================
#SUMMARY TABLE
#============================================================
def save_summary_table(df, save_path, truncate_len=40):
    df.to_csv(save_path, index=False)
