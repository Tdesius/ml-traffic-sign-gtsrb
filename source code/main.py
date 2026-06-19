import logging
import os
import json
import random
import numpy as np
import torch
import platform

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

#Rich for improved console output
from rich import print
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

#Optional: basic logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

#--------------------------
#Screen clearing utility
#--------------------------
def clear_screen():
    os.system('cls' if platform.system() == 'Windows' else 'clear')

#--------------------------
#Centralized seed function
#--------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#--------------------------
#Terminal input function
#--------------------------
def get_user_input():
    clear_screen()
    print(Panel("ML TRAINING CONFIGURATION", style="bold cyan"))

    task_input = Prompt.ask("\nSelect task:\n  1. CNN (GTSRB Traffic Signs)\n  2. FNN (Concrete Regression)\nEnter choice", choices=["1","2"])
    task = 'cnn' if task_input == '1' else 'fnn'

    models = ['all']
    if task == 'cnn':
        model_input = Prompt.ask(
            "\nAvailable CNN models:\n  1. Barebone_CNN\n  2. CNN+Augmentation\n  3. Dropout_CNN\n  4. Dropout_CNN+Augmentation\n  all. Train all models\nEnter model numbers (comma-separated) or 'all'",
            default="all"
        )
        if model_input.lower() != 'all':
            models = [m.strip() for m in model_input.split(',')]

    do_train_input = Prompt.ask("\nDo you want to train the model(s)?", choices=["y","n"], default="y")
    do_train = do_train_input.lower() == 'y'

    print(Panel(f"[bold]Configuration Summary[/bold]\nTask: {'CNN (GTSRB)' if task=='cnn' else 'FNN (Concrete)'}\nModels: {', '.join(models) if task=='cnn' else 'N/A'}\nDevice: auto (cuda if available, else cpu)\nTrain: {'Yes' if do_train else 'No'}", style="yellow"))

    confirm = Prompt.ask("Proceed with this configuration?", choices=["y","n"], default="y")
    if confirm.lower() != 'y':
        print("[red]Configuration cancelled. Exiting...[/red]")
        exit(0)

    clear_screen()
    return task, models, do_train

#--------------------------
#Rich table for experiment summary
#--------------------------
def print_summary(results):
    if not results:
        print("[yellow]No results to display.[/yellow]")
        return
    
    #Check if first result has accuracy (classification) or loss (regression)
    first_result = results[0]
    
    if "Best Test Acc" in first_result or "Test Acc" in first_result:
        #Classification results
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Model", justify="left")
        table.add_column("Train Acc", justify="right")
        table.add_column("Val Acc", justify="right")
        table.add_column("Test Acc", justify="right")
        table.add_column("Time (s)", justify="right")
        
        for r in results:
            table.add_row(
                r["Model"],
                f"{r.get('Best Train Acc', 0):.4f}" if r.get('Best Train Acc') else "N/A",
                f"{r.get('Best Val Acc', 0):.4f}" if r.get('Best Val Acc') else "N/A",
                f"{r.get('Best Test Acc', 0):.4f}" if r.get('Best Test Acc') else "N/A",
                f"{r.get('Total Time', 'N/A')}"
            )
    else:
        #Regression results (MSE loss)
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Model", justify="left")
        table.add_column("Train Loss", justify="right")
        table.add_column("Val Loss", justify="right")
        table.add_column("Test Loss", justify="right")
        table.add_column("Epochs", justify="right")
        table.add_column("Early Stop", justify="center")
        for r in results:
            train_loss = r.get('Final Train Loss')
            val_loss = r.get('Final Val Loss')
            test_loss = r.get('Best Test Loss')
            
            table.add_row(
                r["Model"],
                f"{train_loss:.2f}" if isinstance(train_loss, (int, float)) else str(train_loss),
                f"{val_loss:.2f}" if isinstance(val_loss, (int, float)) else str(val_loss),
                f"{test_loss:.2f}" if isinstance(test_loss, (int, float)) else str(test_loss),
                f"{r.get('Epochs', 'N/A')}",
                f"{r.get('Early Stopping', 'N/A')}",
            )
    
    print(Panel(table, title="[bold yellow]Experiment Summary[/bold yellow]", expand=False))
    
#--------------------------
#Simple progress display function
#--------------------------
def show_progress(description, total=None):
    """Simple progress indicator without callback complexity."""
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
    )
    task = progress.add_task(f"[cyan]{description}...", total=total)
    progress.start()
    return progress, task

def update_progress(progress, task, advance=1, description=None):
    """Update progress task."""
    if description:
        progress.update(task, description=f"[cyan]{description}...")
    if advance:
        progress.update(task, advance=advance)

def finish_progress(progress, task, message=None):
    """Finish and remove progress task."""
    if message:
        progress.update(task, description=f"[green]{message}")
    progress.stop()
    progress.remove_task(task)

#--------------------------
#Main
#--------------------------
def main():
    task, models, do_train = get_user_input()
    set_seed(42)

    import pandas as pd
    from sklearn.model_selection import train_test_split
    from torchvision import transforms
    from torch.utils.data import DataLoader

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(Panel(f"TRAINING SESSION\nDevice: {device}", style="bold green"))

    results = []

    #------------------------- CNN TASK -------------------------
    if task == 'cnn':
        from models.CNN import BaselineCNN, DropoutCNN
        from models.dataset import readTrafficSigns, readTestSet, GTSRBDataset
        from models.train_utils import train_model
        from visualization.viz_utils import (
            plot_history, compute_confusion_matrix,
            plot_confusion_matrix, plot_confusion_matrix_normalized,
            save_summary_table
        )

        transform_none = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
        transform_augment = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.05,0.05)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])

        configs = {
            "1": {
                "name": "Barebone_CNN", 
                "model_class": BaselineCNN, 
                "transform": transform_none,
                "hyperparams": {
                    "epochs": 15,
                    "lr": 0.001,
                    "batch_size": 256,
                    "use_early_stopping": False,
                    "patience": 5
                }
            },
            "2": {
                "name": "CNN+Augmentation", 
                "model_class": BaselineCNN, 
                "transform": transform_augment,
                "hyperparams": {
                    "epochs": 20, 
                    "lr": 0.001,
                    "batch_size": 256,
                    "use_early_stopping": False,  
                    "patience": 7
                }
            },
            "3": {
                "name": "Dropout_CNN", 
                "model_class": DropoutCNN, 
                "transform": transform_none,
                "hyperparams": {
                    "epochs": 20,
                    "lr": 0.001,
                    "batch_size": 256,
                    "use_early_stopping": False,
                    "patience": 6,
                    "dropout_p": 0.5
                }
            },
            "4": {
                "name": "Dropout_CNN+Augmentation+HP_Tuned", 
                "model_class": DropoutCNN, 
                "transform": transform_augment,
                "hyperparams": {
                    "epochs": 25, 
                    "lr": 0.0008,  
                    "batch_size": 128,  
                    "use_early_stopping": True,
                    "patience": 8,  
                    "dropout_p": 0.4  
                }
            },
        }

        #Load dataset
        progress, task = show_progress("Loading GTSRB dataset")
        train_entries, train_labels, train_tar = readTrafficSigns("GTSRB_train.tar")
        test_entries, test_labels, test_tar = readTestSet("GTSRB_test.tar", "GT-final_test.csv")
        finish_progress(progress, task, "✓ GTSRB dataset loaded")
        
        print(f"[bold]GTSRB dataset loaded:[/bold] {len(train_entries)} train, {len(test_entries)} test images")

        model_keys = list(configs.keys()) if 'all' in models else [k for k in models if k in configs]

        for key in model_keys:
            cfg = configs[key]
            name = cfg["name"]
            hyperparams = cfg["hyperparams"]
            print(Panel(f"[bold cyan]Training {name}[/bold cyan]"))

            use_early_stopping = cfg["hyperparams"]["use_early_stopping"]

            #Split data
            train_entries_split, val_entries_split, train_labels_split, val_labels_split = train_test_split(
                train_entries, train_labels, test_size=0.2, random_state=42, stratify=train_labels
            )

            #Create datasets
            progress, ds_task = show_progress(f"Creating datasets for {name}")
            
            train_dataset = GTSRBDataset(train_entries_split, train_tar, transform=cfg["transform"])
            update_progress(progress, ds_task, description=f"Creating train dataset for {name}")
            
            val_dataset = GTSRBDataset(val_entries_split, train_tar, transform=transform_none)
            update_progress(progress, ds_task, description=f"Creating validation dataset for {name}")
            
            test_dataset = GTSRBDataset(test_entries, test_tar, transform=transform_none)
            finish_progress(progress, ds_task, f"✓ Datasets created for {name}")

            #Create dataloaders
            train_batch = hyperparams['batch_size']
            val_batch = hyperparams['batch_size'] * 2
            
            train_loader = DataLoader(train_dataset, batch_size=train_batch, shuffle=True, num_workers=6, pin_memory=True, persistent_workers=True)
            val_loader = DataLoader(val_dataset, batch_size=val_batch, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
            test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
            
            model_args = {'dropout_p': hyperparams['dropout_p']} if hyperparams.get('dropout_p', 0) > 0 else {}
            model_class = cfg["model_class"]
            model = model_class(**model_args)

            if do_train:
                #model = torch.compile(model)
                #Train model
                progress, train_task = show_progress(f"Training {name}")
                print(f"[dim]Training {name} for up to 15 epochs...[/dim]")
                
                history, test_preds, test_labels = train_model(
                    model, train_loader, val_loader, test_loader,
                    device, 
                    epochs=hyperparams['epochs'],
                    lr=hyperparams['lr'],
                    patience=hyperparams['patience'],
                    use_early_stopping=hyperparams['use_early_stopping'],
                    verbose=True
                )
                
                finish_progress(progress, train_task, f"✓ {name} training completed")

                #Create visualizations
                progress, viz_task = show_progress(f"Creating visualizations for {name}")
                
                cm = compute_confusion_matrix(test_preds, test_labels)
                update_progress(progress, viz_task, description=f"Computing confusion matrix for {name}")
                
                os.makedirs("visualization/plots", exist_ok=True)
                plot_confusion_matrix(cm, f"visualization/plots/{name}_confusion_matrix.png")
                update_progress(progress, viz_task, description=f"Saving confusion matrix for {name}")
                
                plot_confusion_matrix_normalized(cm, f"visualization/plots/{name}_confusion_matrix_normalized.png")
                update_progress(progress, viz_task, description=f"Saving normalized confusion matrix for {name}")
                
                plot_history(history, f"visualization/plots/{name}_history.png")
                finish_progress(progress, viz_task, f"✓ Visualizations created for {name}")

                #Save model
                progress, save_task = show_progress(f"Saving {name} model")
                
                model_folder = f"saved_models/{name}"
                os.makedirs(model_folder, exist_ok=True)
                torch.save(model.state_dict(), f"{model_folder}/{name}.pt")
                with open(f"{model_folder}/{name}_history.json",'w') as f:
                    json.dump(history, f, indent=4)
                
                finish_progress(progress, save_task, f"✓ {name} model saved")

                results.append({
                    "Model": name,
                    "Best Train Acc": max(history["train_acc"]) if history["train_acc"] else 0.0,
                    "Best Val Acc": max(history["val_acc"]) if history["val_acc"] else 0.0,
                    "Best Test Acc": history.get("final_test_acc", 0.0),
                    "Total Time": sum(history["epoch_time"]) if history["epoch_time"] else 0,
                    "Best Hyperparams": model_args
                })

        if results:
            df = pd.DataFrame(results)
            save_summary_table(df, "visualization/plots/summary_table.csv", truncate_len=50)

    #------------------------- FNN TASK -------------------------
    elif task == 'fnn':
        from models.FNN import BaselineFNN
        from models.dataset import ConcreteDataset
        from models.train_utils import train_fnn_regression
        from visualization.viz_utils import plot_history, save_summary_table, plot_fnn_comparison

        #Load dataset
        progress, load_task = show_progress("Loading Concrete dataset")
        dataset = ConcreteDataset("Concrete_Data.xls")
        finish_progress(progress, load_task, "✓ Concrete dataset loaded")
        
        print(f"[bold]Concrete dataset loaded:[/bold] {len(dataset.X)} rows, {dataset.X.shape[1]} features")
        
        train_dataset, val_dataset, test_dataset = dataset.train_val_test_split(val_ratio=0.2, test_ratio=0.1)
        
        fnn_configs = [
            {
                "name": "FNN_NoEarlyStop",
                "use_early_stopping": False,
                "epochs": 200,
                "lr": 0.01,
                "patience": 5
            },
            {
                "name": "FNN_WithEarlyStop",
                "use_early_stopping": True,
                "epochs": 200,
                "lr": 0.01,
                "patience": 5,
            }
        ]
            
        if do_train:
            for config in fnn_configs:
                name = config["name"]
                use_early_stopping = config["use_early_stopping"]
                
                print(Panel(f"[bold cyan]Training {name}[/bold cyan]"))
                
                set_seed(42)
                
                #Create new dataloaders for each experiment
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
                test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
                
                #Create new model instance for each experiment
                model = BaselineFNN(input_size=dataset.input_dim, output_size=1)
                
                #Train model
                progress, train_task = show_progress(f"Training {name}")
                print(f"[dim]Training {name} for up to {config['epochs']} epochs (Early stopping: {use_early_stopping})...[/dim]")
                
                history, test_preds, test_labels = train_fnn_regression(
                    model, train_loader, val_loader, test_loader,
                    device, 
                    epochs=config['epochs'], 
                    lr=config['lr'],
                    patience=config['patience'],
                    use_early_stopping=use_early_stopping,
                    verbose=True
                )
                
                finish_progress(progress, train_task, f"✓ {name} training completed")

                #Create visualization
                progress, viz_task = show_progress(f"Creating {name} visualizations")
                
                os.makedirs("visualization/plots", exist_ok=True)
                plot_history(history, f"visualization/plots/{name}_History.png")
                update_progress(progress, viz_task, description=f"Saving {name} history plot")
                
                #Save model
                model_folder = f"saved_models/{name}"
                os.makedirs(model_folder, exist_ok=True)
                torch.save(model.state_dict(), f"{model_folder}/{name}.pt")
                with open(f"{model_folder}/{name}_history.json",'w') as f:
                    json.dump(history, f, indent=4)
                
                finish_progress(progress, viz_task, f"✓ {name} visualizations created")
                
                #Store results
                results.append({
                    "Model": name,
                    "Final Train Loss": history["train_loss"][-1] if history.get("train_loss") else None,
                    "Final Val Loss": history["val_loss"][-1] if history.get("val_loss") else None,
                    "Best Test Loss": history.get("final_test_loss", None),
                    "Epochs": len(history.get("train_loss", [])),
                    "Total Time": sum(history.get("epoch_time", [])),
                    "Early Stopping": "Yes" if use_early_stopping else "No",
                    "Stopped Early": history.get("stopped_early", False) if do_train else False
                })
                
                #Clear memory
                del model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
        else:
            #If not training, just show placeholder results
            results.append({
                "Model": "FNN_NoEarlyStop",
                "Final Train Loss": "N/A (not trained)",
                "Final Val Loss": "N/A (not trained)",
                "Best Test Loss": "N/A (not trained)",
                "Epochs": "N/A",
                "Total Time": "N/A",
                "Early Stopping": "No"
            })
            results.append({
                "Model": "FNN_WithEarlyStop",
                "Final Train Loss": "N/A (not trained)",
                "Final Val Loss": "N/A (not trained)",
                "Best Test Loss": "N/A (not trained)",
                "Epochs": "N/A",
                "Total Time": "N/A",
                "Early Stopping": "Yes"
            })
        if do_train and len(results) > 1:
            progress, comp_task = show_progress("Creating FNN comparison plot")
            from visualization.viz_utils import plot_fnn_comparison
            plot_fnn_comparison(results, "visualization/plots/FNN_Comparison.png")
            finish_progress(progress, comp_task, "✓ FNN comparison plot created")
            
        df = pd.DataFrame(results)
        save_summary_table(df, "visualization/plots/summary_table.csv", truncate_len=50)

    #------------------------- SUMMARY -------------------------
    print_summary(results)
    print("\n[bold green]Training completed ✔[/bold green]")

if __name__ == "__main__":  
    from multiprocessing import freeze_support
    freeze_support() 
    main()