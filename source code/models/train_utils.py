import torch
import torch.nn as nn
import torch.optim as optim
import time

def train_model(
    model, train_loader, val_loader, test_loader, device,
    epochs=30, lr=0.001, patience=5, use_early_stopping=True,
    verbose=True
):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    
    best_val_loss = float("inf")
    early_stop_counter = 0
    best_weights = None
    
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "train_time": [], "val_time": [],
        "epoch_time": []
    }
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        #========== TRAIN ==========
        train_start = time.time()
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_time = time.time() - train_start
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        #========== VALIDATION ==========
        val_start = time.time()
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_time = time.time() - val_start
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        #========== LOGGING ==========
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["train_time"].append(train_time)
        history["val_time"].append(val_time)
        history["epoch_time"].append(time.time() - epoch_start)
        
        if verbose:
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        #========== EARLY STOPPING ==========
        if use_early_stopping:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
                best_weights = model.state_dict().copy()
            else:
                early_stop_counter += 1
                
            if early_stop_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
    
    if best_weights is not None:
        model.load_state_dict(best_weights)
    
    #========== FINAL TEST ==========
    model.eval()
    test_start = time.time()
    test_preds, test_labels = [], []
    test_correct, test_total = 0, 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
            test_preds.append(predicted.cpu())
            test_labels.append(labels.cpu())
    
    test_time = time.time() - test_start
    final_test_acc = test_correct / test_total
    test_preds = torch.cat(test_preds) if test_preds else None
    test_labels = torch.cat(test_labels) if test_labels else None
    
    history["final_test_acc"] = final_test_acc
    history["test_time"] = test_time
    
    if verbose:
        print(f"\nFinal Test Accuracy: {final_test_acc:.4f}")
        print(f"Test Time: {test_time:.2f}s")
    
    return history, test_preds, test_labels

#===========================
#Regression Training (FNN)
#===========================
def train_fnn_regression(
    model, train_loader, val_loader, test_loader, device,
    epochs=200, lr=0.001, patience=20, use_early_stopping=True,
    verbose=True
):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=25,
        min_lr=1e-5,
        threshold=0.01
    )
    
    best_val_loss = float("inf")
    early_stop_counter = 0
    best_weights = None
    
    #Track best epoch and improvement threshold
    best_epoch = 0
    min_delta = 0.01  #1% relative improvement threshold
    min_epochs = 50   #Minimum epochs before early stopping can trigger

    history = {
        "train_loss": [], 
        "val_loss": [],
        "train_time": [], 
        "val_time": [],
        "epoch_time": []
    }

    for epoch in range(epochs):
        epoch_start = time.time()
        
        if verbose and epoch % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}...", end=" ", flush=True)
        
        #---------- TRAIN ----------
        model.train()
        train_start = time.time()
        train_loss = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_time = time.time() - train_start
        train_loss /= len(train_loader)

        #---------- VALIDATION ----------
        model.eval()
        val_start = time.time()
        val_loss = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x)
                val_loss += criterion(preds, y).item()
        
        val_time = time.time() - val_start
        val_loss /= len(val_loader)
        epoch_time = time.time() - epoch_start

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_time"].append(train_time)
        history["val_time"].append(val_time)
        history["epoch_time"].append(epoch_time)

        if verbose and epoch % 10 == 0:
            print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        scheduler.step(val_loss)
        
        #---------- EARLY STOPPING ----------
        if use_early_stopping and epoch >= min_epochs:
            #Check if validation loss improved by at least min_delta percentage
            improvement = (best_val_loss - val_loss) / best_val_loss
            
            if improvement > min_delta:
                best_val_loss = val_loss
                early_stop_counter = 0
                best_weights = model.state_dict().copy()
                best_epoch = epoch
                if verbose and epoch % 10 == 0:
                    print(f"  ✓ Improved by {improvement*100:.2f}%")
            else:
                early_stop_counter += 1
                
            if early_stop_counter >= patience:
                if verbose:
                    print(f"  Early stopping triggered at epoch {epoch+1}")
                    print(f"  Best validation loss: {best_val_loss:.6f} at epoch {best_epoch+1}")
                    print(f"  No improvement for {patience} consecutive epochs")
                break
        elif use_early_stopping and epoch < min_epochs:
            #Still track best loss during warm-up period
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = model.state_dict().copy()
                best_epoch = epoch

    #Load best weights
    if best_weights:
        model.load_state_dict(best_weights)

    #Add early stopping info to history
    history["stopped_early"] = use_early_stopping and (epoch + 1 < epochs)
    history["final_epoch"] = epoch + 1
    history["best_epoch"] = best_epoch + 1
    history["best_val_loss"] = best_val_loss

    #---------- FINAL TEST ----------
    model.eval()
    test_start = time.time()
    test_loss = 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            test_loss += criterion(preds, y).item()
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())
    
    test_time = time.time() - test_start
    test_loss /= len(test_loader)
    
    #Add final test metrics
    history["final_test_loss"] = test_loss
    history["test_time"] = test_time
    
    #Optional: Return predictions for analysis
    final_preds = torch.cat(all_preds) if all_preds else None
    final_labels = torch.cat(all_labels) if all_labels else None
    
    if verbose:
        print(f"\n  Final Test Loss (MSE): {test_loss:.6f}")
        if use_early_stopping and history["stopped_early"]:
            print(f"  Training stopped early at epoch {epoch+1}/{epochs}")
            print(f"  Best model from epoch {best_epoch+1}")
        elif use_early_stopping:
            print(f"  Completed all {epochs} epochs (early stopping not triggered)")
        print(f"  Test Time: {test_time:.2f}s")

    return history, final_preds, final_labels


#===========================
#Utility function for reproducibility
#===========================
def set_seed(seed=42):
    """Set seed for reproducibility"""
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
