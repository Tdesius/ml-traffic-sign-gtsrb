import tarfile
import io
import csv
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import torch
import threading


#-----------------------------
#Dataset Reading Utilities
#-----------------------------

def readTrafficSigns(tar_path):
    """
    Reads training image paths and labels from a GTSRB .tar file.

    Returns:
        entries: list of (image_path_in_tar, label)
        labels: list of integer labels (for stratified splitting)
        tar_path: the path to the tar file
    """
    entries = []
    labels = []
    with tarfile.open(tar_path, 'r') as tar:
        #Look for GT-*.csv files in tar
        members = [m for m in tar.getmembers() if 'GT-' in m.name and m.name.endswith('.csv')]
        for gt_member in members:
            f = tar.extractfile(gt_member)
            gt_csv = csv.reader(io.TextIOWrapper(f, encoding='utf-8'), delimiter=';')
            next(gt_csv)  #skip header
            for row in gt_csv:
                img_path_in_tar = '/'.join(gt_member.name.split('/')[:-1]) + '/' + row[0]
                label = int(row[7])
                entries.append((img_path_in_tar, label))
                labels.append(label)
    return entries, labels, tar_path


def readTestSet(tar_path, gt_csv_member):
    """
    Reads test image paths and labels from GTSRB test .tar file.

    Args:
        tar_path: path to GTSRB test tar
        gt_csv_member: path to GT-final_test.csv inside tar

    Returns:
        entries: list of (image_path_in_tar, label)
        labels: list of integer labels
        tar_path: path to tar
    """
    entries = []
    labels = []
    with tarfile.open(tar_path, 'r') as tar:
        f = tar.extractfile(gt_csv_member)
        gt_csv = csv.reader(io.TextIOWrapper(f, encoding='utf-8'), delimiter=';')
        next(gt_csv)  #skip header
        for row in gt_csv:
            img_path_in_tar = 'Images/' + row[0]
            label = int(row[7])
            entries.append((img_path_in_tar, label))
            labels.append(label)
    return entries, labels, tar_path


#-----------------------------
#PyTorch Dataset
#-----------------------------
class GTSRBDataset(Dataset):
    def __init__(self, entries, tar_path, transform=None):
        self.entries = entries
        self.tar_path = tar_path
        self.transform = transform
        self.labels = [label for _, label in entries]
        
        print(f"Preloading {len(entries)} images into memory...", end='', flush=True)
        self.images = []
        
        with tarfile.open(tar_path, 'r') as tar:
            for img_path, _ in entries:
                member = tar.getmember(img_path)
                f = tar.extractfile(member)
                img = Image.open(f).convert('RGB')
                self.images.append(img)
                f.close()
        
        print(f" Done! ({len(self.images)} images)")
    
    def __len__(self):
        return len(self.entries)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            img = self.transform(img)
        
        return img, label

class ConcreteDataset(Dataset):
    """
    PyTorch Dataset for UCI Concrete Compressive Strength dataset.
    Supports CSV, XLS, and XLSX formats.
    File must have 9 columns: 8 input features + 1 target column
    """
    def __init__(self, file_path):
        """
        Args:
            file_path: path to data file (.csv, .xls, or .xlsx)
        """
        #Determine file type and read accordingly
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xls') or file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format. Must be .csv, .xls, or .xlsx. Got: {file_path}")

        #Last column is the target by default
        self.X = df.iloc[:, :-1].values.astype("float32")
        self.y = df.iloc[:, -1].values.astype("float32").reshape(-1, 1)
        self.input_dim = self.X.shape[1]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx])
        y = torch.tensor(self.y[idx])
        return x, y
    
    def train_val_test_split(self, val_ratio=0.2, test_ratio=0.1, random_state=42):
        """
        Split dataset into train, validation, and test sets.
        
        Args:
            val_ratio: proportion for validation set
            test_ratio: proportion for test set
            random_state: random seed for reproducibility
            
        Returns:
            train_dataset, val_dataset, test_dataset
        """
        from sklearn.model_selection import train_test_split
        
        #First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            self.X, self.y, test_size=test_ratio, random_state=random_state
        )
        
        #Second split: separate train and validation
        val_size_adjusted = val_ratio / (1 - test_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
        )
        
        #Create subset datasets
        train_dataset = ConcreteSubset(X_train, y_train)
        val_dataset = ConcreteSubset(X_val, y_val)
        test_dataset = ConcreteSubset(X_test, y_test)
        
        return train_dataset, val_dataset, test_dataset


class ConcreteSubset(Dataset):
    """Helper class for train/val/test subsets of ConcreteDataset"""
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx])
        y = torch.tensor(self.y[idx])
        return x, y