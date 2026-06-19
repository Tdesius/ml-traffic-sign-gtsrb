# Introduction to Machine Learning — TDTU Final Term

Final term project for the **Introduction to Machine Learning** course at Ton Duc Thang University, covering two tasks: regression with a Feed-Forward Neural Network (FNN) and image classification with a Convolutional Neural Network (CNN).

> Advised by Dr. Tran Luong Quoc Dai · Academic Year 2025–2026

---

## Task 1 — FNN Regression with Backpropagation & Early Stopping

**Dataset:** [UCI Concrete Compressive Strength](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength) (1,030 samples, 8 input features → 1 continuous target) — credit: I-Cheng Yeh, UCI Machine Learning Repository

Implemented a Feed-Forward Neural Network from scratch in PyTorch with manual backpropagation to predict concrete compressive strength (MPa). Compared training with and without Early Stopping to demonstrate overfitting control.

### Architecture

```
Input (8) → FC(128) → ReLU → FC(64) → ReLU → Output (1)
Loss: MSELoss · Optimizer: Adam · LR: 0.001
```

### Results

| Model | Train Loss (MSE) | Val Loss (MSE) | Epochs | Train–Val Gap |
|---|---|---|---|---|
| FNN (no early stopping) | 24.89 | 47.58 | 200 | 22.69 |
| FNN (with early stopping) | 33.29 | 49.25 | **58** | **15.96** |

Early stopping halted training at epoch 58, eliminating 142 unnecessary epochs and reducing the train–val gap by **30%**, demonstrating significantly reduced overfitting.

---

## Task 2 — CNN Classification on GTSRB (43 Traffic Sign Classes)

**Dataset:** [GTSRB — German Traffic Sign Recognition Benchmark](https://benchmark.ini.rub.de/gtsrb_news.html)

Built and compared three CNN configurations on 43-class traffic sign classification, progressively adding regularization techniques to reduce overfitting.

### Architecture (Baseline)

```
Conv(3→32, 3×3) → ReLU → MaxPool(2×2)
Conv(32→64, 3×3) → ReLU → MaxPool(2×2)
FC(64×8×8 → 256) → ReLU → FC(256 → 43)
```

### Results

| Model | Train Acc | Val Acc | Test Acc | Train–Test Gap | Time |
|---|---|---|---|---|---|
| Baseline CNN | 99.66% | 98.74% | 91.19% | 8.47% | 229s |
| CNN + Augmentation | 98.84% | 99.53% | 93.51% | 5.33% | 344s |
| **CNN + Dropout** | **99.16%** | **99.53%** | **94.84%** | **4.32%** | 287s |

**Key finding:** Dropout outperformed data augmentation both in accuracy (+1.33%) and training efficiency (1.25× baseline vs 1.5×). It also produced the cleanest confusion matrix diagonal across all 43 classes.

| Regularization Method | Test Acc Gain | Gap Reduction |
|---|---|---|
| Baseline → Augmentation | +2.32% | −3.14% |
| Baseline → Dropout | +3.65% | −4.15% |

---

## Project Structure

```
ML-Traffic-Sign-GTSRB/
├── source code/
│   ├── main.py                   # Entry point — runs all experiments
│   ├── models/
│   │   ├── CNN.py                # BaselineCNN and DropoutCNN architectures
│   │   ├── FNN.py                # BaselineFNN for regression
│   │   ├── dataset.py             # GTSRBDataset and ConcreteDataset loaders
│   │   └── train_utils.py         # Training loops for classification and regression
│   ├── visualization/
│   │   └── viz_utils.py           # Plotting utilities (gitignored: visualization/plots/)
│   ├── saved_models/               # Trained checkpoints (gitignored)
│   └── Concrete_Data.xls           # UCI Concrete Compressive Strength dataset
├── requirements.txt
├── install_requirements.bat
└── README.md
```

---

## Setup & Usage

```bash
pip install torch torchvision pandas scikit-learn matplotlib seaborn pillow
python main.py
```

Datasets are loaded automatically. GTSRB requires downloading the `.tar` files from the [official benchmark site](https://benchmark.ini.rub.de/gtsrb_news.html) and placing them in the project root.

---

## Authors

| Name | Student ID |
|---|---|
| Pham Quoc Hung | 523H0135 |
| Dinh Bui Khanh Huy | 523H0136 |
| Nguyen Dong Quan | 523H0171 |

*Ton Duc Thang University — Faculty of Information Technology*