from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn

# Project rooot = one level above src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
X_test = pd.read_csv(PROCESSED_DIR/ "X_test.csv")
Y_train =  pd.read_csv(PROCESSED_DIR/ "Y_train.csv")
Y_test = pd.read_csv(PROCESSED_DIR/ "Y_test.csv")

#Convert to PyTorch Tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32) # torch.Size([5634, 38])
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32) # torch.Size([1409, 38])
Y_train_tensor = torch.tensor(Y_train.values, dtype=torch.float32) # torch.Size([5634, 1])
Y_test_tensor = torch.tensor(Y_test.values, dtype=torch.float32) # torch.Size([1409, 1])

# Building a neural net

model = nn.Sequential(
    nn.Linear(X_train_tensor.shape[1], 64), # Input layer 38 -> 64 (64 is a hyperparameter)
    nn.ReLU(), # Hidden Layer
    nn.Linear(64,1) # Output layer 64 -> 1
)

