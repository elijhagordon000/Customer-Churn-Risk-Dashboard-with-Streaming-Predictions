import torch
from pathlib import Path
import pandas as pd


# Project rooot = one level above src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"



X_test = pd.read_csv(PROCESSED_DIR/ "X_test.csv")

model = torch.nn.Sequential(
    torch.nn.Sequential(
        torch.nn.Linear(X_test.shape[1], 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64,1)
    )
)

model.load_state_dict(torch.load(MODELS_DIR / "churn_model.pt"))
model.eval()
