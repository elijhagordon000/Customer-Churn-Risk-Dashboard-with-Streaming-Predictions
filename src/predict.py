import torch
from pathlib import Path
import pandas as pd


# Project rooot = one level above src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PREDICTIONS_DIR = PROJECT_ROOT / "data" / "predictions"
PREDICTIONS_DIR.mkdir(exist_ok=True)

X_test = pd.read_csv(PROCESSED_DIR/ "X_test.csv")
readable_X_test = pd.read_csv(PROCESSED_DIR / "readable_X_test.csv")

model = torch.nn.Sequential(
        torch.nn.Linear(X_test.shape[1], 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64,1)
)

model.load_state_dict(torch.load(MODELS_DIR / "churn_model.pt"))
model.eval()

# Converting X_test to a tensor:
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32) 
# forward pass
with torch.no_grad(): # Turn off Gradient tracking
    test_outputs = model(X_test_tensor)
    test_probs = torch.sigmoid(test_outputs) # turns logits into probabilities between 0 and 1
    test_preds = (test_probs >= 0.5).int() # turns those probabilities into predicted labels
    # converting tensor into pandas DF
    test_probs_df = pd.DataFrame(test_probs.flatten().numpy(), columns=['churn_probability'])
    test_preds_df = pd.DataFrame(test_preds.flatten().numpy(), columns=['predicted_churn'])
    # combing dataframe:
    scored_df = pd.concat([test_probs_df, test_preds_df], axis=1)
    # adding relevant features to scored_df
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    readable_X_test = pd.read_csv(PROCESSED_DIR / "readable_X_test.csv")

    result_df = pd.concat([readable_X_test, scored_df], axis=1)
    result_df.to_csv(PREDICTIONS_DIR / "scored_customers.csv", index=False)

print(result_df.head())
