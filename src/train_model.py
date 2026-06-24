import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn

# Project rooot = one level above src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODELS_DIR / "churn_model.pt"
PLOT_PATH = MODELS_DIR / "training_loss.png"

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

criterion = nn.BCEWithLogitsLoss() #Binary classification loss function

# Adam = the optimization algorithm
# model.parameters() = tells PyTorch which weights/biases to update
# lr=0.001 = learning rate, meaning how big the update steps should be
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_losses = []
# Training Loop:

epochs = 5

for epoch in range(epochs):
    model.train()

    optimizer.zero_grad() # Clears out gradients from the previous step.

    outputs = model(X_train_tensor) # This is the forward pass.; The model takes inputs and produces predictions.

    loss = criterion(outputs, Y_train_tensor) # Now compares predictions to the true labels.

    loss.backward() # PyTorch computes gradients.

    optimizer.step() # Adam updates the model weights and biases.

    train_losses.append(loss.item())

    #Evaluation:

    model.eval() # Switch to Eval mode
    with torch.no_grad(): # Turn off Gradient tracking
        test_outputs = model(X_test_tensor) # Making predictions on X_test_Tensor
        test_loss = criterion(test_outputs, Y_test_tensor) # compute test loss using Y_test_tensor

        test_probs = torch.sigmoid(test_outputs) # turns logits into probabilities between 0 and 1
        test_preds = (test_probs >= 0.5).float() # turns those probabilities into predicted labels

        accuracy = (test_preds == Y_test_tensor).float().mean() # comparing to Y_test_tensor gives True/False; converting to float makes those become 1.0/0.0 ; .mean() gives the proportion correct
    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Train Loss: {loss.item():.4f}")
    print(f"Test Loss: {test_loss.item():.4f}")
    print(f"Accuracy: {accuracy.item():.4f}")
    print(f"Predicted positive rate: {test_preds.mean().item():.4f}")
    print(f"Actual positive rate: {Y_test_tensor.mean().item():.4f}")

torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved to: {MODEL_PATH}")

plt.plot(range(1, epochs + 1), train_losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig(PLOT_PATH)
plt.show()

print(f"Training loss plot saved to: {PLOT_PATH}")
    



