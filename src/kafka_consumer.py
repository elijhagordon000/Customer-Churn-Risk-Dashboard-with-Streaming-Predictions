from pathlib import Path
import json
import pandas as pd
import torch
from kafka import KafkaConsumer

# Project Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PREDICTIONS_DIR = PROJECT_ROOT / "data" / "predictions"
PREDICTIONS_DIR.mkdir(exist_ok=True)
# A topic is like a mailbox where messages are placed
TOPIC_NAME = "customer-churn-input" # Gives a name to the Kafka topic the consumer will listen to. 
OUTPUT_PATH = PREDICTIONS_DIR / "stream_scored_customers.csv" # Where the consumer will append scored results

# Load X_test just to recover the feature column order
X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv") # Loads processed feature data. Not using it to score the whole file directly instead using it to reover exact column order the model expects
feature_columns = list(X_test.columns) # Stores the the feature names in order because the Kafka message is a dictionary and the model expects the same features in the order they were trained on

# Rebuilds the model the ssame neural network structure used in training: input layer, hidden layer with ReLU and output layer
# Neede because the weights are saved and not the full model object
model = torch.nn.Sequential(
    torch.nn.Linear(len(feature_columns), 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64,1)
)

model.load_state_dict(torch.load(MODELS_DIR / "churn_model.pt")) # loads train weights into the rebuilt model
model.eval() # Puts model in evaluation mode

#Create Kafka consumer that listens for incoming messages
consumer = KafkaConsumer( # creates Kafka consumer that listens for incoming messages
    TOPIC_NAME, # tells which topic to read from
    bootstrap_servers="localhost:9092", # where kafka is running
    auto_offset_reset="earliest", # If the consumer has not read the topic before it starts from the earliest available messages
    enable_auto_commit=True, # Kafka will remember what messages the consumer has already used
    value_deserializer=lambda m: json.loads(m.decode("utf-8")) # Kafka message arrives as bytes so this converts the bytes back into a Python dict
)

# If file doesn't exist yet, write header later
# if file is new: write to header now
# If file already exists: append more rows
first_write = not OUTPUT_PATH.exists() 

print("Consumer is listening for messages...")

with torch.no_grad(): # No tracking gradients because this is prediction and not training
    for message in consumer: # keeps listening for messages
        row_dict = message.value # Extracts the python dictionary from the kafka message

        row_id = row_dict.pop("row_id", None) # Removes row_id field from the dictionary and stores it seperately; because row_id is not a feature the model was trained on

        # Rebuilds the row in the exact column order the model expects
        ordered_values = [row_dict[col] for col in feature_columns]

        # Convert to tensor with shape (1, num_features)
        row_tensor = torch.tensor([ordered_values], dtype=torch.float32)

        # Score row
        logit = model(row_tensor) # runs the customer row through model
        prob = torch.sigmoid(logit).item()# turns logit into a probability 
        pred = int(prob >= 0.5) # turns probability into a binary churn prediction

        scored_row =pd.DataFrame([{ # creates a dataframe containing the scored result
            "row_id": row_id,
            "churn_probability": prob,
            "predicted_churn": pred
        }])

        scored_row.to_csv( # appends row to streaming output CSV
            OUTPUT_PATH,
            mode="a", # Append mode 
            header=first_write, # Only write header if first row is being saved
            index=False # Prevents writing of panda row numbers into the CSV
        )
        first_write = False # After the first row is written future writes should not include the header again

        print(f"scored row {row_id}: prob={prob:.4f}, pred={pred}")
