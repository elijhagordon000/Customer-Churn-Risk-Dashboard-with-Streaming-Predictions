from pathlib import Path
import json
import pandas as pd
from kafka import KafkaProducer

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TOPIC_NAME = "customer-churn-input"

# Load processed test data
X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")

#Create Kafka Producer
producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8") # Kafka sends bytes not Python dicts so need to convert each message to JSON bytes.
)

# Send one customer row at a time
for idx, row in X_test.iterrows():
    message = row.to_dict() # Turns one pandas row into a normal Python dictionary, which is easy to serialize to JSON.

    message["row_id"] = int(idx) # Gives each message an identifier so the consumer can keep track of which row it scored.

    producer.send(TOPIC_NAME, value=message)
    print(f"Sent row {idx}")

# Make sure all messages are actually sent
producer.flush() # Makes sure Kafka actually sends everything before the script exits.
producer.close()

print("Finished sending test rows to Kafka.")