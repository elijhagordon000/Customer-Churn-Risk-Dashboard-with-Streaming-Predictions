from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError, ProfileNotFound

# Locate the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Local file produced by the prediction pipeline
LOCAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "predictions"
    /"scored_customers.csv"
)

BUCKET_NAME = "elijha-churn-predictions-2026-738021988978-us-east-2-an"

# Object's location inside the S3 bucket
S3_KEY = "predictions/scored_customers.csv"

PROFILE_NAME = "churn-project"

def upload_prediction_file() -> None:
    """ Upload churn prediction CSV to amazon S3"""

    if not LOCAL_FILE.is_file():
        raise FileNotFoundError(
            f"Prediction file was not found: {LOCAL_FILE}"
        )
    
    try:
        # Creating an AWS session using IAM profile
        session = boto3.Session(profile_name=PROFILE_NAME)

        # Create an S3 client through that authenticated session.
        s3_client = session.client("s3")

        print(f"Uploading: {LOCAL_FILE}")
        print(f"Destination: s3://{BUCKET_NAME}/{S3_KEY}")

        s3_client.upload_file(
            Filename=str(LOCAL_FILE),
            Bucket=BUCKET_NAME,
            Key=S3_KEY,
        )

        print("Upload completed successfully.")

    except ProfileNotFound as error:
        print(
            f"AWS profile '{PROFILE_NAME}' was not found"
            "Run AWS login again"
        )
        raise error
    
    except (BotoCoreError, ClientError) as error:
        print(f"AWS upload file failed: {error}")
        raise

if __name__ == "__main__":
    upload_prediction_file()