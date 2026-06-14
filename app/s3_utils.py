import os
from pathlib import Path
from datetime import datetime
from botocore.exceptions import ClientError
import boto3



BASE_DIR = Path(__file__).resolve().parent.parent

ARTIFACTS_DIR = BASE_DIR / "artifacts"

ARTIFACTS_DIR.mkdir(
    exist_ok=True
)


def download_model_from_s3(model_name: str):

    bucket_name = os.getenv(
        "AWS_BUCKET_NAME"
    )

    s3 = boto3.client(
        "s3",
        region_name=os.getenv(
            "AWS_REGION"
        )
    )

    local_model_path = (
        ARTIFACTS_DIR / model_name
    )

    s3.download_file(
        bucket_name,
        f"models/{model_name}",
        str(local_model_path)
    )

    return local_model_path


def append_prediction_to_s3(
    environment: str,
    prediction: int,
    label: str
):

    bucket_name = os.getenv(
        "AWS_BUCKET_NAME"
    )

    s3 = boto3.client(
        "s3",
        region_name=os.getenv(
            "AWS_REGION"
        )
    )

    file_name = (
        "predictions_dev.txt"
        if environment == "dev"
        else "predictions_prod.txt"
    )

    s3_key = (
        f"predictions/{file_name}"
    )

    timestamp = datetime.utcnow().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    new_line = (
        f"{timestamp},"
        f"prediction={prediction},"
        f"label={label}\n"
    )

    try:

        response = s3.get_object(
            Bucket=bucket_name,
            Key=s3_key
        )

        content = (
            response["Body"]
            .read()
            .decode("utf-8")
        )

    except ClientError:

        content = ""

    content += new_line

    s3.put_object(
        Bucket=bucket_name,
        Key=s3_key,
        Body=content.encode("utf-8")
    )