import os
from pathlib import Path

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