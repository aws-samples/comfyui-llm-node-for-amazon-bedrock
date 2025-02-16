import json
import os
from io import BytesIO
from datetime import datetime
import boto3
from PIL import Image
import numpy as np
import time

MAX_RETRY = 3

def get_default_region():
    return "us-west-2"  # Use us-west-2 for Luma AI Ray 2

s3_client = boto3.client("s3", region_name=get_default_region())


def get_account_id():
    sts_client = boto3.client("sts", region_name=get_default_region())
    return sts_client.get_caller_identity().get("Account")


def is_video_downloaded_for_invocation_job(invocation_job, output_folder="output"):
    invocation_arn = invocation_job["invocationArn"]
    invocation_id = invocation_arn.split("/")[-1]
    folder_name = get_folder_name_for_job(invocation_job)
    output_folder = os.path.abspath(f"{output_folder}/{folder_name}")
    file_name = f"{invocation_id}.mp4"
    local_file_path = os.path.join(output_folder, file_name)
    return os.path.exists(local_file_path)


def get_folder_name_for_job(invocation_job):
    invocation_arn = invocation_job["invocationArn"]
    invocation_id = invocation_arn.split("/")[-1]
    submit_time = invocation_job["submitTime"]
    timestamp = submit_time.astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"{timestamp}_{invocation_id}"
    return folder_name


def download_video_for_invocation_arn(invocation_arn, bucket_name, destination_folder):
    invocation_id = invocation_arn.split("/")[-1]
    file_name = f"{invocation_id}.mp4"
    output_folder = os.path.abspath(destination_folder)
    local_file_path = os.path.join(output_folder, file_name)
    os.makedirs(output_folder, exist_ok=True)
    s3 = boto3.client("s3", region_name=get_default_region())
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=invocation_id)
    for obj in response.get("Contents", []):
        object_key = obj["Key"]
        if object_key.endswith(".mp4"):
            print(f'Downloading "{object_key}"...')
            s3.download_file(bucket_name, object_key, local_file_path)
            print(f"Downloaded to {local_file_path}")
            return local_file_path
    print(f"Problem: No MP4 file was found in S3 at {bucket_name}/{invocation_id}")


def get_job_id_from_arn(invocation_arn):
    return invocation_arn.split("/")[-1]


def save_completed_job(job, output_folder="output"):
    job_id = get_job_id_from_arn(job["invocationArn"])
    output_folder_abs = os.path.abspath(
        f"{output_folder}/{get_folder_name_for_job(job)}"
    )
    os.makedirs(output_folder_abs, exist_ok=True)
    if is_video_downloaded_for_invocation_job(job, output_folder=output_folder):
        print(f"Skipping completed job {job_id}, video already downloaded.")
        return
    s3_bucket_name = (
        job["outputDataConfig"]["s3OutputDataConfig"]["s3Uri"]
        .split("//")[1]
        .split("/")[0]
    )
    localPath = download_video_for_invocation_arn(
        job["invocationArn"], s3_bucket_name, output_folder_abs
    )
    return localPath


bedrock_runtime_client = boto3.client(
    "bedrock-runtime", region_name=get_default_region()
)
region = get_default_region()
account_id = get_account_id()

class BedrockLumaVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "aspect_ratio": (
                    ["16:9", "1:1", "3:4", "4:3", "9:16", "21:9", "9:21"],
                ),
                "resolution": (["540p", "720p"],),
                "duration": (["5s", "9s"],),
                "destination_bucket": (
                    [b["Name"] for b in s3_client.list_buckets()["Buckets"]],),
                "loop_video": (["False", "True"],),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "forward"
    CATEGORY = "aws"

    def forward(self, **kwargs):
        prompt = kwargs.get("prompt")
        aspect_ratio = kwargs.get("aspect_ratio")
        resolution = kwargs.get("resolution")
        duration = kwargs.get("duration")
        loop_video = kwargs.get("loop_video")
        s3_destination_bucket = kwargs.get("destination_bucket")

        model_input_body = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "loop": loop_video,
            "duration": duration,
        }

        # Start asynchronous invocation
        invocation_response = bedrock_runtime_client.start_async_invoke(
            modelId="luma.ray-v2:0",
            modelInput=model_input_body,
            outputDataConfig={
                "s3OutputDataConfig": {"s3Uri": f"s3://{s3_destination_bucket}"}
            },
        )

        invocation_arn = invocation_response["invocationArn"]
        print("\nInvocation Response:")
        print(json.dumps(invocation_response, indent=2))

        # Poll for job completion
        save_local_path = ""
        while True:
            job_update_response = bedrock_runtime_client.get_async_invoke(
                invocationArn=invocation_arn
            )
            status = job_update_response["status"]

            if status == "Completed":
                save_local_path = save_completed_job(
                    job_update_response, 
                    output_folder=os.path.expanduser("~/ComfyUI/output/")
                )
                break
            elif status == "Failed":
                print(f"Job failed: {job_update_response}")
                raise Exception(f"Job failed with details: {job_update_response}")
            else:
                time.sleep(15)

        return (save_local_path,)


NODE_CLASS_MAPPINGS = {"Bedrock - Luma AI Ray Video": BedrockLumaVideo}
