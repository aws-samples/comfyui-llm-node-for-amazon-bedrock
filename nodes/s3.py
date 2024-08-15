import json
import os
import re
import base64
from io import BytesIO

import requests
from retry import retry
import boto3


from PIL import Image
import numpy as np
import torch


from .session import get_client


s3_client = get_client(service_name="s3")

MAX_RETRY = 3


class ImageFromURL:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "url": ("STRING", {"multiline": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "forward"
    CATEGORY = "aws"

    def download_s3(self, bucket, key):
        response = s3_client.get_object(Bucket=bucket, Key=key)
        image = Image.open(response["Body"])
        image = torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(
            0
        )
        return image

    def download_http(self, url):
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Cache-Control": "max-age=0",
        }
        request = requests.get(url, headers=headers)
        image = Image.open(BytesIO(request.content))
        image = torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(
            0
        )
        return image

    @retry(tries=MAX_RETRY)
    def forward(self, url):
        if url.startswith("s3://"):
            bucket, key = url.split("s3://")[1].split("/", 1)
            image = self.download_s3(bucket, key)
        elif re.match(r"^https://.*\.s3\..*\.amazonaws\.com/.*", url):
            _, _, bucket, key = url.split("/", 3)
            bucket = bucket.split(".")[0]
            image = self.download_s3(bucket, key)
        elif url.startswith("http"):
            image = self.download_http(url)
        else:
            raise ValueError("Invalid URL")
        return (image,)


class ImageFromS3:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "bucket": ([b["Name"] for b in s3_client.list_buckets()["Buckets"]],),
                "key": ("STRING", {"multiline": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "forward"
    CATEGORY = "aws"

    def download_s3(self, bucket, key):
        response = s3_client.get_object(Bucket=bucket, Key=key)
        image = Image.open(response["Body"])
        image = torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(
            0
        )
        return image

    @retry(tries=MAX_RETRY)
    def forward(self, bucket, key):
        image = self.download_s3(bucket, key)
        return (image,)


class ImageToS3:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "bucket": ([b["Name"] for b in s3_client.list_buckets()["Buckets"]],),
                "key": ("STRING", {"multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "forward"
    CATEGORY = "aws"
    OUTPUT_NODE = True

    def upload_s3(self, image, bucket, key):
        image = image[0] * 255.0
        image = Image.fromarray(image.clamp(0, 255).numpy().round().astype(np.uint8))
        buffer = BytesIO()
        image.save(buffer, format=key.split(".")[-1])
        buffer.seek(0)
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=buffer,
            ContentType=f"image/{key.split('.')[-1].lower()}",
        )

    @retry(tries=MAX_RETRY)
    def forward(self, image, bucket, key):
        self.upload_s3(image, bucket, key)
        return (f"s3://{bucket}/{key}",)


NODE_CLASS_MAPPINGS = {
    "Image From URL": ImageFromURL,
    "Image From S3": ImageFromS3,
    "Image To S3": ImageToS3,
}
