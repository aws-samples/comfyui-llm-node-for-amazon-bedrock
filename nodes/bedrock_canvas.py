import json
import os
import re
import base64
from io import BytesIO
import requests
from retry import retry
from PIL import Image
import numpy as np
import torch
from .session import get_client
from .utils import image_to_base64
import logging
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

bedrock_runtime_client = get_client(service_name="bedrock-runtime")


class BedrockNovaCanvasTextImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "number_of_images": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 5,
                        "step": 1,
                        "display": "number",
                    },
                ),
                "resolution": (
                    [
                        "1024 x 1024",
                        "2048 x 2048",
                        "1024 x 336",
                        "1024 x 512",
                        "1024 x 576",
                        "1024 x 627",
                        "1024 x 816",
                        "1280 x 720",
                        "2048 x 512",
                        "2288 x 1824",
                        "2512 x 1664",
                        "2720 x 1520",
                        "2896 x 1440",
                        "3536 x 1168",
                        "4096 x 1024",
                        "336 x 1024",
                        "512 x 1024",
                        "512 x 2048",
                        "576 x 1024",
                        "672 x 1024",
                        "720 x 1280",
                        "816 x 1024",
                        "1024 x 4096",
                        "1168 x 3536",
                        "1440 x 2896",
                        "1520 x 2720",
                        "1664 x 2512",
                        "1824 x 2288",
                    ],
                ),
                "cfg_scale": ("FLOAT", {
                    "default": 6.5,
                    "min": 1.1,
                    "max": 10.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                "seed": ("INT", {
                    "default": 12,
                    "min": 0,
                    "max": 858993459,
                    "step": 1,
                    "display": "number"
                }),
                "quality": (
                    [
                        "standard",
                        "premium",
                    ],
                ),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "forward"
    CATEGORY = "aws"


    def validate_input(self, prompt, width, height):
        if not (width % 16 == 0 and height % 16 == 0):
            raise ValueError("Width and height must be divisible by 16")

        if not (320 <= width <= 4096 and 320 <= height <= 4096):
            raise ValueError("Dimensions must be between 320 and 4096 pixels")

        if (width * height) > 4194304:
            raise ValueError("Total pixel count exceeds maximum allowed (4.19M)")

        aspect_ratio = width / height
        if not (0.25 <= aspect_ratio <= 4.0):
            raise ValueError("Aspect ratio must be between 1:4 and 4:1")

        if not (1 <= len(prompt) <= 1024):
            raise ValueError("Prompt must be between 1 and 1024 characters")

    @retry(exceptions=ClientError, tries=3, delay=1, backoff=2)
    def forward(
        self,
        prompt,
        number_of_images,
        resolution,
        cfg_scale,
        seed,
        negative_prompt=None,
        quality="standard",
    ):
        """
        A ComfyUI node class for Amazon Bedrock Nova Canvas text-to-image generation.
        Invokes the Nova Canvas Text-to-Image model API to generate an image using the input provided in the request body.

        Supported Parameters:
        - Text to image generation with optional negative prompts to avoid certain aspects
        - Multiple image generation (1-5 images)
        - Various preset resolutions
        - CFG scale adjustment (1.1-10.0)
        - Seed control for reproducibility
        - Quality selection (standard/premium)
        """

        height, width = map(int, re.findall(r"\d+", resolution))
        self.validate_input(prompt, width, height)

        request_json = {
            "taskType": "TEXT_IMAGE",
            "textToImageParams": {
                "text": prompt,
                "negativeText": negative_prompt if negative_prompt else "bad quality"
            },
            "imageGenerationConfig": {
                "numberOfImages": number_of_images,
                "quality": quality,
                "height": height,
                "width": width,
                "cfgScale": cfg_scale,
                "seed": seed
            }
        }

        if negative_prompt:
            request_json["textToImageParams"]["negativeText"] = negative_prompt

        try:
            response = bedrock_runtime_client.invoke_model(
                body=json.dumps(request_json),
                modelId="amazon.nova-canvas-v1:0",
                accept="application/json",
                contentType="application/json"
            )

            response_body = json.loads(response.get("body").read())

            if "error" in response_body:
                raise ImageError(f"Image generation error: {response_body['error']}")

            images = [
                np.array(Image.open(BytesIO(base64.b64decode(base64_image))))
                for base64_image in response_body.get("images")
            ]
            images = [
                torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)
                for image in images
            ]
            if len(images) == 1:
                images = images[0]
            else:
                images = torch.cat(images, 0)
            return (images,)


        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'ThrottlingException':
                logging.error("Rate limit exceeded")
            elif error_code == 'ValidationException':
                logging.error("Invalid parameter value")
            else:
                logging.error(f"AWS Bedrock error: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}")
            raise

class ImageError(Exception):
    "Custom exception for errors returned by Amazon Nova Canvas"

    def __init__(self, message):
        self.message = message

NODE_CLASS_MAPPINGS = {
    "Bedrock - Amazon Nova Canvas Text to Image": BedrockNovaCanvasTextImage,
}