import json
import os
import re
import base64
from io import BytesIO
from random import randint
import nova_image_gen
import file_utils
import logging
from datetime import datetime
import requests
from retry import retry
from PIL import Image
import numpy as np
import torch
from .session import get_client

MAX_RETRY = 3

bedrock_runtime_client = get_client(service_name="bedrock-runtime")

class BedrockNovaTextImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "num_images": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 5,
                        "step": 1,
                        "round": 1,  # The value represeting the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                        "display": "number",
                    },
                ),
                "quality": (
                    [
                        "standard",
                        "premium",
                    ],
                ),
                "resolution": (
                    [
                        "1024 x 1024",
                        "768 x 768",
                        "512 x 512",
                        "1152 x 768",
                        "576 x 384",
                        "768 x 1280",
                        "384 x 640",
                        "1280 x 768"
                    ],
                ),
                "cfg_scale": (
                    "FLOAT",
                    {
                        "default": 7.0,
                        "min": 0.0,
                        "max": 35.0,
                        "step": 0.1,
                        "round": 0.001,  # The value represeting the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                        "display": "slider",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2147483646,
                        "step": 1,
                        "round": 1,  # The value represeting the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                        "display": "number",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "forward"
    CATEGORY = "aws"

    @retry(tries=MAX_RETRY)
    def forward(self, prompt, num_images, quality, resolution, cfg_scale, seed):

        height, width = map(int, re.findall(r"\d+", resolution))

        # The different model providers have individual request and response formats.
        # For the format, ranges, and default values for Titan Image models refer to:
        # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-image.html

        inference_params =
        {
                "taskType": "TEXT_IMAGE",
                "textToImageParams": {"text": prompt},
                "imageGenerationConfig": {
                    "numberOfImages": num_images,
                    "quality": quality,
                    "cfgScale": cfg_scale,
                    "height": height,
                    "width": width,
                    "seed": seed,
                },
        }

        # Define an output directory with a unique name.
        generation_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_directory = f"output/{generation_id}"

        # Generate the image(s).
        response = nova_image_gen.generate_images(
            inference_params=inference_params,
            model_id="amazon.nova-canvas-v1:0",
            output_directory=output_directory
        )

        response_body = json.loads(response["body"].read())
        # base64_image_data = response_body["images"][0]
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


class BedrockNovaIpAdatper:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True}),
                "negative_prompt": ("STRING", {"multiline": True}),
                "similarity_strength": (
                    "FLOAT",
                    {
                          "default": 0.9,
                          "min": 0.2,
                          "max": 1.0,
                          "step": 0.1,
                          "round": 0.01,  # The value represeting the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                          "display": "slider",
                    },
                ),
                "num_images": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 5,
                        "step": 1,
                        "round": 1,  # The value represeting the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                        "display": "number",
                    },
                ),
                "cfg_scale": (
                    "FLOAT",
                    {
                        "default": 8.0,
                        "min": 1.1,
                        "max": 10.0,
                        "step": 0.1,
                        "round": 0.001,  # The value represeting the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                        "display": "slider",
                    },
                ),
                "resolution": (
                    [
                        "1024 x 1024",
                        "1024 x 1024",
                        "768 x 768",
                        "512 x 512",
                        "1152 x 768",
                        "576 x 384",
                        "768 x 1280"
                    ],
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2147483646,
                        "step": 1,
                        "round": 1,  # The value represeting the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                        "display": "number",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "forward"
    CATEGORY = "aws"

    @retry(tries=MAX_RETRY)
    def forward(self, image, prompt, negative_prompt, similarity_strength, num_images, cfg_scale, resolution):

        image = image[0] * 255.0
        image = Image.fromarray(image.clamp(0, 255).numpy().round().astype(np.uint8))

        height, width = map(int, re.findall(r"\d+", resolution))

        buffer = BytesIO()
        image.save(buffer, format="PNG")

        image_data = buffer.getvalue()

        image_base64 = base64.b64encode(image_data).decode("utf-8")

        # The different model providers have individual request and response formats.
        # For the format, ranges, and default values for Titan Image models refer to:
        # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-image.html

        # add a key value to json only when the a parameter is not empty
        inference_params = {
                "taskType": "IMAGE_VARIATION",
                "inPaintingParams": {
                    "image": image_base64,                         
                    "text": prompt,
                    "negativeText": negative_prompt,        
                    "similarityStrength": similarity_strength,
                },                                                 
                "imageGenerationConfig": {
                    "numberOfImages": num_images,
                    "quality": "premium",
                    "height": height,
                    "width": width,
                    "cfgScale": cfg_scale,
                    "seed": seed
                }
            }
        
        response_body = nova_image_gen.generate_images(
            inference_params=inference_params,
            model_id="amazon.nova-canvas-v1:0",
            output_directory=output_directory
        )

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


class BedrockNovaBackgroundPEReplace:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True}),
                "mask_prompt": ("STRING", {"multiline": True}),
                "num_images": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 5,
                        "step": 1,
                        "round": 1,  # The value represeting the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                        "display": "number",
                    },
                ),
                "cfg_scale": (
                    "FLOAT",
                    {
                        "default": 8.0,
                        "min": 1.1,
                        "max": 10.0,
                        "step": 0.1,
                        "round": 0.001,  # The value represeting the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                        "display": "slider",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2147483646,
                        "step": 1,
                        "round": 1,  # The value represeting the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                        "display": "number",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "forward"
    CATEGORY = "aws"

    @retry(tries=MAX_RETRY)
    def forward(self, image, prompt, mask_prompt, num_images, cfg_scale):
        image = image[0] * 255.0
        image = Image.fromarray(image.clamp(0, 255).numpy().round().astype(np.uint8))


        buffer = BytesIO()
        image.save(buffer, format="PNG")

        image_data = buffer.getvalue()

        image_base64 = base64.b64encode(image_data).decode("utf-8")

        inference_params = {
                "taskType": "OUTPAINTING",
                "inPaintingParams": {
                    "image": image_base64,
                    "text": prompt,
                    "maskPrompt": mask_prompt,
                    "outPaintingMode": "PRECISE",
                },
                "imageGenerationConfig": {
                    "numberOfImages": num_images,
                    "quality": "premium",
                    "cfgScale": cfg_scale,
                    "seed": seed
                }
            }

        response_body = nova_image_gen.generate_images(
            inference_params=inference_params,
            model_id="amazon.nova-canvas-v1:0",
            output_directory=output_directory
        )

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



NODE_CLASS_MAPPINGS = {
    "Bedrock - Nova Text to Image": BedrockNovaTextImage,
    "Bedrock - Nova IpAdapter": BedrockNovaIpAdatper,
    "Bedrock - Nova Background Prompt Replace": BedrockNovaBackgroundPEReplace,
}