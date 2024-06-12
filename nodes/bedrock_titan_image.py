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

MAX_RETRY = 3

bedrock_runtime_client = get_client(service_name="bedrock-runtime")

class BedrockTitanTextImage:
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
                        "768 x 1152",
                        "384 x 576",
                        "1152 x 768",
                        "576 x 384",
                        "768 x 1280",
                        "384 x 640",
                        "1280 x 768",
                        "640 x 384",
                        "896 x 1152",
                        "448 x 576",
                        "1152 x 896",
                        "576 x 448",
                        "768 x 1408",
                        "384 x 704",
                        "1408 x 768",
                        "704 x 384",
                        "640 x 1408",
                        "320 x 704",
                        "1408 x 640",
                        "704 x 320",
                        "1152 x 640",
                        "1173 x 640",
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
        """
        Invokes the Titan Image model to create an image using the input provided in the request body.

        :param prompt: The prompt that you want Amazon Titan to use for image generation.
        :param seed: Random noise seed (range: 0 to 2147483647)
        :return: Base64-encoded inference response from the model.
        """

        height, width = map(int, re.findall(r"\d+", resolution))

        # The different model providers have individual request and response formats.
        # For the format, ranges, and default values for Titan Image models refer to:
        # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-image.html

        request = json.dumps(
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
            },
            ensure_ascii=False,
        )

        response = bedrock_runtime_client.invoke_model(
            modelId="amazon.titan-image-generator-v1", body=request
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


class BedrockTitanInpainting:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True}),
                "negative_prompt": ("STRING", {"multiline": True}),
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
                "resolution": (
                    [
                        "1024 x 1024",
                        "768 x 768",
                        "512 x 512",
                        "768 x 1152",
                        "384 x 576",
                        "1152 x 768",
                        "576 x 384",
                        "768 x 1280",
                        "384 x 640",
                        "1280 x 768",
                        "640 x 384",
                        "896 x 1152",
                        "448 x 576",
                        "1152 x 896",
                        "576 x 448",
                        "768 x 1408",
                        "384 x 704",
                        "1408 x 768",
                        "704 x 384",
                        "640 x 1408",
                        "320 x 704",
                        "1408 x 640",
                        "704 x 320",
                        "1152 x 640",
                        "1173 x 640",
                    ],
                ),
            },
            "optional": {
                "mask_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "forward"
    CATEGORY = "aws"

    @retry(tries=MAX_RETRY)
    def forward(self, image, prompt, 
                negative_prompt, mask_prompt, 
                num_images, cfg_scale, resolution, **kwargs):
        """
        Invokes the Titan Image model to create an image using the input provided in the request body.

        :param prompt: The prompt that you want Amazon Titan to use for image generation.
        :param seed: Random noise seed (range: 0 to 2147483647)
        :return: Base64-encoded inference response from the model.
        """

        height, width = map(int, re.findall(r"\d+", resolution))
        image_base64 = image_to_base64(image)

        maskimage_base64 = ""
        if "mask_image" in kwargs:
            maskimage_base64 = image_to_base64(kwargs["mask_image"])


        # The different model providers have individual request and response formats.
        # For the format, ranges, and default values for Titan Image models refer to:
        # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-image.html

        # add a key value to json only when the a parameter is not empty
        request_json = {
                "taskType": "INPAINTING",
                "inPaintingParams": {
                    "image": image_base64,                         
                    "text": prompt,
                    "negativeText": negative_prompt,        
                    "maskPrompt": mask_prompt,
                    "maskImage": maskimage_base64,                                   
                },                                                 
                "imageGenerationConfig": {
                    "numberOfImages": num_images,
                    "height": height,
                    "width": width,
                    "cfgScale": cfg_scale
                }
            }
        
        # remove empty key value pairs (optional parameters) in the nested json
        for k, v in request_json.items():
            if isinstance(v, dict):
                request_json[k] = {k2: v2 for k2, v2 in v.items() if v2}

        request = json.dumps(request_json, ensure_ascii=False)

        response = bedrock_runtime_client.invoke_model(
            modelId="amazon.titan-image-generator-v1", body=request
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


class BedrockTitanOutpainting:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True}),
                "negative_prompt": ("STRING", {"multiline": True}),
                "mask_prompt": ("STRING", {"multiline": True}),
                "out_paint_mode": (
                    ["DEFAULT", "PRECISE"],
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
                        "768 x 768",
                        "512 x 512",
                        "768 x 1152",
                        "384 x 576",
                        "1152 x 768",
                        "576 x 384",
                        "768 x 1280",
                        "384 x 640",
                        "1280 x 768",
                        "640 x 384",
                        "896 x 1152",
                        "448 x 576",
                        "1152 x 896",
                        "576 x 448",
                        "768 x 1408",
                        "384 x 704",
                        "1408 x 768",
                        "704 x 384",
                        "640 x 1408",
                        "320 x 704",
                        "1408 x 640",
                        "704 x 320",
                        "1152 x 640",
                        "1173 x 640",
                    ],
                ),
            },
            "optional": {
                "mask_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "forward"
    CATEGORY = "aws"

    @retry(tries=MAX_RETRY)
    def forward(self, image, prompt, 
                negative_prompt, mask_prompt, out_paint_mode, 
                num_images, cfg_scale, resolution, **kwargs):
        """
        Invokes the Titan Image model to create an image using the input provided in the request body.

        :param prompt: The prompt that you want Amazon Titan to use for image generation.
        :param seed: Random noise seed (range: 0 to 2147483647)
        :return: Base64-encoded inference response from the model.
        """

        height, width = map(int, re.findall(r"\d+", resolution))
        image_base64 = image_to_base64(image)

        maskimage_base64 = ""
        # if kwargs has mask_image in the key
        if "mask_image" in kwargs:
            maskimage_base64 = image_to_base64(kwargs["mask_image"])


        # The different model providers have individual request and response formats.
        # For the format, ranges, and default values for Titan Image models refer to:
        # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-image.html

        # add a key value to json only when the a parameter is not empty
        request_json = {
                "taskType": "OUTPAINTING",
                "outPaintingParams": {
                    "image": image_base64,                         
                    "text": prompt,
                    "negativeText": negative_prompt,        
                    "maskPrompt": mask_prompt,
                    "maskImage": maskimage_base64, 
                    "outPaintingMode": out_paint_mode,                                  
                },                                                 
                "imageGenerationConfig": {
                    "numberOfImages": num_images,
                    "height": height,
                    "width": width,
                    "cfgScale": cfg_scale
                }
            }
        
        # remove empty key value pairs (optional parameters) in the nested json
        for k, v in request_json.items():
            if isinstance(v, dict):
                request_json[k] = {k2: v2 for k2, v2 in v.items() if v2}

        request = json.dumps(request_json, ensure_ascii=False)

        response = bedrock_runtime_client.invoke_model(
            modelId="amazon.titan-image-generator-v1", body=request
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


class BedrockTitanVariation:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True}),
                "negative_prompt": ("STRING", {"multiline": True}),
                "similarity": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": 0.2,
                        "max": 1.0,
                        "step": 0.1,
                        "round": 0.1,  # The value represeting the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
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
                        "768 x 768",
                        "512 x 512",
                        "768 x 1152",
                        "384 x 576",
                        "1152 x 768",
                        "576 x 384",
                        "768 x 1280",
                        "384 x 640",
                        "1280 x 768",
                        "640 x 384",
                        "896 x 1152",
                        "448 x 576",
                        "1152 x 896",
                        "576 x 448",
                        "768 x 1408",
                        "384 x 704",
                        "1408 x 768",
                        "704 x 384",
                        "640 x 1408",
                        "320 x 704",
                        "1408 x 640",
                        "704 x 320",
                        "1152 x 640",
                        "1173 x 640",
                    ],
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "forward"
    CATEGORY = "aws"

    @retry(tries=MAX_RETRY)
    def forward(self, image, prompt, 
                negative_prompt, similarity,
                num_images, cfg_scale, resolution):
        """
        Invokes the Titan Image model to create an image using the input provided in the request body.

        :param prompt: The prompt that you want Amazon Titan to use for image generation.
        :param seed: Random noise seed (range: 0 to 2147483647)
        :return: Base64-encoded inference response from the model.
        """

        height, width = map(int, re.findall(r"\d+", resolution))
        image_base64 = image_to_base64(image)

        maskimage_base64 = ""


        # The different model providers have individual request and response formats.
        # For the format, ranges, and default values for Titan Image models refer to:
        # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-image.html

        # add a key value to json only when the a parameter is not empty
        request_json = {
                "taskType": "IMAGE_VARIATION",
                "imageVariationParams": {
                    "images": [image_base64],                         
                    "text": prompt,
                    "negativeText": negative_prompt,        
                    "similarityStrength": similarity,                                 
                },                                                 
                "imageGenerationConfig": {
                    "numberOfImages": num_images,
                    "height": height,
                    "width": width,
                    "cfgScale": cfg_scale
                }
            }
        
        # remove empty key value pairs (optional parameters) in the nested json
        for k, v in request_json.items():
            if isinstance(v, dict):
                request_json[k] = {k2: v2 for k2, v2 in v.items() if v2}

        request = json.dumps(request_json, ensure_ascii=False)

        response = bedrock_runtime_client.invoke_model(
            modelId="amazon.titan-image-generator-v1", body=request
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
    

NODE_CLASS_MAPPINGS = {
    "Bedrock - Titan Text to Image": BedrockTitanTextImage,
    "Bedrock - Titan Inpainting": BedrockTitanInpainting,
    "Bedrock - Titan Outpainting": BedrockTitanOutpainting,
    "Bedrock - Titan Variation": BedrockTitanVariation,
}