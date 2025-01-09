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

MAX_RETRY = 3

CLAUDE3_MAX_SIZE = 1568

bedrock_runtime_client = get_client(service_name="bedrock-runtime")



class BedrockNovaMultimodal:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True}),
                "model_id": (
                    [
                        "amazon.nova-pro-v1:0",
                        "amazon.nova-lite-v1:0",
                    ],
                ),
                "maxTokens": (
                    "INT",
                    {
                        "default": 2048,
                        "min": 0,  # Minimum value
                        "max": 4096,  # Maximum value
                        "step": 1,  # Slider's step
                        "display": "number",  # Cosmetic only: display as "number" or "slider"
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "round": 0.001,  # The value represeting the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                        "display": "slider",
                    },
                ),
                "topP": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "round": 0.001,  # The value represeting the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                        "display": "slider",
                    },
                )
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "forward"
    CATEGORY = "aws"

    @retry(tries=MAX_RETRY)
    def forward(
        self,
        image,
        prompt,
        model_id,
        max_tokens,
        temperature,
        top_p,
        top_k,
    ):
        """
        Invokes the Anthropic Claude model to run an inference using the input
        provided in the request body.

        :param prompt: The prompt that you want Claude to complete.
        :return: Inference response from the model.
        """

        image = image[0] * 255.0
        image = Image.fromarray(image.clamp(0, 255).numpy().round().astype(np.uint8))

        width, height = image.size
        max_size = max(width, height)
        if max_size > CLAUDE3_MAX_SIZE:
            width = round(width * CLAUDE3_MAX_SIZE / max_size)
            height = round(height * CLAUDE3_MAX_SIZE / max_size)
            image = image.resize((width, height))

        buffer = BytesIO()
        image.save(buffer, format="webp", quality=80)
        image_data = buffer.getvalue()

        image_base64 = base64.b64encode(image_data).decode("utf-8")

        body = json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "maxTokens": max_tokens,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/webp",
                                    "data": image_base64,
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt,
                            },
                        ],
                    }
                ],
                "temperature": temperature,
                "topP": top_p,
            },
            ensure_ascii=False,
        )

        response = bedrock_runtime_client.invoke_model(
            body=body,
            modelId=model_id,
        )
        message = json.loads(response.get("body").read())["content"][0]["text"]

        return (message,)

class BedrockClaudeMultimodal:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True}),
                "model_id": (
                    [
                        "anthropic.claude-3-haiku-20240307-v1:0",
                        "anthropic.claude-3-sonnet-20240229-v1:0",
                        "anthropic.claude-3-opus-20240229-v1:0",
                    ],
                ),
                "max_tokens": (
                    "INT",
                    {
                        "default": 2048,
                        "min": 0,  # Minimum value
                        "max": 4096,  # Maximum value
                        "step": 1,  # Slider's step
                        "display": "number",  # Cosmetic only: display as "number" or "slider"
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "round": 0.001,  # The value represeting the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                        "display": "slider",
                    },
                ),
                "top_p": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "round": 0.001,  # The value represeting the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                        "display": "slider",
                    },
                ),
                "top_k": (
                    "INT",
                    {
                        "default": 250,
                        "min": 0,
                        "max": 500,
                        "step": 1,
                        "round": 1,  # The value represeting the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                        "display": "slider",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "forward"
    CATEGORY = "aws"

    @retry(tries=MAX_RETRY)
    def forward(
        self,
        image,
        prompt,
        model_id,
        max_tokens,
        temperature,
        top_p,
        top_k,
    ):
        """
        Invokes the Anthropic Claude model to run an inference using the input
        provided in the request body.

        :param prompt: The prompt that you want Claude to complete.
        :return: Inference response from the model.
        """

        image = image[0] * 255.0
        image = Image.fromarray(image.clamp(0, 255).numpy().round().astype(np.uint8))

        width, height = image.size
        max_size = max(width, height)
        if max_size > CLAUDE3_MAX_SIZE:
            width = round(width * CLAUDE3_MAX_SIZE / max_size)
            height = round(height * CLAUDE3_MAX_SIZE / max_size)
            image = image.resize((width, height))

        buffer = BytesIO()
        image.save(buffer, format="webp", quality=80)
        image_data = buffer.getvalue()

        image_base64 = base64.b64encode(image_data).decode("utf-8")

        # The different model providers have individual request and response formats.
        # For the format, ranges, and default values for Anthropic Claude, refer to:
        # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html

        body = json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/webp",
                                    "data": image_base64,
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt,
                            },
                        ],
                    }
                ],
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
            },
            ensure_ascii=False,
        )

        response = bedrock_runtime_client.invoke_model(
            body=body,
            modelId=model_id,
        )
        message = json.loads(response.get("body").read())["content"][0]["text"]

        return (message,)


class BedrockClaude:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model_id": (
                    [
                        "anthropic.claude-3-haiku-20240307-v1:0",
                        "anthropic.claude-3-sonnet-20240229-v1:0",
                        "anthropic.claude-3-opus-20240229-v1:0",
                        "anthropic.claude-v2:1",
                        "anthropic.claude-v2",
                        "anthropic.claude-v1",
                        "anthropic.claude-instant-v1",
                    ],
                ),
                "max_tokens": (
                    "INT",
                    {
                        "default": 2048,
                        "min": 0,  # Minimum value
                        "max": 4096,  # Maximum value
                        "step": 1,  # Slider's step
                        "display": "number",  # Cosmetic only: display as "number" or "slider"
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "round": 0.001,  # The value represeting the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                        "display": "slider",
                    },
                ),
                "top_p": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "round": 0.001,  # The value represeting the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                        "display": "slider",
                    },
                ),
                "top_k": (
                    "INT",
                    {
                        "default": 250,
                        "min": 0,
                        "max": 500,
                        "step": 1,
                        "round": 1,  # The value represeting the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                        "display": "slider",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "forward"
    CATEGORY = "aws"

    @retry(tries=MAX_RETRY)
    def forward(
        self,
        prompt,
        model_id,
        max_tokens,
        temperature,
        top_p,
        top_k,
    ):
        """
        Invokes the Anthropic Claude model to run an inference using the input
        provided in the request body.

        :param prompt: The prompt that you want Claude to complete.
        :return: Inference response from the model.
        """

        # The different model providers have individual request and response formats.
        # For the format, ranges, and default values for Anthropic Claude, refer to:
        # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html
        print("prompt input:",prompt)
        body = json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
                            }
                        ],
                    }
                ],
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
            },
            ensure_ascii=False,
        )

        response = bedrock_runtime_client.invoke_model(
            body=body,
            modelId=model_id,
        )
        message = json.loads(response.get("body").read())["content"][0]["text"]
        print("output message:",message)

        return (message,)



class BedrockSDXL:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "resolution": (
                    [
                        "1024 x 1024",
                        "1152 x 896",
                        "1216 x 832",
                        "1344 x 768",
                        "1536 x 640",
                        "640 x 1536",
                        "768 x 1344",
                        "832 x 1216",
                        "896 x 1152",
                    ],
                ),
                "style_preset": (
                    [
                        "None",
                        "3d-model",
                        "analog-film",
                        "anime",
                        "cinematic",
                        "comic-book",
                        "digital-art",
                        "enhance",
                        "fantasy-art",
                        "isometric",
                        "line-art",
                        "low-poly",
                        "modeling-compound",
                        "neon-punk",
                        "origami",
                        "photographic",
                        "pixel-art",
                        "tile-texture",
                    ],
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
                "steps": (
                    "INT",
                    {
                        "default": 30,
                        "min": 10,
                        "max": 50,
                        "step": 1,
                        "round": 1,  # The value represeting the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                        "display": "number",
                    },
                ),
                "clip_guidance_preset": (
                    [
                        "NONE",
                        "FAST_BLUE",
                        "FAST_GREEN",
                        "SIMPLE",
                        "SLOW",
                        "SLOWER",
                        "SLOWEST",
                    ],
                ),
                "sampler": (
                    [
                        "Auto",
                        "DDIM",
                        "DDPM",
                        "K_DPMPP_2M",
                        "K_DPMPP_2S_ANCESTRAL",
                        "K_DPM_2",
                        "K_DPM_2_ANCESTRAL",
                        "K_EULER",
                        "K_EULER_ANCESTRAL",
                        "K_HEUN",
                        "K_LMS",
                    ],
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4294967295,
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
    def forward(
        self,
        prompt,
        resolution,
        style_preset,
        cfg_scale,
        steps,
        clip_guidance_preset,
        sampler,
        seed,
    ):
        height, width = map(int, re.findall(r"\d+", resolution))

        # The different model providers have individual request and response formats.
        # For the format, ranges, and available style_presets of Stable Diffusion models refer to:
        # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-stability-diffusion.html

        request = {
            "text_prompts": [{"text": prompt}],
            "seed": seed,
            "cfg_scale": cfg_scale,
            "steps": steps,
            "height": height,
            "width": width,
            "clip_guidance_preset": clip_guidance_preset,
            "seed": seed,
        }
        if style_preset != "None":
            request["style_prompts"] = style_preset
        if sampler != "Auto":
            request["sampler"] = sampler

        response = bedrock_runtime_client.invoke_model(
            modelId="stability.stable-diffusion-xl-v1",
            body=json.dumps(request, ensure_ascii=False),
        )

        response_body = json.loads(response["body"].read())
        images = [
            np.array(Image.open(BytesIO(base64.b64decode(base64_image["base64"]))))
            for base64_image in response_body.get("artifacts")
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
    "Bedrock - Claude": BedrockClaude,
    "Bedrock - Claude Multimodal": BedrockClaudeMultimodal,
    "Bedrock - SDXL": BedrockSDXL,
}
