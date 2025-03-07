import json
import os
import re
import base64
from io import BytesIO
from retry import retry
from PIL import Image
import numpy as np
import torch
import boto3
import folder_paths

MAX_RETRY = 2
DEBUG_MODE = False

# Rely on ComfyUI Paths
comfyui_root = folder_paths.base_path
output_directory = f"{comfyui_root}/output"
bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-west-2")

# Model IDs
MODEL_ID_SD3_LARGE = "stability.sd3-large-v1:0"
MODEL_ID_SD3_5_LARGE = "stability.sd3-5-large-v1:0"
MODEL_ID_STABLE_IMAGE_CORE = "stability.stable-image-core-v1:1"
MODEL_ID_STABLE_IMAGE_ULTRA = "stability.stable-image-ultra-v1:1"


def encode_image(image_tensor: torch.Tensor) -> str:
    """Convert ComfyUI image tensor to Base64 string"""
    image_np = image_tensor.cpu().numpy()[0] * 255
    image_np = image_np.astype(np.uint8)

    img = Image.fromarray(image_np)
    buffered = BytesIO()
    img.save(buffered, format="PNG")

    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def parse_resolution(resolution_str: str) -> tuple:
    """Extract width/height from resolution string"""
    match = re.findall(r"\d+", resolution_str)
    if len(match) != 2:
        raise ValueError(f"Invalid resolution format: {resolution_str}")
    return int(match[0]), int(match[1])


def resolution_to_aspect_ratio(resolution_str: str) -> str:
    """Convert resolution string to aspect ratio string for Stability AI models"""
    # Direct mapping for common resolutions
    aspect_ratio_map = {
        "1024 x 1024": "1:1",
        "1088 x 896": "5:4",
        "1152 x 896": "5:4",
        "1216 x 832": "16:9",
        "1344 x 768": "16:9",
        "1536 x 640": "21:9",
        "640 x 1536": "9:21",
        "768 x 1344": "9:16",
        "832 x 1216": "9:16",
        "896 x 1088": "4:5",
        "896 x 1152": "4:5",
        "512 x 512": "1:1",
        "512 x 640": "4:5",
        "640 x 512": "5:4",
        "768 x 768": "1:1",
        "1536 x 1536": "1:1",
        "1792 x 1344": "4:3",
        "1920 x 1280": "3:2",
        "2048 x 1024": "2:1",
        "1024 x 2048": "1:2",
        "1280 x 1920": "2:3",
        "1344 x 1792": "3:4",
    }

    # If resolution is in the map, return the corresponding aspect ratio
    if resolution_str in aspect_ratio_map:
        return aspect_ratio_map[resolution_str]

    # Otherwise, calculate it
    width, height = parse_resolution(resolution_str)

    # Find greatest common divisor
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    divisor = gcd(width, height)
    return f"{width // divisor}:{height // divisor}"


@retry(tries=MAX_RETRY)
def generate_images(
    inference_params,
    model_id,
    output_directory=output_directory,
):
    """Generate images using AWS Bedrock API"""
    os.makedirs(output_directory, exist_ok=True)

    # Display seed if available
    if "seed" in inference_params:
        print(f"Using seed: {inference_params['seed']}")

    body_json = json.dumps(inference_params, indent=2)

    # For debugging
    if DEBUG_MODE:
        request_file_path = os.path.join(output_directory, "request.json")
        with open(request_file_path, "w") as f:
            f.write(body_json)

    try:
        response = bedrock.invoke_model(
            body=body_json,
            modelId=model_id,
            accept="application/json",
            contentType="application/json",
        )

        response_body = json.loads(response.get("body").read())

        if DEBUG_MODE:
            response_metadata = response.get("ResponseMetadata")
            # Write response metadata to JSON file
            response_metadata_file_path = os.path.join(
                output_directory, "response_metadata.json"
            )
            with open(response_metadata_file_path, "w") as f:
                json.dump(response_metadata, f, indent=2)

            # Write response body to JSON file
            response_file_path = os.path.join(output_directory, "response_body.json")
            with open(response_file_path, "w") as f:
                json.dump(response_body, f, indent=2)

        # Log the request ID
        print(f"Request ID: {response['ResponseMetadata']['RequestId']}")

        # Check for non-exception errors
        if "error" in response_body:
            error_msg = response_body["error"]
            if "blocked by our content filters" in error_msg:
                raise ValueError(f"Content moderation blocked generation: {error_msg}")
            else:
                raise ValueError(f"API Error: {error_msg}")

        # Check for artifacts key in SD3/SD3.5 responses
        if "images" in response_body:
            return response_body
        else:
            raise ValueError("No images generated - empty response from API")

    except Exception as e:
        # If e has "response", write it to disk as JSON
        if hasattr(e, "response"):
            if DEBUG_MODE:
                error_response = e.response
                error_response_file_path = os.path.join(
                    output_directory, "error_response.json"
                )
                with open(error_response_file_path, "w") as f:
                    json.dump(error_response, f, indent=2)
            print(e)
        raise e


class BedrockStabilityText2Image:
    """ComfyUI node for Stability AI models text-to-image"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    [
                        "SD3_Large",
                        "SD3.5_Large",
                        "Stable_Image_Core",
                        "Stable_Image_Ultra",
                    ],
                    {"default": "Stable_Image_Core"},
                ),
                "prompt": ("STRING", {"multiline": True}),
                "resolution": (
                    [
                        "1024 x 1024",
                        "1088 x 896",
                        "1216 x 832",
                        "1344 x 768",
                        "1536 x 640",
                        "640 x 1536",
                        "768 x 1344",
                        "832 x 1216",
                        "896 x 1088",
                    ],
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4294967295,
                        "step": 1,
                        "round": 1,
                        "display": "number",
                    },
                ),
            },
            "optional": {
                "negative_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "placeholder": "Negative prompt e.g., low quality, blurry, distorted",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "forward"
    CATEGORY = "aws"

    def forward(
        self,
        model,
        prompt,
        resolution,
        seed,
        negative_prompt=None,
    ):

        aspect_ratio = resolution_to_aspect_ratio(resolution)

        if model == "SD3_Large":
            mode_id_input = MODEL_ID_SD3_LARGE
        elif model == "SD3.5_Large":
            mode_id_input = MODEL_ID_SD3_5_LARGE
        elif model == "Stable_Image_Core":
            mode_id_input = MODEL_ID_STABLE_IMAGE_CORE
        elif model == "Stable_Image_Ultra":
            mode_id_input = MODEL_ID_STABLE_IMAGE_ULTRA

        # Build parameters for Stability AI Models
        params = {
            "mode": "text-to-image",
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "output_format": "png",
            "seed": seed,
        }

        # Add negative prompt if provided
        if negative_prompt:
            params["negative_prompt"] = negative_prompt

        response_body = generate_images(
            inference_params=params,
            model_id=mode_id_input,
        )

        # Process response
        if "images" in response_body:
            base64_output_image = response_body["images"][0]
            image_data = base64.b64decode(base64_output_image)
            image = Image.open(BytesIO(image_data))
        else:
            raise ValueError("No successful images generated")

        # Convert to ComfyUI format
        result = torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(
            0
        )

        return (result,)


class BedrockSD3xImage2Image:
    """ComfyUI node for Stability Diffusion 3 & 3.5 Large image-to-image"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"forceInput": True}),
                "model": (
                    [
                        "SD3_Large",
                        "SD3.5_Large",
                    ],
                    {"default": "SD3.5_Large"},
                ),
                "prompt": ("STRING", {"multiline": True}),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4294967295,
                        "step": 1,
                        "round": 1,
                        "display": "number",
                    },
                ),
            },
            "optional": {
                "strength": (
                    "FLOAT",
                    {
                        "default": 0.6,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "round": 0.01,
                        "display": "slider",
                    },
                ),
                "negative_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "placeholder": "Negative prompt e.g., low quality, blurry, distorted",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "forward"
    CATEGORY = "aws"

    def forward(
        self,
        image,
        model,
        prompt,
        seed,
        strength,
        negative_prompt=None,
    ):

        if model == "SD3_Large":
            mode_id_input = MODEL_ID_SD3_LARGE
        elif model == "SD3.5_Large":
            mode_id_input = MODEL_ID_SD3_5_LARGE

        # Build parameters for Stability AI Models
        params = {
            "mode": "image-to-image",
            "prompt": prompt,
            "image": encode_image(image),
            "strength": strength,
            "output_format": "png",
            "seed": seed
        }

        # Add negative prompt if provided
        if negative_prompt:
            params["negative_prompt"] = negative_prompt

        response_body = generate_images(
            inference_params=params,
            model_id=mode_id_input,
        )

        # Process response
        if "images" in response_body:
            base64_output_image = response_body["images"][0]
            image_data = base64.b64decode(base64_output_image)
            image = Image.open(BytesIO(image_data))
        else:
            raise ValueError("No successful images generated")

        # Convert to ComfyUI format
        result = torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(
            0
        )

        return (result,)


# Register all node classes for ComfyUI
NODE_CLASS_MAPPINGS = {
    "Amazon Bedrock - Stability AI Models | Text to Image": BedrockStabilityText2Image,
    "Amazon Bedrock - SD3 & SD3.5 Large | Image to Image": BedrockSD3xImage2Image,
}
