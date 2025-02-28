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
import re

MAX_RETRY = 2
DEBUG_MODE = False

# Rely on ComfyUI Paths
comfyui_root = folder_paths.base_path
output_directory = f"{comfyui_root}/output"
bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

def encode_image(image_tensor: torch.Tensor) -> str:
    """Convert ComfyUI image tensor to Base64 string"""
    image_np = image_tensor.cpu().numpy()[0] * 255
    image_np = image_np.astype(np.uint8)

    img = Image.fromarray(image_np)
    buffered = BytesIO()
    img.save(buffered, format="PNG")

    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def parse_colors(color_str: str) -> list:
    """Validate and parse comma-separated hex color codes"""
    if not color_str:
        return []

    colors = [c.strip().upper() for c in color_str.split(",")]
    if len(colors) > 10:
        raise ValueError(f"Too many colors provided: {len(colors)}. Maximum is 10.")

    hex_regex = r"^#?([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$"

    valid_colors = []
    for color in colors:
        # Add missing hash prefix
        if not color.startswith("#"):
            color = f"#{color}"
        # Validate format
        if not re.match(hex_regex, color):
            raise ValueError(f"Invalid color format: {color}")
        valid_colors.append(color)

    return valid_colors


def parse_resolution(resolution_str: str) -> tuple:
    """Extract width/height from resolution string"""
    match = re.findall(r"\d+", resolution_str)
    if len(match) != 2:
        raise ValueError(f"Invalid resolution format: {resolution_str}")
    return int(match[0]), int(match[1])


def generate_images(
    inference_params,
    model_id="amazon.nova-canvas-v1:0",
    output_directory=output_directory,
):

    os.makedirs(output_directory, exist_ok=True)

    image_count = 1
    if "imageGenerationConfig" in inference_params:
        if "numberOfImages" in inference_params["imageGenerationConfig"]:
            image_count = inference_params["imageGenerationConfig"]["numberOfImages"]

    print(f"Generating {image_count} image(s) with {model_id}")

    # Display the seed value if one is being used.
    if "imageGenerationConfig" in inference_params:
        if "seed" in inference_params["imageGenerationConfig"]:
            print(f"Using seed: {inference_params['imageGenerationConfig']['seed']}")

    body_json = json.dumps(inference_params, indent=2)

    # For debugging you might want to set DEBUG_MODE to True
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
            # Write response metadata to JSON file.
            response_metadata_file_path = os.path.join(
                output_directory, "response_metadata.json"
            )
            with open(response_metadata_file_path, "w") as f:
                json.dump(response_metadata, f, indent=2)

            # Write response body to JSON file.
            response_file_path = os.path.join(output_directory, "response_body.json")
            with open(response_file_path, "w") as f:
                json.dump(response_body, f, indent=2)

        # Log the request ID.
        print(f"Request ID: {response['ResponseMetadata']['RequestId']}")

        # Check for non-exception errors.
        if "error" in response_body:
            error_msg = response_body["error"]
            if "blocked by our content filters" in error_msg:
                raise ValueError(f"Content moderation blocked generation: {error_msg}")
            else:
                raise ValueError(f"API Error: {error_msg}")

        # Add this check before returning
        if not response_body.get("images"):
            raise ValueError("No images generated - empty response from API")

        return response_body

    except Exception as e:
        # If e has "response", write it to disk as JSON.
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
                        "round": 1,
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
                "cfg_scale": (
                    "FLOAT",
                    {
                        "default": 7.0,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.1,
                        "round": 0.1,
                        "display": "slider",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 858993459,
                        "step": 1,
                        "round": 1,
                        "display": "number",
                    },
                ),
            },
            "optional": {
                "image": ("IMAGE", {"forceInput": True}),
                "negative_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "placeholder": "Negative prompt eg. low quality, blurry, distorted, text, watermark",
                    },
                ),
                "color_palette": (
                    "STRING",
                    {
                        "multiline": True,
                        "placeholder": "Enter comma-separated hex color codes for providing reference colors (e.g., #FF0000,#00FF00,#0000FF)",
                    },
                ),
                "control_mode": (
                    ["CANNY_EDGE", "SEGMENTATION"],
                    {"default": "CANNY_EDGE"},
                ),
                "control_strength": (
                    "FLOAT",
                    {"default": 0.7, "min": 0.0, "max": 1.0, "display": "slider"},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "forward"
    CATEGORY = "aws"

    @retry(tries=MAX_RETRY)
    def forward(
        self,
        prompt,
        negative_prompt,
        num_images,
        resolution,
        cfg_scale,
        seed,
        color_palette=None,
        image=None,
        control_mode=None,
        control_strength=None,
    ):

        height, width = map(int, re.findall(r"\d+", resolution))

        # Build base parameters
        base_params = {
            "imageGenerationConfig": {
                "numberOfImages": num_images,
                "cfgScale": cfg_scale,
                "quality": "premium",
                "height": height,
                "width": width,
                "seed": seed,
            }
        }

        # Determine task type based on color input
        if color_palette:
            color_list = parse_colors(color_palette)
            if not color_list:
                raise ValueError("At least one valid hex color required in palette")

            color_params = {
                "colors": color_list,
                "text": prompt,
                "negativeText": negative_prompt or " ",
            }
            # Conditionally add referenceImage only when provided
            if image is not None:
                color_params["referenceImage"] = encode_image(image)

            task_params = {
                "taskType": "COLOR_GUIDED_GENERATION",
                "colorGuidedGenerationParams": color_params,
            }
        elif image is not None:
            # Image-conditioned generation
            if not control_mode or control_strength is None:
                raise ValueError(
                    "control_mode and control_strength required when using image input"
                )

            task_params = {
                "taskType": "TEXT_IMAGE",
                "textToImageParams": {
                    "conditionImage": encode_image(image),
                    "controlMode": control_mode,
                    "controlStrength": control_strength,
                    "text": prompt,
                    "negativeText": negative_prompt or " ",
                },
            }
        else:
            # Basic text-to-image
            task_params = {
                "taskType": "TEXT_IMAGE",
                "textToImageParams": {
                    "text": prompt,
                    "negativeText": negative_prompt or " ",
                },
            }

        # Replace base_params.update(task_params) with:
        final_params = base_params.copy()
        final_params.update(task_params)

        print(final_params)

        response_body = generate_images(
            inference_params=final_params,
            model_id="amazon.nova-canvas-v1:0",
        )

        if not response_body.get("images"):
            raise ValueError("Empty image response - check API error logs")

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
                "negative_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "placeholder": "Negative prompt eg. low quality, blurry, distorted, text, watermark",
                    },
                ),
                "similarity_strength": (
                    "FLOAT",
                    {
                        "default": 0.9,
                        "min": 0.2,
                        "max": 1.0,
                        "step": 0.1,
                        "round": 0.01,
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
                        "round": 1,
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
                        "round": 0.1,
                        "display": "slider",
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
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 858993459,
                        "step": 1,
                        "round": 1,
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
        image,
        prompt,
        negative_prompt,
        similarity_strength,
        num_images,
        cfg_scale,
        resolution,
        seed,
    ):

        image = image[0] * 255.0
        image = Image.fromarray(image.clamp(0, 255).numpy().round().astype(np.uint8))

        height, width = map(int, re.findall(r"\d+", resolution))

        buffer = BytesIO()
        image.save(buffer, format="PNG")

        image_data = buffer.getvalue()

        image_base64 = base64.b64encode(image_data).decode("utf-8")

        inference_params = {
            "taskType": "IMAGE_VARIATION",
            "imageVariationParams": {
                "images": [image_base64],
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
                "seed": seed,
            },
        }

        response_body = generate_images(
            inference_params=inference_params,
            model_id="amazon.nova-canvas-v1:0",
            output_directory=output_directory,
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


class BedrockNovaBackgroundPromptReplace:
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
                        "round": 1, 
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
                        "round": 0.1, 
                        "display": "slider",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 858993459,
                        "step": 1,
                        "round": 1,  
                        "display": "number",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "forward"
    CATEGORY = "aws"

    @retry(tries=MAX_RETRY)
    def forward(self, image, prompt, mask_prompt, num_images, cfg_scale, seed):
        image = image[0] * 255.0
        image = Image.fromarray(image.clamp(0, 255).numpy().round().astype(np.uint8))

        buffer = BytesIO()
        image.save(buffer, format="PNG")

        image_data = buffer.getvalue()

        image_base64 = base64.b64encode(image_data).decode("utf-8")

        inference_params = {
            "taskType": "OUTPAINTING",
            "outPaintingParams": {
                "image": image_base64,
                "text": prompt,
                "maskPrompt": mask_prompt,
                "outPaintingMode": "PRECISE",
            },
            "imageGenerationConfig": {
                "numberOfImages": num_images,
                "quality": "premium",
                "cfgScale": cfg_scale,
                "seed": seed,
            },
        }

        response_body = generate_images(
            inference_params=inference_params, model_id="amazon.nova-canvas-v1:0"
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
    "Amazon Bedrock - Nova Canvas Generate Image": BedrockNovaTextImage,
    "Amazon Bedrock - Nova Canvas Generate Variations": BedrockNovaIpAdatper,
    "Amazon Bedrock - Nova Canvas Background Prompt Replace": BedrockNovaBackgroundPromptReplace,
}