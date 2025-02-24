import json
import os
import base64
from io import BytesIO
from random import randint
import logging
from datetime import datetime
from retry import retry
from PIL import Image
import numpy as np
import torch
from .session import get_client
import boto3
from .utils import image_to_base64

MAX_RETRY = 3

def generate_images(
    inference_params,
    model_id="stability.sd3-5-large-v1:0",
    region_name="us-east-1",
    endpoint_url=None,
    output_directory="~/ComfyUI/output",
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
            print(
                f"Using seed: {inference_params['imageGenerationConfig']['seed']}"
            )

    bedrock_client_optional_args = (
        {} if endpoint_url is None else {"endpoint_url": endpoint_url}
    )
    bedrock = boto3.client(
        service_name="bedrock-runtime",
        #region_name=region_name,
        **bedrock_client_optional_args,
    )

    body_json = json.dumps(inference_params, indent=2)
    #print("here1===",body_json)

    # Write request body to JSON file.
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

        response_metadata = response.get("ResponseMetadata")
        # WRite response metadata to JSON file.
        response_metadata_file_path = os.path.join(
            output_directory, "response_metadata.json"
        )
        with open(response_metadata_file_path, "w") as f:
            json.dump(response_metadata, f, indent=2)

        response_body = json.loads(response.get("body").read())

        # Write response body to JSON file.
        response_file_path = os.path.join(output_directory, "response_body.json")
        with open(response_file_path, "w") as f:
            json.dump(response_body, f, indent=2)

        # Log the request ID.
        print(f"Request ID: {response['ResponseMetadata']['RequestId']}")

        # Check for non-exception errors.
        if "error" in response_body:
            if response_body["error"] == "":
                print(
                    """Response included 'error' of "" (empty string). This indicates a bug with the Bedrock API."""
                )
            else:
                printf("Error: {response_body['error']}")

        return response_body

    except Exception as e:
        # If e has "response", write it to disk as JSON.
        if hasattr(e, "response"):
            error_response = e.response
            error_response_file_path = os.path.join(
                output_directory, "error_response.json"
            )
            with open(error_response_file_path, "w") as f:
                json.dump(error_response, f, indent=2)

        print(e)
        raise e


bedrock_runtime_client = get_client(service_name="bedrock-runtime")
output_directory = "/home/ubuntu/ComfyUI/output/"
class BedrockStability3ImageImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "image": ("IMAGE",),
                "strength": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0,
                        "max": 1,
                        "step": 0.1,
                        "round": 0.001,
                        "display": "slider",
                    },
                ),

            },
            "optional": {
                "negative_prompt":("STRING", {"multiline": True}),
                "output_format": (
                    [
                        "png",
                        "jpeg",
                        "webp"
                    ],
                ),
                "seed": (
                    "INT",
                    {
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
    def forward(self, prompt, negative_prompt, image, strength, output_format, seed):
        image_base64 = image_to_base64(image)

        inference_params = {
            "image": image_base64,
            "prompt": prompt,
            "strength": strength,
            "negative_prompt": negative_prompt,
            "mode": "image-to-image",
            "output_format": output_format,
            "seed": seed,
        }

        # Define an output directory with a unique name.
        generation_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_directory = f"output/{generation_id}"

        # Generate the image(s).
        response_body = generate_images(
            inference_params=inference_params,
            model_id="stability.sd3-5-large-v1:0",
            output_directory=output_directory
        )

        # response_body = json.loads(response["body"].read())
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

class BedrockStability3TextImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
            },
            "optional": {
                "negative_prompt":("STRING", {"multiline": True}),
                "aspect_ratio": (
                    [
                        "16:9",
                        "1:1",
                        "21:9",
                        "2:3",
                        "3:2",
                        "4:5",
                        "5:4",
                        "9:16",
                        "9:21"
                    ],
                    {
                        "default": "1:1"
                    }
                ),
                "output_format": (
                    [
                        "png",
                        "jpeg",
                        "webp"
                    ],
                    {
                        "default": "png"
                    }
                ),
                "seed": (
                    "INT",
                    {
                        "min": 0,
                        "max": 2147483646,
                        "step": 1,
                        "round": 1,  # The value represeting the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                        "display": "number",
                    },
                )
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "forward"
    CATEGORY = "aws"

    @retry(tries=MAX_RETRY)
    def forward(self, prompt, negative_prompt, aspect_ratio, output_format, seed):
        inference_params = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "aspect_ratio": aspect_ratio,
            "mode": "text-to-image",
            "output_format": output_format,
            "seed": seed,
        }

        # Define an output directory with a unique name.
        generation_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_directory = f"output/{generation_id}"

        # Generate the image(s).
        response_body = generate_images(
            inference_params=inference_params,
            model_id="stability.sd3-5-large-v1:0",
            output_directory=output_directory
        )

        # response_body = json.loads(response["body"].read())
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
    "Bedrock - Stable Diffusion 3.5 Large Image-to-Image": BedrockStability3ImageImage,
    "Bedrock - Stable Diffusion 3.5 Large Text-to-Image": BedrockStability3TextImage
}