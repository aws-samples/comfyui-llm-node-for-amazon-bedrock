import json
import logging
import os
from datetime import datetime

import boto3

logger = logging.getLogger(__name__)


def generate_images(
    inference_params,
    model_id="amazon.nova-canvas-v1:0",
    region_name="us-east-1",
    endpoint_url=None,
    output_directory="~/ComfyUI/output",
):
    """
    Generate an image using using Titan Image Generator or Olympus Image Generator.
    Args:
        body (dict) : The request body to use.
        model_id (str): The model ID to use. Defaults to "amazon.titan-image-generator-v1".
        region_name (str): The AWS region name to use. Defaults to "us-east-1".
        endpoint_url (str): The endpoint URL to use. Defaults to None.
        output_directory (str): The directory to save the generated images. Defaults to "./output".
    Returns:
        response_body (dict): The response body from the model.
        job_output_directory (str): The directory where the request and response JSON values are saved.
        generation_id (str): The generation ID for the request.
    Usage:
        response_body, job_output_directory, generation_id = generate_images(inference_params)
    Raises:
        Exception: Any exception thrown by .invoke_model().
    """
    os.makedirs(output_directory, exist_ok=True)

    image_count = 1
    if "imageGenerationConfig" in inference_params:
        if "numberOfImages" in inference_params["imageGenerationConfig"]:
            image_count = inference_params["imageGenerationConfig"]["numberOfImages"]

    logger.info(f"Generating {image_count} image(s) with {model_id}")

    # Display the seed value if one is being used.
    if "imageGenerationConfig" in inference_params:
        if "seed" in inference_params["imageGenerationConfig"]:
            logger.info(
                f"Using seed: {inference_params['imageGenerationConfig']['seed']}"
            )

    bedrock_client_optional_args = (
        {} if endpoint_url is None else {"endpoint_url": endpoint_url}
    )
    bedrock = boto3.client(
        service_name="bedrock-runtime",
        region_name=region_name,
        **bedrock_client_optional_args,
    )

    body_json = json.dumps(inference_params, indent=2)

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
        logger.info(f"Request ID: {response['ResponseMetadata']['RequestId']}")

        # Check for non-exception errors.
        if "error" in response_body:
            if response_body["error"] == "":
                logger.warning(
                    """Response included 'error' of "" (empty string). This indicates a bug with the Bedrock API."""
                )
            else:
                logger.warning(f"Error: {response_body['error']}")

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

        logger.error(e)
        raise e
