import json
import os
import re
import base64
from io import BytesIO

from PIL import Image
import numpy as np
import torch


def image_to_base64(image):
    image = image[0] * 255.0
    image = Image.fromarray(image.clamp(0, 255).numpy().round().astype(np.uint8))

    buffer = BytesIO()
    image.save(buffer, format="PNG")

    image_data = buffer.getvalue()

    image_base64 = base64.b64encode(image_data).decode("utf-8")

    return image_base64