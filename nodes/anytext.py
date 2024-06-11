from modelscope.pipelines import pipeline
from util import save_images
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
import cv2

class AnyTextGen:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ori_image": ("IMAGE",),
                "mask_image":("IMAGE",),
                "prompt":("STRING",)
            }
        }

    RETURN_TYPES = ("IMAGE")
    RETURN_NAMES = ("Converted Image")
    FUNCTION = "forward"
    CATEGORY = "aws"
    OUTPUT_NODE = True

    def any_text_edit_image(self,image_input):
        pipe = pipeline('my-anytext-task', model='damo/cv_anytext_text_generation_editing', model_revision='v1.1.1', use_fp16=True, use_translator=False)
        img_save_folder = "SaveImages"
        params = {
            "show_debug": True,
            "image_count": 1,
            "ddim_steps": 20,
        }

        mode = 'text-editing'
        input_data = {
            "prompt": prompt,
            "seed": 8943410,
            "draw_pos": ori_image,
            "ori_image": mask_image
        }
        results, rtn_code, rtn_warning, debug_info = pipe(input_data, mode=mode, **params)
        image = torch.from_numpy(np.array(results[0]).astype(np.float32) / 255.0).unsqueeze(
                    0
                )
        return image

    @retry(tries=MAX_RETRY)
    def forward(self, image):
        return self.any_text_edit_image(image)



NODE_CLASS_MAPPINGS = {
    "AnyText Gen": AnyTextGen,
}