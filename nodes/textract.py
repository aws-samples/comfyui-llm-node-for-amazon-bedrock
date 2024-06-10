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


from .session import get_client


s3_client = get_client(service_name="textract")

MAX_RETRY = 3




class ImageOCRByTextract:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",)
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "forward"
    CATEGORY = "aws"
    OUTPUT_NODE = True

    def ocr_by_textract(self,image_file):
    # 获取图像原始尺寸
        with Image.open(image_file) as img:
            width, height = img.size

        # 调用Textract DetectDocumentText函数
        with open(image_file, 'rb') as file:
            img_test = file.read()
            response = textract.detect_document_text(Document={'Bytes': img_test})

        # 初始化结果列表
        result = []

        # 提取文本和边界框信息
        for item in response['Blocks']:
            if item['BlockType'] == 'LINE':
                text = item['Text']
                box = item['Geometry']['BoundingBox']
                left = int(box['Left'] * width)
                top = int(box['Top'] * height)
                width = int(box['Width'] )
                height = int(box['Height'])

                # 将信息添加到结果列表中
                result.append({
                    'Text': text,
                    'BoundingBox': {
                        'Left': left,
                        'Top': top,
                        'Width': width,
                        'Height': height
                    }
                })

        # 将结果列表转换为JSON格式
        json_result = json.dumps(result, indent=2)
        return json_result

    @retry(tries=MAX_RETRY)
    def forward(self, image):
        return self.ocr_by_textract(image)


NODE_CLASS_MAPPINGS = {
    "Image OCR By Textract": ImageOCRByTextract
}


class ImageToMaskByClip:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "position_info": ("STRING", ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "forward"
    CATEGORY = "aws"
    OUTPUT_NODE = True

    def genMaskByClip(self,image_file,position_info):
        position=json.loads(position_info)
        left = position['Left']
        top = position['Top']
        width = position['Width']
        height = position['Height']
        # 读取原始图像
        img = cv2.imread('image.jpg')

        # 指定矩形框的左上角和右下角坐标
        x1, y1 = int(left), int(top)
        x2, y2 = int(left + width), int(top + height)

        # 创建一个与原始图像大小相同的黑色遮罩图像
        mask = np.zeros_like(img)

        # 在遮罩图像上绘制白色矩形框
        mask = cv2.rectangle(mask, (x1, y1), (x2, y2), (255, 255, 255), -1)

        # 将遮罩图像和原始图像进行位运算,生成遮罩后的图像
        masked_img = cv2.bitwise_and(img, mask)
        image = torch.from_numpy(np.array(masked_img).astype(np.float32) / 255.0).unsqueeze(
                    0
                )

    @retry(tries=MAX_RETRY)
    def forward(self, image, position_info):
        return self.genMaskByClip(image,position_info)


    NODE_CLASS_MAPPINGS = {
        "get mask by textract clip": ImageToMaskByClip
    }


