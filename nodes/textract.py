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


textract = get_client(service_name="textract")

MAX_RETRY = 3

class ImageOCRByTextract:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",)
            }
        }

    RETURN_TYPES = ("STRING","IMAGE")
    RETURN_NAMES = ("Position Info","Mask Image")
    FUNCTION = "forward"
    CATEGORY = "aws"
    OUTPUT_NODE = True

    def ocr_by_textract(self,image_input):
        image = image_input[0] * 255.0
        image = Image.fromarray(image.clamp(0, 255).numpy().round().astype(np.uint8))
        # 获取图像原始尺寸
        width, height = image.size

        # 调用Textract DetectDocumentText函数
        img_test = image.read()
        response = textract.detect_document_text(Document={'Bytes': img_test})

        # 初始化结果列表
        result = []
        # 创建一个与原始图像大小相同的黑色遮罩图像
        mask = np.zeros_like(img)


        # 提取文本和边界框信息
        for item in response['Blocks']:
            if item['BlockType'] == 'LINE':
                text = item['Text']
                box = item['Geometry']['BoundingBox']
                left = int(box['Left'] * width)
                top = int(box['Top'] * height)
                width = int(box['Width'] * width))
                height = int(box['Height']* height)

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
                # 对每个文本信息框绘制mask遮罩
                # 指定矩形框的左上角和右下角坐标
                x1, y1 = int(left), int(top)
                x2, y2 = int(left + width), int(top + height)
                # 在遮罩图像上绘制白色矩形框
                mask = cv2.rectangle(mask, (x1, y1), (x2, y2), (255, 255, 255), -1)

        # 将结果列表转换为JSON格式
        json_result = json.dumps(result, indent=2)
        # 将遮罩图像和原始图像进行位运算,生成遮罩后的图像
        masked_img = cv2.bitwise_and(img, mask)
                        image = torch.from_numpy(np.array(masked_img).astype(np.float32) / 255.0).unsqueeze(
                                            0
                                        )

        return json_result,masked_img


    @retry(tries=MAX_RETRY)
    def forward(self, image):
        return self.ocr_by_textract(image)



NODE_CLASS_MAPPINGS = {
    "Image OCR By Textract": ImageOCRByTextract,
}

