import json
import os
import re
import base64
from io import BytesIO
import io
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

    RETURN_TYPES = ("STRING","INT","INT","INT","INT","IMAGE")
    RETURN_NAMES = ("Text","Left","Top","Width","Height","Mask Image")
    FUNCTION = "forward"
    CATEGORY = "aws"
    OUTPUT_NODE = True

    def ocr_by_textract(self,image_input):
        numpy_image = (image_input[0] * 255.0).clamp(0, 255).numpy()
        image = image_input[0] * 255.0
        image = Image.fromarray(image.clamp(0, 255).numpy().round().astype(np.uint8))
        # 获取图像原始尺寸
        img_width, img_height = image.size

        # 调用Textract DetectDocumentText函数
        byte_stream = io.BytesIO()
        image.save(byte_stream, format='PNG')
        byte_image = byte_stream.getvalue()
        response = textract.detect_document_text(Document={'Bytes': byte_image})

        # 初始化结果列表
        result = []
        # 创建一个与原始图像大小相同的黑色遮罩图像
        mask = np.zeros_like(numpy_image)
        all_text=""


        # 提取文本和边界框信息
        for item in response['Blocks']:
            if item['BlockType'] == 'LINE':
                text = item['Text']
                box = item['Geometry']['BoundingBox']
                left = int(box['Left'] * img_width)
                top = int(box['Top'] * img_height)
                width = int(box['Width'] * img_width)
                height = int(box['Height']* img_height)

                # 将信息添加到结果列表中
                result.append({
                    'Text': text,
                    'Left': left,
                    'Top': top,
                    'Width': width,
                    'Height': height

                })
                all_text=all_text+text
                # 对每个文本信息框绘制mask遮罩
                # 指定矩形框的左上角和右下角坐标
                temp_mask = np.zeros_like(numpy_image)
                x1, y1 = int(left), int(top)
                x2, y2 = int(left + width), int(top + height)
                # 在遮罩图像上绘制白色矩形框
                cv2.rectangle(temp_mask, (x1, y1), (x2, y2), (255, 255, 255), -1)
                mask = cv2.bitwise_or(mask, temp_mask)

        # 将遮罩图像和原始图像进行位运算,生成遮罩后的图像
        masked_img = cv2.bitwise_and(numpy_image, mask)
        masked_img = torch.from_numpy(np.array(masked_img).astype(np.float32) / 255.0).unsqueeze(0)
        print("result",result)

        return all_text,result[0]['Left'],result[0]['Top'],result[0]['Width'],result[0]['Height'],masked_img


    @retry(tries=MAX_RETRY)
    def forward(self, image):
        return self.ocr_by_textract(image)



NODE_CLASS_MAPPINGS = {
    "Image OCR By Textract": ImageOCRByTextract
}

