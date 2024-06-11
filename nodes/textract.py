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

class GetDominatColor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",)
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("color")
    FUNCTION = "forward"
    CATEGORY = "aws"
    OUTPUT_NODE = True

    def get_dominant_color(self, image, box):
        """
        获取图像中指定边界框区域内的主要颜色,并返回 #RRGGBB 格式的颜色字符串
        参数:
            image: 输入图像
            box: 边界框坐标,格式为(left, top, width, height)
        返回值:
            主要颜色的 #RRGGBB 格式字符串
        """

        # 从边界框坐标构造ROI区域
        position = json.loads(box)['BoundingBox']
        left = position['left']
        top = position['top']
        width = position['width']
        height = position['height']

        roi = image[top:top+height, left:left+width]

        # 计算ROI区域的像素值直方图
        pixels = np.float32(roi.reshape(-1, 3))
        n_colors = 8
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
        _, _, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

        # 获取最多像素对应的颜色
        counts = np.histogram(pixels, bins=np.arange(n_colors+1))[0]
        dominant_index = np.argmax(counts)
        dominant_color = palette[dominant_index].astype(np.uint8)

        # 转换为 #RRGGBB 格式的字符串
        hex_color = '#%02x%02x%02x' % (int(dominant_color[0]), int(dominant_color[1]), int(dominant_color[2]))

        return hex_color


    @retry(tries=MAX_RETRY)
    def forward(self, image,box):
        return self.ocr_by_textract(image)

class ImageOCRByTextract:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",)
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("OCR result")
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

        # 将结果列表转换为JSON格式
        json_result = json.dumps(result, indent=2)
        return json_result

    @retry(tries=MAX_RETRY)
    def forward(self, image):
        return self.ocr_by_textract(image)





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
        position=json.loads(position_info)['BoundingBox']
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
    "Image OCR By Textract": ImageOCRByTextract,
    "Get Dominat Color": GetDominatColor,
    "get mask by textract clip": ImageToMaskByClip
}

