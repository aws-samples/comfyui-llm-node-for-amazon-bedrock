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
import folder_paths
from .session import get_client
import torch
from torchvision import transforms
from paddleocr import PaddleOCR, draw_ocr
from paddleocr import PPStructure,draw_structure_result
import tempfile


current_directory = os.path.dirname(os.path.abspath(__file__))
temp_img_path = os.path.join(current_directory, "temp_dir", "AnyText_manual_mask_pos_img.png")
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

        ## Call Textract DetectDocumentText function
        byte_stream = io.BytesIO()
        image.save(byte_stream, format='PNG')
        byte_image = byte_stream.getvalue()
        response = textract.detect_document_text(Document={'Bytes': byte_image})

        #initial the scan result
        result = []
        # Create a black mask image of the same size as the original image
        mask = np.zeros_like(numpy_image)
        all_text=""


        # Extract text and bounding box information
        for item in response['Blocks']:
            if item['BlockType'] == 'LINE':
                text = item['Text']
                box = item['Geometry']['BoundingBox']
                left = int(box['Left'] * img_width)
                top = int(box['Top'] * img_height)
                width = int(box['Width'] * img_width)
                height = int(box['Height']* img_height)

                ##Add information to the result list
                result.append({
                    'Text': text,
                    'Left': left,
                    'Top': top,
                    'Width': width,
                    'Height': height

                })
                all_text=all_text+text
                #Draw mask for each text information box
                #Specify the coordinates of the top-left and bottom-right corners of the rectangle
                temp_mask = np.zeros_like(numpy_image)
                x1, y1 = int(left), int(top)
                x2, y2 = int(left + width), int(top + height)
                # Draw white rectangular boxes on the mask image
                cv2.rectangle(temp_mask, (x1, y1), (x2, y2), (255, 255, 255), -1)
                mask = cv2.bitwise_or(mask, temp_mask)

        # Perform bitwise operation between the mask image and the original image to generate the masked image
        masked_img = cv2.bitwise_and(numpy_image, mask)
        masked_img = torch.from_numpy(np.array(masked_img).astype(np.float32) / 255.0).unsqueeze(0)

        print("result",result)

        return all_text,result[0]['Left'],result[0]['Top'],result[0]['Width'],result[0]['Height'],masked_img


    @retry(tries=MAX_RETRY)
    def forward(self, image):
        return self.ocr_by_textract(image)


### for anyText Nodes
class ImageOCRByTextractV2:

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                  {
                      "image": (sorted(files), {"image_upload": True}),
                    },
                }

    RETURN_TYPES = ("STRING","STRING","STRING","IMAGE")
    RETURN_NAMES = ("Texts","Original Image Path","Mask Image Path","AnyText Mask Image")
    FUNCTION = "forward"
    CATEGORY = "aws"
    OUTPUT_NODE = True

    def ocr_by_textract(self,image_input):
        ori_image_path = folder_paths.get_annotated_filepath(image_input)
        pos_img_path = os.path.join(temp_img_path)

        ## 转换ori_image为tensor张量
        pil_image = Image.open(ori_image_path)
        transform = transforms.Compose([
            transforms.ToTensor(),  # Convert PIL Image to tensor and normalize pixel values to [0, 1]
        ])
        image_input = transform(pil_image)

        ## The image input is already in the standard ComfyUI image tensor format
        numpy_image = (image_input[0] * 255.0).clamp(0, 255).numpy()
        image = image_input[0] * 255.0
        image = Image.fromarray(image.clamp(0, 255).numpy().round().astype(np.uint8))
        # Get the original dimensions of the image
        img_width, img_height = image.size

        # Call the Textract DetectDocumentText function
        byte_stream = io.BytesIO()
        image.save(byte_stream, format='PNG')
        byte_image = byte_stream.getvalue()
        response = textract.detect_document_text(Document={'Bytes': byte_image})

        # Initialize the results list
        result = []
        # Create a black mask image with the same size as the original image
        #mask = np.zeros_like(numpy_image)
        mask = np.ones_like(numpy_image) * 255
        all_text=""


        # Extract text and bounding box information
        for item in response['Blocks']:
            if item['BlockType'] == 'LINE':
                text = item['Text']
                box = item['Geometry']['BoundingBox']
                left = int(box['Left'] * img_width)
                top = int(box['Top'] * img_height)
                width = int(box['Width'] * img_width)
                height = int(box['Height']* img_height)

                # Add the information to the result list
                result.append(text)
                # Draw mask for each text information box
                # Specify the coordinates of the top-left and bottom-right corners of the rectangle
                temp_mask = np.zeros_like(numpy_image)
                x1, y1 = int(left), int(top)
                x2, y2 = int(left + width), int(top + height)
                # Draw black rectangular boxes on the mask image
                #cv2.rectangle(temp_mask, (x1, y1), (x2, y2), (0, 0, 0), -1)
                #mask = cv2.bitwise_or(mask, temp_mask)

                cv2.rectangle(mask, (left, top), (left + width, top + height), (0, 0, 0), -1)

        # Perform bitwise operation between the mask image and the original image to generate the masked image
        masked_img = np.where(mask == 0, 0, numpy_image)
        #masked_img = cv2.bitwise_and(numpy_image, mask)
        masked_img = torch.from_numpy(np.array(masked_img).astype(np.float32) / 255.0).unsqueeze(0)
        #masked_img.save(temp_img_path)
        # Convert PyTorch tensor to PIL image
        to_pil = transforms.ToPILImage()
        pil_image = to_pil(masked_img.squeeze(0))
        # save PIL image
        pil_image.save(temp_img_path)

        all_text="|".join(result)
        print("result",result)

        return all_text ,ori_image_path,temp_img_path, masked_img



    @retry(tries=MAX_RETRY)
    def forward(self, image):
        return self.ocr_by_textract(image)


## for layer style nodes
class ImageOCRByTextractV4:

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":{
                  "image": ("IMAGE",),
                  }
                }

    RETURN_TYPES = ("STRING","STRING","STRING","STRING","STRING","STRING","STRING","IMAGE","IMAGE")
    RETURN_NAMES = ("Texts","x_offsets","y_offsets","widths","heights","img_width","img_height","Mask Image","Original Image")
    FUNCTION = "forward"
    CATEGORY = "aws"
    OUTPUT_NODE = True

    def convert_to_xywh(self,coordinates):
        # Extract all x and y coordinates
        x_coords = [coord[0] for coord in coordinates]
        y_coords = [coord[1] for coord in coordinates]

        # Calculate x offset and y offset
        x_offset = min(x_coords)
        y_offset = min(y_coords)

        # Calculate width and height
        width = max(x_coords) - x_offset
        height = max(y_coords) - y_offset

        print(x_offset,y_offset,width,height)
        return int(x_offset), int(y_offset), int(width), int(height)

    def ocr_by_paddleocr(self,image_input):

        image = image_input[0] * 255.0
        image = Image.fromarray(image.clamp(0, 255).numpy().round().astype(np.uint8))
        numpy_image = (image_input[0] * 255.0).clamp(0, 255).numpy()

        img_width, img_height = image.size


        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png', dir='/tmp/') as temp_file:
            temp_filename = temp_file.name

        # Save numpy_image as a temporary file
        cv2.imwrite(temp_filename, cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR))

        ocr = PaddleOCR(
        det_model_dir='/home/ubuntu/ComfyUI/models/checkpoints/PaddleOCR/det',
        rec_model_dir='/home/ubuntu/ComfyUI/models/checkpoints/PaddleOCR/rec',
        det_limit_side_len=2048,
        use_angle_cls=True,
        #use_gpu=False,
        )

        ocr_results = ocr.ocr(temp_filename, cls=True)[0]

        # Create a mask image with the same size as the original image
        result = []
        masked_img = numpy_image.copy()
        all_text=""
        x_offsets=[]
        y_offsets=[]
        widths=[]
        heights=[]


        # Extract text and bounding box information
        for line in ocr_results:
             if not isinstance(line, list):
                continue
             boxes = line[0]
             x_offset,y_offset,width,height = self.convert_to_xywh(boxes)
             x_offsets.append(str(x_offset))
             y_offsets.append(str(y_offset))
             widths.append(str(width))
             heights.append(str(height))

             text = line[1][0]
             print("text")
             print(text)
             result.append(text)

             # Draw mask for each text information box
             # Specify the coordinates of the top-left and bottom-right corners of the rectangle
             x1, y1 = int(x_offset), int(y_offset)
             x2, y2 = int(x_offset + width), int(y_offset + height)
             # Draw a black rectangle on the mask image
             cv2.rectangle(masked_img, (x1, y1), (x2, y2), (0, 0, 0), -1)

        masked_img = torch.from_numpy(np.array(masked_img).astype(np.float32) / 255.0).unsqueeze(0)

        all_text="|".join(result)
        x_offsets="|".join(x_offsets)
        y_offsets="|".join(y_offsets)
        widths="|".join(widths)
        heights="|".join(heights)

        print("result",result)

        # Add original image output
        original_img = image_input
        # delete temp files
        os.unlink(temp_filename)

        return all_text ,x_offsets,y_offsets,widths,heights,img_width,img_height, masked_img,original_img



    @retry(tries=MAX_RETRY)
    def forward(self, image):
        return self.ocr_by_paddleocr(image)


## for layer style nodes
class ImageOCRByTextractV3:

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":{
                  "image": ("IMAGE",),
                  }
                }

    RETURN_TYPES = ("STRING","STRING","STRING","STRING","STRING","STRING","STRING","IMAGE","IMAGE")
    RETURN_NAMES = ("Texts","x_offsets","y_offsets","widths","heights","img_width","img_height","Mask Image","Original Image")
    FUNCTION = "forward"
    CATEGORY = "aws"
    OUTPUT_NODE = True

    def ocr_by_textract(self,image_input):


        image = image_input[0] * 255.0
        image = Image.fromarray(image.clamp(0, 255).numpy().round().astype(np.uint8))
        numpy_image = (image_input[0] * 255.0).clamp(0, 255).numpy()


        img_width, img_height = image.size


        byte_stream = io.BytesIO()
        image.save(byte_stream, format='PNG')
        byte_image = byte_stream.getvalue()
        response = textract.detect_document_text(Document={'Bytes': byte_image})


        result = []

        masked_img = numpy_image.copy()
        all_text=""
        x_offsets=[]
        y_offsets=[]
        widths=[]
        heights=[]



        for item in response['Blocks']:
            if item['BlockType'] == 'LINE':
                text = item['Text']
                box = item['Geometry']['BoundingBox']

                left = int(box['Left'] * img_width)
                x_offsets.append(str(left))

                top = int(box['Top'] * img_height)
                y_offsets.append(str(top))

                width = int(box['Width'] * img_width)
                widths.append(str(width))

                height = int(box['Height']* img_height)
                heights.append(str(height))


                result.append(text)
                # Draw mask for each text information box
                # Specify the coordinates of the top-left and bottom-right corners of the rectangle
                x1, y1 = int(left), int(top)
                x2, y2 = int(left + width), int(top + height)
                # Draw black rectangular boxes on the mask image
                cv2.rectangle(masked_img, (x1, y1), (x2, y2), (0, 0, 0), -1)


        #masked_img = torch.from_numpy(np.array(masked_img).astype(np.float32) / 255.0).unsqueeze(0)
        masked_img = torch.from_numpy(np.array(masked_img).astype(np.float32) / 255.0).unsqueeze(0)
        ###summary the output
        all_text="|".join(result)
        x_offsets="|".join(x_offsets)
        y_offsets="|".join(y_offsets)
        widths="|".join(widths)
        heights="|".join(heights)

        print("result",result)

        # Add original image output
        original_img = image_input

        return all_text ,x_offsets,y_offsets,widths,heights,img_width,img_height, masked_img,original_img



    @retry(tries=MAX_RETRY)
    def forward(self, image):
        return self.ocr_by_textract(image)


NODE_CLASS_MAPPINGS = {
    "Image OCR By Textract": ImageOCRByTextract,
    "Image OCR By Textract V2":ImageOCRByTextractV2,
    "Image OCR By Textract V3":ImageOCRByTextractV3,
    "Image OCR by PaddleOCR": ImageOCRByTextractV4
}

