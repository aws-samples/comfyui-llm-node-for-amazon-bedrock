import json

JSON_KEY_NUMBERS = 4

class JSONTextExtraction:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "json_text": ("STRING", {"multiline": False}),
            },
            "optional":{
                f"key{i+1}": ("STRING", {"multiline": False})
                for i in range(JSON_KEY_NUMBERS)
            }
        }

    RETURN_TYPES = ("STRING","STRING","STRING","STRING",)
    FUNCTION = "process"
    CATEGORY = "aws"

    def process(self, json_text, **kwargs):
        # Parse the JSON text
        data = json.loads(json_text)
        output = ()
        # loop 4 times
        for i in range(JSON_KEY_NUMBERS):
            key = f"key{i+1}"
            if key in kwargs:
                output += (data.get(kwargs[key], ""),)
            else:
                output += ("",)
                
        return output

NODE_CLASS_MAPPINGS = {
    "JSON Text Extraction": JSONTextExtraction,
}