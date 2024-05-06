import re


class PromptTemplate:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "prompt_template": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "forward"

    CATEGORY = "utils"

    def forward(self, prompt, prompt_template):
        output = prompt_template.replace("{prompt}", prompt).replace("[prompt]", prompt)
        return (output,)


class PromptRegexRemove:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "regex_string": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "replace"

    CATEGORY = "utils"

    def replace(self, prompt, regex_string):
        output = re.sub(regex_string, "", prompt)
        return (output,)


NODE_CLASS_MAPPINGS = {
    "Prompt Template": PromptTemplate,
    "Prompt Regex Remove": PromptRegexRemove,
}
