import json

from comfy.comfy_types import IO

from .iotypes import OAIAPIIO


class Debug:
    CATEGORY = "OpenAI API"
    RETURN_TYPES = (IO.STRING,)
    RETURN_NAMES = ("ENCODED_JSON",)
    FUNCTION = "marshall"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "options_or_history": (OAIAPIIO.DEBUG, {
                    "tooltip": "Options chain or chat history you want to encode to JSON string for debugging purposes.",
                    "forceInput": True
                }),
            },
        }

    def marshall(self, options_or_history):
        return (json.dumps(options_or_history, indent=4),)
