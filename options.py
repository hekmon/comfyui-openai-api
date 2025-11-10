import json

from comfy.comfy_types import IO

from .iotypes import ParamOptions


class OptionSeed:
    CATEGORY = "OpenAI API/Options"
    RETURN_TYPES = (OAIAPIIO.OPTIONS,)
    RETURN_NAMES = ("OPTIONS",)
    FUNCTION = "merge"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": (IO.INT, {
                    "tooltip": "Repeated requests with the same seed and parameters should return the same result. Control randomness.",
                    "forceInput": True,
                }),
            },
            "optional": {
                "options": (OAIAPIIO.OPTIONS, {
                    "tooltip": "Others options to merge with",
                }),
            },
        }

    def merge(self, seed, options=None):
        if options is None:
            options = {"seed": seed}
        else:
            options = options.copy()
            options["seed"] = seed
        return (options,)


class OptionTemperature:
    CATEGORY = "OpenAI API/Options"
    RETURN_TYPES = (OAIAPIIO.OPTIONS,)
    RETURN_NAMES = ("OPTIONS",)
    FUNCTION = "merge"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "temperature": (IO.FLOAT, {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Controls randomness in the generated text. Lower values make the output more deterministic",
                }),
            },
            "optional": {
                "options": (OAIAPIIO.OPTIONS, {
                    "tooltip": "Others options to merge with",
                }),
            },
        }

    def merge(self, temperature, options=None):
        if options is None:
            options = {"temperature": temperature}
        else:
            options = options.copy()
            options["temperature"] = temperature
        return (options,)


class OptionMaxTokens:
    CATEGORY = "OpenAI API/Options"
    RETURN_TYPES = (OAIAPIIO.OPTIONS,)
    RETURN_NAMES = ("OPTIONS",)
    FUNCTION = "merge"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "max_tokens": (IO.INT, {
                    "default": 512,
                    "min": 1,
                    "tooltip": "An upper bound for the number of tokens that can be generated for a response",
                }),
            },
            "optional": {
                "options": (OAIAPIIO.OPTIONS, {
                    "tooltip": "Others options to merge with",
                }),
            },
        }

    def merge(self, max_tokens, options=None):
        if options is None:
            options = {"max_tokens": max_tokens}
        else:
            options = options.copy()
            options["max_tokens"] = max_tokens
        return (options,)


class OptionTopP:
    CATEGORY = "OpenAI API/Options"
    RETURN_TYPES = (OAIAPIIO.OPTIONS,)
    RETURN_NAMES = ("OPTIONS",)
    FUNCTION = "merge"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "top_p": (IO.FLOAT, {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "tooltip": "An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.",
                })
            },
            "optional": {
                "options": (OAIAPIIO.OPTIONS, {
                    "tooltip": "Others options to merge with",
                }),
            },
        }

    def merge(self, top_p, options=None):
        if options is None:
            options = {"top_p": top_p}
        else:
            options = options.copy()
            options["top_p"] = top_p
        return (options,)


class OptionFrequencyPenalty:
    CATEGORY = "OpenAI API/Options"
    RETURN_TYPES = (OAIAPIIO.OPTIONS,)
    RETURN_NAMES = ("OPTIONS",)
    FUNCTION = "merge"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frequency_penalty": (IO.FLOAT, {
                    "default": 0,
                    "min": -2.0,
                    "max": 2.0,
                    "tooltip": "Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.",
                }),
            },
            "optional": {
                "options": (OAIAPIIO.OPTIONS, {
                    "tooltip": "Others options to merge with",
                }),
            },
        }

    def merge(self, frequency_penalty, options=None):
        if options is None:
            options = {"frequency_penalty": frequency_penalty}
        else:
            options = options.copy()
            options["frequency_penalty"] = frequency_penalty
        return (options,)


class OptionPresencePenalty:
    CATEGORY = "OpenAI API/Options"
    RETURN_TYPES = (OAIAPIIO.OPTIONS,)
    RETURN_NAMES = ("OPTIONS",)
    FUNCTION = "merge"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "presence_penalty": (IO.FLOAT, {
                    "default": 0,
                    "min": -2.0,
                    "max": 2.0,
                    "tooltip": "Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.",
                }),
            },
            "optional": {
                "options": (OAIAPIIO.OPTIONS, {
                    "tooltip": "Others options to merge with",
                }),
            },
        }

    def merge(self, presence_penalty, options=None):
        if options is None:
            options = {"presence_penalty": presence_penalty}
        else:
            options = options.copy()
            options["presence_penalty"] = presence_penalty
        return (options,)


class OptionExtraBody:
    CATEGORY = "OpenAI API/Options"
    RETURN_TYPES = (OAIAPIIO.OPTIONS,)
    RETURN_NAMES = ("OPTIONS",)
    FUNCTION = "merge"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "extra_body": (IO.STRING, {
                    "default": json.dumps({"repetition_penalty": 0.5, "seed": 42}, indent=4),
                    "tooltip": "Extra body content to include in the request",
                    "multiline": True,
                }),
            },
            "optional": {
                "options": (OAIAPIIO.OPTIONS, {
                    "tooltip": "Others options to merge with",
                }),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, extra_body):
        try:
            data = json.loads(extra_body)
            if not isinstance(data, dict):
                return "extra body must be a JSON object (dictionary)"
        except json.JSONDecodeError as e:
            return f"extra body is not a valid JSON: {e}"
        return True

    def merge(self, extra_body, options=None):
        if options is None:
            options = json.loads(extra_body)
        else:
            options = options.copy()
            options.update(json.loads(extra_body))
        return (options,)


class OptionDeveloperRole:
    CATEGORY = "OpenAI API/Options"
    RETURN_TYPES = (OAIAPIIO.OPTIONS,)
    RETURN_NAMES = ("OPTIONS",)
    FUNCTION = "merge"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "use_developer_role": (IO.BOOLEAN, {
                    "default": True,
                    "tooltip": "With o1 models and newer, OpenAI has changed the 'system' prompt role to 'developper' prompt role. Set this switch to true to set the system prompt as 'developper'.",
                }),
            },
            "optional": {
                "options": (OAIAPIIO.OPTIONS, {
                    "tooltip": "Others options to merge with",
                }),
            },
        }

    def merge(self, use_developer_role, options=None):
        if options is None:
            options = {"use_developer_role": use_developer_role}
        else:
            options = options.copy()
            options["use_developer_role"] = use_developer_role
        return (options,)
