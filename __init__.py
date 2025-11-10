from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

from .client import Client
from .completions import ChatCompletion
from .options import OptionSeed, OptionTemperature, OptionMaxTokens, OptionTopP, OptionFrequencyPenalty, OptionPresencePenalty, OptionExtraBody, OptionDeveloperRole

# NODE_CLASS_MAPPINGS = {
#     "OAIAPIClient": Client,
#     "OAIAPIChatCompletion": ChatCompletion,
#     "OAIAPIDebug": Debug,
#     "OAIAPISeed": OptionSeed,
#     "OAIAPITemperature": OptionTemperature,
#     "OAIAPIMaxTokens": OptionMaxTokens,
#     "OAIAPITopP": OptionTopP,
#     "OAIAPIFrequencyPenalty": OptionFrequencyPenalty,
#     "OAIAPIPresencePenalty": OptionPresencePenalty,
#     "OAIAPIExtraBody": OptionExtraBody,
#     "OAIAPIDeveloperRole": OptionDeveloperRole,
# }

# NODE_DISPLAY_NAME_MAPPINGS = {
#     "OAIAPIClient": "OpenAI API - Client",
#     "OAIAPIChatCompletion": "OpenAI API - Chat Completion",
#     "OAIAPIDebug": "OpenAI API - Debug",
#     "OAIAPISeed": "OpenAI API - Seed",
#     "OAIAPITemperature": "OpenAI API - Temperature",
#     "OAIAPIMaxTokens": "OpenAI API - Max Tokens",
#     "OAIAPITopP": "OpenAI API - Top P",
#     "OAIAPIFrequencyPenalty": "OpenAI API - Frequency Penalty",
#     "OAIAPIPresencePenalty": "OpenAI API - Presence Penalty",
#     "OAIAPIExtraBody": "OpenAI API - Extra Body",
#     "OAIAPIDeveloperRole": "OpenAI API - Developer Role",
# }

# __all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']


class OpenAIAPIExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            Client,
            ChatCompletion,
            # Debug,
            # OptionSeed,
            # OptionTemperature,
            # OptionMaxTokens,
            # OptionTopP,
            # OptionFrequencyPenalty,
            # OptionPresencePenalty,
            # OptionExtraBody,
            # OptionDeveloperRole
        ]


async def comfy_entrypoint() -> OpenAIAPIExtension:
    return OpenAIAPIExtension()
