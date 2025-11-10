from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

from .client import Client
from .completions import ChatCompletion
from .options import OptionSeed, OptionTemperature, OptionMaxTokens, OptionTopP, OptionFrequencyPenalty, OptionPresencePenalty, OptionExtraBody, OptionDeveloperRole


class OpenAIAPIExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            Client,
            ChatCompletion,
            OptionSeed,
            OptionTemperature,
            OptionMaxTokens,
            OptionTopP,
            OptionFrequencyPenalty,
            OptionPresencePenalty,
            OptionExtraBody,
            OptionDeveloperRole
        ]


async def comfy_entrypoint() -> OpenAIAPIExtension:
    return OpenAIAPIExtension()
