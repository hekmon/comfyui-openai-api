from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

from .client import Client
from .completions import ChatCompletion
from .history import TrimHistory
from .memory import SaveMemory, LoadMemory
from .options import OptionSeed, OptionTemperature, OptionMaxTokens, OptionMaxCompletionTokens, OptionTopP, OptionFrequencyPenalty, OptionPresencePenalty, OptionReasoningEffort, OptionExtraBody, OptionDeveloperRole


class OpenAIAPIExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            Client,
            ChatCompletion,
            OptionSeed,
            OptionTemperature,
            OptionMaxTokens,
            OptionMaxCompletionTokens,
            OptionTopP,
            OptionFrequencyPenalty,
            OptionPresencePenalty,
            OptionReasoningEffort,
            OptionDeveloperRole,
            OptionExtraBody,
            TrimHistory,
            SaveMemory,
            LoadMemory,
        ]


async def comfy_entrypoint() -> OpenAIAPIExtension:
    return OpenAIAPIExtension()
