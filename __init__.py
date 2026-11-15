from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

from .client import Client
from .completions import ChatCompletion
from .history import TrimHistory, TrimImages
from .memory import SaveMemory, LoadMemory
from .unload import UnloadModel
from .options import OPTION_NODES


class OpenAIAPIExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            Client,
            ChatCompletion,
            *OPTION_NODES,
            TrimHistory,
            TrimImages,
            SaveMemory,
            LoadMemory,
            UnloadModel,
        ]


async def comfy_entrypoint() -> OpenAIAPIExtension:
    return OpenAIAPIExtension()
