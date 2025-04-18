from comfy.comfy_types.node_typing import StrEnum

class OAIAPIIO(StrEnum):
    """
    Node input/output data types.
    """
    CLIENT = "CLIENT"
    OPTIONS = "OPTIONS"
    HISTORY = "HISTORY"
    DEBUG = f"{OPTIONS},{HISTORY}"
