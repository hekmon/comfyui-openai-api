import json
from typing import Any

from comfy_api.latest import io

from .iotypes import ParamOptions, OptionsPayload, check_options


def _option_schema(node_id: str, display_name: str, description: str, value: io.Input) -> io.Schema:
    """Schema shared by all option nodes: the value input, the options to merge
    with, and the merged options as output."""
    return io.Schema(
        node_id=node_id,
        display_name=display_name,
        category="OpenAI API/Options",
        description=description,
        inputs=[
            value,
            ParamOptions.Input(
                id="other_options",
                display_name="Options",
                optional=True,
                tooltip="Others options to merge with",
            ),
        ],
        outputs=[
            ParamOptions.Output(
                id="options",
                display_name="Options",
                tooltip="Merged options to forward",
            ),
        ],
    )


class _SetOption(io.ComfyNode):
    """Base of the option nodes that set one option to the value of their input.

    The option key is the input id, unless KEY is set. To add an option: subclass
    this, return _option_schema(...) from define_schema, register the class in
    __init__.py. Never register this base class itself (ComfyUI caches the schema
    properties on the class, and subclasses would inherit them)."""

    KEY: str | None = None

    @classmethod
    def execute(cls, *, other_options: OptionsPayload | None = None, **inputs: Any) -> io.NodeOutput:
        # ComfyUI passes the inputs by name: the only one left is the value input
        ((input_id, value),) = inputs.items()
        check_options(other_options, cls.define_schema().display_name.removeprefix("OpenAI API - "))
        options = {} if other_options is None else other_options.get_options_copy()
        options[cls.KEY or input_id] = value
        return io.NodeOutput(
            OptionsPayload(options)
        )


class OptionSeed(_SetOption):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return _option_schema(
            node_id="OAIAPI_Seed",
            display_name="OpenAI API - Seed",
            description="Sets the seed for reproducible results in OpenAI API requests. Repeated requests with the same seed and parameters will return the same result, allowing for consistent randomness control.",
            value=io.Int.Input(
                id="seed",
                display_name="Seed",
                default=42,
                # Explicit bounds, otherwise the frontend applies a 0-2048
                # default range (and randomizes within it). 2^31-1 fits the
                # signed 32 bits seeds of every common backend.
                min=0,
                max=2**31 - 1,
                control_after_generate=True,
                display_mode=io.NumberDisplay.number,
            ),
        )


class OptionTemperature(_SetOption):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return _option_schema(
            node_id="OAIAPI_Temperature",
            display_name="OpenAI API - Temperature",
            description="Controls the randomness of the generated text. Lower values make the output more deterministic and focused, while higher values introduce more variation and creativity.",
            value=io.Float.Input(
                id="temperature",
                display_name="Temperature",
                tooltip="Number between 0.0 and 2.0. Defaults to 1.0.",
                default=1.0,
                min=0.0,
                max=2.0,
                step=0.1,
                display_mode=io.NumberDisplay.number,
            ),
        )


class OptionMaxTokens(_SetOption):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return _option_schema(
            node_id="OAIAPI_MaxTokens",
            display_name="OpenAI API - Max Tokens",
            description="Sets the maximum number of tokens that can be generated in the response. This helps control the length and complexity of the output.",
            value=io.Int.Input(
                id="max_tokens",
                display_name="Max Tokens",
                tooltip="An upper bound for the number of tokens that can be generated for a response",
                default=512,
                min=1,
                max=1000000, # if max is not set, ComfyUI will applied a default max value at 2048: we do not want this too low value
                control_after_generate=False,
                display_mode=io.NumberDisplay.number,
            ),
        )


class OptionMaxCompletionTokens(_SetOption):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return _option_schema(
            node_id="OAIAPI_MaxCompletionTokens",
            display_name="OpenAI API - Max Completion Tokens",
            description="Sets the maximum number of tokens that can be generated in the response for models that reject max_tokens. Do not combine this node with OpenAI API - Max Tokens.",
            value=io.Int.Input(
                id="max_completion_tokens",
                display_name="Max Completion Tokens",
                tooltip="An upper bound for the number of tokens that can be generated for a response. Use this instead of Max Tokens on models that reject max_tokens.",
                default=512,
                min=1,
                max=1000000, # if max is not set, ComfyUI will applied a default max value at 2048: we do not want this too low value
                control_after_generate=False,
                display_mode=io.NumberDisplay.number,
            ),
        )


class OptionTopP(_SetOption):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return _option_schema(
            node_id="OAIAPI_TopP",
            display_name="OpenAI API - Top P",
            description="An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass.",
            value=io.Float.Input(
                id="top_p",
                display_name="Top P",
                tooltip="0.1 means only the tokens comprising the top 10% probability mass are considered",
                default=1.0,
                min=0.0,
                max=1.0,
                step=0.01,
                display_mode=io.NumberDisplay.number,
            ),
        )


class OptionFrequencyPenalty(_SetOption):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return _option_schema(
            node_id="OAIAPI_FrequencyPenalty",
            display_name="OpenAI API - Frequency Penalty",
            description="Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.",
            value=io.Float.Input(
                id="frequency_penalty",
                display_name="Frequency Penalty",
                tooltip="Number between -2.0 and 2.0",
                default=0.0,
                min=-2.0,
                max=2.0,
                step=0.1,
                display_mode=io.NumberDisplay.number,
            ),
        )


class OptionPresencePenalty(_SetOption):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return _option_schema(
            node_id="OAIAPI_PresencePenalty",
            display_name="OpenAI API - Presence Penalty",
            description="Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.",
            value=io.Float.Input(
                id="presence_penalty",
                display_name="Presence Penalty",
                tooltip="Number between -2.0 and 2.0",
                default=0.0,
                min=-2.0,
                max=2.0,
                step=0.1,
                display_mode=io.NumberDisplay.number,
            ),
        )


class OptionReasoningEffort(_SetOption):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return _option_schema(
            node_id="OAIAPI_ReasoningEffort",
            display_name="OpenAI API - Reasoning Effort",
            description="Sets the reasoning effort of reasoning models: how much the model thinks before answering. Lower values give faster answers using fewer tokens.",
            value=io.Combo.Input(
                id="reasoning_effort",
                display_name="Reasoning Effort",
                tooltip="Backends and models support different subsets of these values (e.g. gpt-oss: low/medium/high): an unsupported value may be rejected or ignored by the server. 'none' is sent literally as the \"none\" string (on vLLM, SGLang and llama.cpp it disables thinking for models whose chat template supports it). Use the Extra Body node to send any other value.",
                options=["none", "minimal", "low", "medium", "high", "xhigh", "max"],
                default="medium",
            ),
        )


class OptionDeveloperRole(_SetOption):
    # The key read by Chat Completion; the input keeps its own id (saved workflows)
    KEY = "use_developer_role"

    @classmethod
    def define_schema(cls) -> io.Schema:
        return _option_schema(
            node_id="OAIAPI_DeveloperRole",
            display_name="OpenAI API - Developer Role",
            description="With o1 models and newer, OpenAI has changed the 'system' prompt role to 'developer' prompt role. Use this node to adapt the generation with the new 'developer' role.",
            value=io.Boolean.Input(
                id="instructions_role",
                display_name="Instructions Role",
                tooltip="Set this switch to true to set the instructions prompt role as 'developer' instead of 'system'",
                default=False,
                label_on="developer",
                label_off="system",
            ),
        )


class OptionExtraBody(io.ComfyNode):
    """Merges a JSON object into the options, instead of setting one key."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return _option_schema(
            node_id="OAIAPI_ExtraBody",
            display_name="OpenAI API - Extra Body",
            description="Add extra body parameters to the OpenAI API request. This allows you to include additional parameters that are not covered by other nodes.",
            value=io.String.Input(
                id="extra_body",
                display_name="Extra Body",
                tooltip="Extra body content to include in the request, as a JSON object. Keys are merged into the request body and override the ones set by upstream option nodes (e.g. a \"seed\" key here overrides the Seed node). Example: {\"top_k\": 40, \"repetition_penalty\": 1.1}",
                multiline=True,
                default="{}",
            ),
        )

    @staticmethod
    def check_extra_body(extra_body: Any) -> str | None:
        """The error message for an invalid extra body, None if valid."""
        if not isinstance(extra_body, str):
            return f"extra body must be a text, not {type(extra_body).__name__}"
        try:
            data = json.loads(extra_body)
        except json.JSONDecodeError as e:
            return f"extra body is not a valid JSON: {e}"
        if not isinstance(data, dict):
            return "extra body must be a JSON object (dictionary)"
        return None

    @classmethod
    def validate_inputs(cls, extra_body: str | None) -> bool | str:
        # None when coming from another node: checked at execution
        if extra_body is None:
            return True
        return cls.check_extra_body(extra_body) or True

    @classmethod
    def execute(cls,
                extra_body: str,
                other_options: OptionsPayload | None = None,
                ) -> io.NodeOutput:
        error = cls.check_extra_body(extra_body)
        if error is not None:
            raise ValueError(f"Extra Body: {error}")
        check_options(other_options, "Extra Body")
        if other_options is None:
            options = json.loads(extra_body)
        else:
            options = other_options.get_options_copy()
            options.update(json.loads(extra_body))
        return io.NodeOutput(
            OptionsPayload(options)
        )
