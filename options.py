import json

from comfy_api.latest import io

from .iotypes import ParamOptions, OptionsPayload


class OptionSeed(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="OAIAPI_Seed",
            display_name="OpenAI API - Seed",
            category="OpenAI API/Options",
            description="Sets the seed for reproducible results in OpenAI API requests. Repeated requests with the same seed and parameters will return the same result, allowing for consistent randomness control.",
            inputs=[
                io.Int.Input(
                    id="seed",
                    display_name="Seed",
                    default=42,
                    control_after_generate=True,
                    display_mode=io.NumberDisplay.number,
                ),
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

    @classmethod
    def execute(cls,
                seed: int,
                other_options: OptionsPayload | None = None,
                ) -> io.NodeOutput:
        if other_options is None:
            options = {"seed": seed}
        else:
            options = other_options.get_options_copy()
            options["seed"] = seed
        return io.NodeOutput(
            OptionsPayload(options)
        )


class OptionTemperature(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="OAIAPI_Temperature",
            display_name="OpenAI API - Temperature",
            category="OpenAI API/Options",
            description="Controls the randomness of the generated text. Lower values make the output more deterministic and focused, while higher values introduce more variation and creativity.",
            inputs=[
                io.Float.Input(
                    id="temperature",
                    display_name="Temperature",
                    tooltip="Number between 0.0 and 2.0. Defaults to 1.0.",
                    default=1.0,
                    min=0.0,
                    max=2.0,
                    step=0.1,
                    display_mode=io.NumberDisplay.number,
                ),
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

    @classmethod
    def execute(cls,
                temperature: float,
                other_options: OptionsPayload | None = None,
                ) -> io.NodeOutput:
        if other_options is None:
            options = {"temperature": temperature}
        else:
            options = other_options.get_options_copy()
            options["temperature"] = temperature
        return io.NodeOutput(
            OptionsPayload(options)
        )


class OptionMaxTokens(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="OAIAPI_MaxTokens",
            display_name="OpenAI API - Max Tokens",
            category="OpenAI API/Options",
            description="Sets the maximum number of tokens that can be generated in the response. This helps control the length and complexity of the output.",
            inputs=[
                io.Int.Input(
                    id="max_tokens",
                    display_name="Max Tokens",
                    tooltip="An upper bound for the number of tokens that can be generated for a response",
                    default=512,
                    min=1,
                    control_after_generate=False,
                    display_mode=io.NumberDisplay.number,
                ),
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

    @classmethod
    def execute(cls,
                max_tokens: int,
                other_options: OptionsPayload | None = None,
                ) -> io.NodeOutput:
        if other_options is None:
            options = {"max_tokens": max_tokens}
        else:
            options = other_options.get_options_copy()
            options["max_tokens"] = max_tokens
        return io.NodeOutput(
            OptionsPayload(options)
        )


class OptionTopP(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="OAIAPI_TopP",
            display_name="OpenAI API - Top P",
            category="OpenAI API/Options",
            description="An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass.",
            inputs=[
                io.Float.Input(
                    id="top_p",
                    display_name="Top P",
                    tooltip="0.1 means only the tokens comprising the top 10% probability mass are considered",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    display_mode=io.NumberDisplay.number,
                ),
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

    @classmethod
    def execute(cls,
                top_p: float,
                other_options: OptionsPayload | None = None,
                ) -> io.NodeOutput:
        if other_options is None:
            options = {"top_p": top_p}
        else:
            options = other_options.get_options_copy()
            options["top_p"] = top_p
        return io.NodeOutput(
            OptionsPayload(options)
        )


class OptionFrequencyPenalty(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="OAIAPI_FrequencyPenalty",
            display_name="OpenAI API - Frequency Penalty",
            category="OpenAI API/Options",
            description="Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.",
            inputs=[
                io.Float.Input(
                    id="frequency_penalty",
                    display_name="Top P",
                    tooltip="Number between -2.0 and 2.0",
                    default=0.0,
                    min=-2.0,
                    max=2.0,
                    step=0.1,
                    display_mode=io.NumberDisplay.number,
                ),
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

    @classmethod
    def execute(cls,
                frequency_penalty: float,
                other_options: OptionsPayload | None = None,
                ) -> io.NodeOutput:
        if other_options is None:
            options = {"frequency_penalty": frequency_penalty}
        else:
            options = other_options.get_options_copy()
            options["frequency_penalty"] = frequency_penalty
        return io.NodeOutput(
            OptionsPayload(options)
        )


class OptionPresencePenalty(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="OAIAPI_PresencePenalty",
            display_name="OpenAI API - Presence Penalty",
            category="OpenAI API/Options",
            description="Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.",
            inputs=[
                io.Float.Input(
                    id="presence_penalty",
                    display_name="Top P",
                    tooltip="Number between -2.0 and 2.0",
                    default=0.0,
                    min=-2.0,
                    max=2.0,
                    step=0.1,
                    display_mode=io.NumberDisplay.number,
                ),
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

    @classmethod
    def execute(cls,
                presence_penalty: float,
                other_options: OptionsPayload | None = None,
                ) -> io.NodeOutput:
        if other_options is None:
            options = {"presence_penalty": presence_penalty}
        else:
            options = other_options.get_options_copy()
            options["presence_penalty"] = presence_penalty
        return io.NodeOutput(
            OptionsPayload(options)
        )


class OptionDeveloperRole(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="OAIAPI_DeveloperRole",
            display_name="OpenAI API - Developer Role",
            category="OpenAI API/Options",
            description="With o1 models and newer, OpenAI has changed the 'system' prompt role to 'developer' prompt role. Use this node to adapt the generation with the new 'developer' role.",
            inputs=[
                io.Boolean.Input(
                    id="instructions_role",
                    display_name="Instructions Role",
                    tooltip="Set this switch to true to set the instructions prompt role as 'developer' instead of 'system'",
                    default=False,
                    label_on="developer",
                    label_off="system",
                ),
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

    @classmethod
    def execute(cls,
                instructions_role: bool,
                other_options: OptionsPayload | None = None,
                ) -> io.NodeOutput:
        if other_options is None:
            options = {"use_developer_role": instructions_role}
        else:
            options = other_options.get_options_copy()
            options["use_developer_role"] = instructions_role
        return io.NodeOutput(
            OptionsPayload(options)
        )


class OptionExtraBody(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="OAIAPI_ExtraBody",
            display_name="OpenAI API - Extra Body",
            category="OpenAI API/Options",
            description="Add extra body parameters to the OpenAI API request. This allows you to include additional parameters that are not covered by other nodes.",
            inputs=[
                io.String.Input(
                    id="extra_body",
                    display_name="Extra Body",
                    tooltip="Extra body content to include in the request",
                    multiline=True,
                    default=json.dumps(
                        {"repetition_penalty": 0.5, "seed": 42}, indent=4),
                ),
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

    @classmethod
    def validate_inputs(cls, extra_body: str) -> bool | str:
        try:
            data = json.loads(extra_body)
            if not isinstance(data, dict):
                return "extra body must be a JSON object (dictionary)"
        except json.JSONDecodeError as e:
            return f"extra body is not a valid JSON: {e}"
        return True

    @classmethod
    def execute(cls,
                extra_body: str,
                other_options: OptionsPayload | None = None,
                ) -> io.NodeOutput:
        if other_options is None:
            options = json.loads(extra_body)
        else:
            options = other_options.get_options_copy()
            options.update(json.loads(extra_body))
        return io.NodeOutput(
            OptionsPayload(options)
        )
