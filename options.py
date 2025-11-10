import json

from comfy_api.latest import io

from .iotypes import ParamOptions, OptionsPayload


class OptionSeed(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="OAIAPISeed",
            display_name="OpenAI API - Seed",
            category="OpenAI API/Options",
            inputs=[
                ParamOptions.Input(
                    id="previous_options",
                    display_name="Options",
                    optional=True,
                    tooltip="Others options to merge with",
                ),
                io.Int.Input(
                    id="seed",
                    display_name="Seed",
                    tooltip="Repeated requests with the same seed and parameters should return the same result. Control randomness.",
                    default=42,
                    control_after_generate=True,
                    display_mode=io.NumberDisplay.number,
                )
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
                previous_options: OptionsPayload | None,
                ) -> io.NodeOutput:
        if previous_options is None:
            options = {"seed": seed}
        else:
            options = previous_options.get_options_copy()
            options["seed"] = seed
        return io.NodeOutput(
            OptionsPayload(options)
        )


class OptionTemperature(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="OAIAPITemperature",
            display_name="OpenAI API - Temperature",
            category="OpenAI API/Options",
            inputs=[
                ParamOptions.Input(
                    id="previous_options",
                    display_name="Options",
                    optional=True,
                    tooltip="Others options to merge with",
                ),
                io.Float.Input(
                    id="temperature",
                    display_name="Temperature",
                    tooltip="Controls randomness in the generated text. Lower values make the output more deterministic.",
                    default=1.0,
                    min=0.0,
                    max=2.0,
                    step=0.1,
                    display_mode=io.NumberDisplay.number,
                )
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
                previous_options: OptionsPayload | None,
                ) -> io.NodeOutput:
        if previous_options is None:
            options = {"temperature": temperature}
        else:
            options = previous_options.get_options_copy()
            options["temperature"] = temperature
        return io.NodeOutput(
            OptionsPayload(options)
        )


class OptionMaxTokens(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="OAIAPIMaxTokens",
            display_name="OpenAI API - Max Tokens",
            category="OpenAI API/Options",
            inputs=[
                ParamOptions.Input(
                    id="previous_options",
                    display_name="Options",
                    optional=True,
                    tooltip="Others options to merge with",
                ),
                io.Int.Input(
                    id="max_tokens",
                    display_name="Max Tokens",
                    tooltip="An upper bound for the number of tokens that can be generated for a response",
                    default=512,
                    min=1,
                    control_after_generate=False,
                    display_mode=io.NumberDisplay.number,
                )
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
                previous_options: OptionsPayload | None,
                ) -> io.NodeOutput:
        if previous_options is None:
            options = {"max_tokens": max_tokens}
        else:
            options = previous_options.get_options_copy()
            options["max_tokens"] = max_tokens
        return io.NodeOutput(
            OptionsPayload(options)
        )


class OptionTopP(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="OAIAPITopP",
            display_name="OpenAI API - Top P",
            category="OpenAI API/Options",
            inputs=[
                ParamOptions.Input(
                    id="previous_options",
                    display_name="Options",
                    optional=True,
                    tooltip="Others options to merge with",
                ),
                io.Float.Input(
                    id="top_p",
                    display_name="Top P",
                    tooltip="An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    display_mode=io.NumberDisplay.number,
                )
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
                previous_options: OptionsPayload | None,
                ) -> io.NodeOutput:
        if previous_options is None:
            options = {"top_p": top_p}
        else:
            options = previous_options.get_options_copy()
            options["top_p"] = top_p
        return io.NodeOutput(
            OptionsPayload(options)
        )


class OptionFrequencyPenalty(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="OAIAPIFrequencyPenalty",
            display_name="OpenAI API - Frequency Penalty",
            category="OpenAI API/Options",
            inputs=[
                ParamOptions.Input(
                    id="previous_options",
                    display_name="Options",
                    optional=True,
                    tooltip="Others options to merge with",
                ),
                io.Float.Input(
                    id="frequency_penalty",
                    display_name="Top P",
                    tooltip="Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.",
                    default=0.0,
                    min=-2.0,
                    max=2.0,
                    step=0.1,
                    display_mode=io.NumberDisplay.number,
                )
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
                previous_options: OptionsPayload | None,
                ) -> io.NodeOutput:
        if previous_options is None:
            options = {"frequency_penalty": frequency_penalty}
        else:
            options = previous_options.get_options_copy()
            options["frequency_penalty"] = frequency_penalty
        return io.NodeOutput(
            OptionsPayload(options)
        )


class OptionPresencePenalty(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="OAIAPIPresencePenalty",
            display_name="OpenAI API - Presence Penalty",
            category="OpenAI API/Options",
            inputs=[
                ParamOptions.Input(
                    id="previous_options",
                    display_name="Options",
                    optional=True,
                    tooltip="Others options to merge with",
                ),
                io.Float.Input(
                    id="presence_penalty",
                    display_name="Top P",
                    tooltip="Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.",
                    default=0.0,
                    min=-2.0,
                    max=2.0,
                    step=0.1,
                    display_mode=io.NumberDisplay.number,
                )
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
                previous_options: OptionsPayload | None,
                ) -> io.NodeOutput:
        if previous_options is None:
            options = {"presence_penalty": presence_penalty}
        else:
            options = previous_options.get_options_copy()
            options["presence_penalty"] = presence_penalty
        return io.NodeOutput(
            OptionsPayload(options)
        )


class OptionExtraBody(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="OAIAPIExtraBody",
            display_name="OpenAI API - Extra Body",
            category="OpenAI API/Options",
            inputs=[
                ParamOptions.Input(
                    id="previous_options",
                    display_name="Options",
                    optional=True,
                    tooltip="Others options to merge with",
                ),
                io.String.Input(
                    id="extra_body",
                    display_name="Extra Body",
                    tooltip="Extra body content to include in the request",
                    multiline=True,
                    default=json.dumps(
                        {"repetition_penalty": 0.5, "seed": 42}, indent=4),
                )
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
                previous_options: OptionsPayload | None,
                ) -> io.NodeOutput:
        if previous_options is None:
            options = json.loads(extra_body)
        else:
            options = previous_options.get_options_copy()
            options.update(json.loads(extra_body))
        return io.NodeOutput(
            OptionsPayload(options)
        )


class OptionDeveloperRole(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="OAIAPIDeveloperRole",
            display_name="OpenAI API - Developer Role",
            category="OpenAI API/Options",
            inputs=[
                ParamOptions.Input(
                    id="previous_options",
                    display_name="Options",
                    optional=True,
                    tooltip="Others options to merge with",
                ),
                io.Boolean.Input(
                    id="use_developer_role",
                    display_name="Instructions Role",
                    tooltip="With o1 models and newer, OpenAI has changed the 'system' prompt role to 'developer' prompt role. Set this switch to true to set the system prompt as 'developer'.",
                    default=False,
                    label_on="System",
                    label_off="Developer",
                )
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
                use_developer_role: bool,
                previous_options: OptionsPayload | None,
                ) -> io.NodeOutput:
        if previous_options is None:
            options = {"use_developer_role": use_developer_role}
        else:
            options = previous_options.get_options_copy()
            options["use_developer_role"] = use_developer_role
        return io.NodeOutput(
            OptionsPayload(options)
        )
