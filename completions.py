import base64
from io import BytesIO

import torch
import numpy as np
from PIL import Image
from comfy_api.latest import io
from openai import OpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_content_part_param import ChatCompletionContentPartParam

from .iotypes import ParamClient, ParamHistory, ParamOptions, HistoryPayload


def comfy_image_to_base64_png_url(image: torch.Tensor) -> str:
    # Taken from the SaveImage ComfyUI node, convert the tensor into a regular image
    i = np.multiply(255., image.cpu().numpy())
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
    # Encode the image as PNG in base64 format
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    b64_png = base64.b64encode(buffer.getvalue())
    # Return the formated string URL
    return f"data:image/png;base64,{b64_png.decode('utf-8')}"


class ChatCompletion(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="OAIAPI_ChatCompletion",
            display_name="OpenAI API - Chat Completion",
            category="OpenAI API",
            inputs=[
                ParamClient.Input(
                    id="client",
                    display_name="API Client",
                    tooltip="The OpenAI API client to use to perform the request"
                ),
                io.String.Input(
                    id="model",
                    display_name="Model",
                    tooltip="The model to use for generating text",
                    placeholder="Model name",
                ),
                io.String.Input(
                    id="prompt",
                    display_name="Prompt",
                    tooltip="The prompt to use for generating text",
                    multiline=True,
                    placeholder="user prompt is mandatory",
                ),
                io.String.Input(
                    id="system_prompt",
                    display_name="System Prompt",
                    optional=True,
                    tooltip="The system prompt to send along with the user prompt",
                    multiline=True,
                    placeholder="system/developer prompt is optional",
                ),
                ParamHistory.Input(
                    id="history",
                    display_name="History",
                    optional=True,
                    tooltip="Previous conversation history",
                ),
                ParamOptions.Input(
                    id="options",
                    display_name="Options",
                    optional=True,
                    tooltip="Additional options to pass with the request",
                ),
                io.Image.Input(
                    id="images",
                    display_name="image(s)",
                    optional=True,
                    tooltip="Image(s) to include in the request",
                ),
            ],
            outputs=[
                io.String.Output(
                    id="response",
                    display_name="Response",
                    tooltip="Generated text response",
                ),
                ParamHistory.Output(
                    id="history",
                    display_name="History",
                    tooltip="Conversation history",
                ),
            ],
        )

    @classmethod
    def validate_inputs(cls, model: str, prompt: str) -> bool | str:
        if model == "":
            return "model must be specified"
        if prompt == "":
            return "prompt must be specified"
        return True

    @classmethod
    def execute(cls,
                client: OpenAI,
                model: str,
                prompt: str,
                system_prompt: str | None = None,
                history: HistoryPayload | None = None,
                options: dict | None = None,
                images: list[torch.Tensor] | None = None,
                ) -> io.NodeOutput:
        # Handle options
        seed: int | None = None
        temperature: float | None = None
        max_tokens: int | None = None
        top_p: float | None = None
        frequency_penalty: float | None = None
        presence_penalty: float | None = None
        use_developer_role: bool = False
        if options is not None:
            options = options.copy()
            if "seed" in options:
                seed = options["seed"]
                del options["seed"]
            if "temperature" in options:
                temperature = options["temperature"]
                del options["temperature"]
            if "max_tokens" in options:
                max_tokens = options["max_tokens"]
                del options["max_tokens"]
            if "top_p" in options:
                top_p = options["top_p"]
                del options["top_p"]
            if "frequency_penalty" in options:
                frequency_penalty = options["frequency_penalty"]
                del options["frequency_penalty"]
            if "presence_penalty" in options:
                presence_penalty = options["presence_penalty"]
                del options["presence_penalty"]
            if "use_developer_role" in options:
                use_developer_role = options["use_developer_role"]
                del options["use_developer_role"]
        # Handle system prompt
        if history is not None:
            messages = history.get_msgs_copy()
            if system_prompt is not None:
                # Should we insert it at the beginning or replace the existing system message?
                first_msg_role = messages[0].get('role')
                if first_msg_role == "system" or first_msg_role == "developer":
                    # Replace the existing system message
                    if use_developer_role:
                        messages[0] = {
                            "role": "developer",  # need literal for type hint check
                            "content": system_prompt,
                        }
                    else:
                        messages[0] = {
                            "role": "system",  # need literal for type hint check
                            "content": system_prompt,
                        }
                else:
                    # insert a new system/dev message at the begining of the list
                    if use_developer_role:
                        messages.insert(0, {
                            "role": "developer",  # need literal for type hint check
                            "content": system_prompt,
                        })
                    else:
                        messages.insert(0, {
                            "role": "system",  # need literal for type hint check
                            "content": system_prompt,
                        })
        else:
            messages: list[ChatCompletionMessageParam] = []
            if system_prompt:
                if use_developer_role:
                    messages.append({
                        "role": "developer",  # need literal for type hint check
                        "content": system_prompt,
                    })
                else:
                    messages.append({
                        "role": "system",  # need literal for type hint check
                        "content": system_prompt,
                    })
        # Handle user message
        if images is not None:
            # Build multi modal content
            content: list[ChatCompletionContentPartParam] = []
            for image in images:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": comfy_image_to_base64_png_url(image)
                        }
                    }
                )
                content.append(
                    {
                        "type": "text",
                        "text": prompt
                    }
                )
            # Add the multi-modal content to the messages list
            messages.append(
                {
                    "role": "user",
                    "content": content,
                }
            )
        else:
            messages.append(
                {
                    "role": "user",
                    "content": prompt
                }
            )
        # Create the completion
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            seed=seed,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            extra_body=options,
            n=1
        )
        # Add the response to the history
        messages.append(
            {
                "role": completion.choices[0].message.role,
                "content": completion.choices[0].message.content
            }
        )
        # Return the response and the history
        return io.NodeOutput(
            completion.choices[0].message.content,
            messages,
        )
