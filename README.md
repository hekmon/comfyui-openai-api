# ComfyUI OpenAI API

This repository contains ComfyUI nodes that integrates with the OpenAI API: it allows you to use language models and vision language models within your workflow.

It is [KISS](https://en.wikipedia.org/wiki/KISS_principle) by design and intended for those who only wants basic capabilities without having to import massive projects like [LLM party](https://github.com/heshengtao/comfyui_LLM_party): only the chat completions endpoint (with vision support) is implemented as it should be enough for 99% of use cases.

Thanks to its simplicity the project has a low footprint: it only has 1 external dependency (3 in total) !

- `openai` the official OpenAI API bindings for Python
- `numpy` for computation but it is already a dependency of ComfyUI
- `Pillow` also already used by ComfyUI for image processing

## Usage

The default `base_url` parameter value targets the official OpenAI API endpoint by default but by changing it, you can also use this project with any OpenAI API compatible servers like Ollama, vLLM, SGLang, etc...

Multiples images are supported as long as they are fed batched to the chat completion node.

### Client node

- `Base URL`: the OpenAI-compatible endpoint, e.g. `http://localhost:8000/v1` for a local server.
- `Max Retries` (0 to 10, default 2): how many times a failed request (connection error, timeout, HTTP 408/409/429/5xx) is retried by the OpenAI SDK, with an exponential backoff.
- `Timeout` (seconds, default 600): maximum duration of a single request attempt. Raise it for slow local models generating long answers.

### Chat completion node

- `System Prompt` (optional): sent as the first `system` message (or `developer` with the `Developer Role` option).
- `History`: chain the `History` output of a chat completion node into the `History` input of the next one to continue the conversation. The history contains the previous messages (images included) and the model answers. A non empty `System Prompt` replaces the system message of the history; if left empty, the history one is kept (its role follows the `Developer Role` option when that option is connected).
- `Force Regen`: ComfyUI caches the node output and only calls the API again when one of its inputs changes. Enable it to send a new request on every run, e.g. to get a different answer with the same prompt. If this input is linked to another node, the request is always sent again.
- If the model returns no text (for example a reasoning model that used its whole token budget), the response is an empty string and a warning with the `finish_reason` is logged in the console. A refusal or a response without any choice raises an error.

If you want to customize the chat completion, you can chain options to modify the request. Most common options are available as predefined nodes but you can inject any key/value pair using the `Extra body` node.

Options nodes are available for:
- `seed`
- `temperature`
- `max_tokens`
- `max_completion_tokens`
- `top_p`
- `frequency_penalty`
- `presence_penalty`
- `developer_role`
- `extra_body` (for any other key/value pair)

Use either `max_tokens` or `max_completion_tokens`, not both. Older OpenAI-compatible servers keep using `max_tokens`. Newer OpenAI chat models that reject `max_tokens` use `max_completion_tokens`.

## API key

The client node reads the API key from the `OPENAI_API_KEY` environment variable **at execution time**. It is intentionally impossible to enter an API key inside the workflow.

**Never store a real API key in a workflow.** ComfyUI serializes every node widget value — even from bypassed nodes — into the `prompt` and `workflow` metadata chunks of generated images and exported workflow files. Anyone you share an image or a workflow with (Discord, Reddit, forum attachments...) can read the key back from those metadata. This is a ComfyUI core behavior (see the [official workflow metadata documentation](https://docs.comfy.org/development/api-development/workflow-metadata)) which no custom node can opt out of: any node exposing a key widget leaks it.

How to configure:

- **Official OpenAI API or any authenticated endpoint**: set the environment variable before starting ComfyUI, e.g. `export OPENAI_API_KEY=sk-...` (Linux/macOS) or `setx OPENAI_API_KEY sk-...` then restart your shell (Windows). In Docker, pass it with `-e OPENAI_API_KEY=sk-...`.
- **Keyless inference servers** (vLLM, some Ollama setups, etc.): those servers require an `Authorization` header but do not validate its value. If the environment variable is not set, the client automatically falls back to the `-` placeholder, so nothing to configure.
- **Imported workflows**: the key is sent to whatever `Base URL` the workflow's client node uses, so check it before running a workflow you did not write.
- **Defense in depth**: if you generate images to share, launch ComfyUI with `--disable-metadata` (or enable *Settings > Server-Config > Disable saving prompt metadata in files* in ComfyUI Desktop) to strip workflow metadata from output files entirely.

If you have used a version of this project prior to v3.0.0 with a real key and shared generated images or workflow files, **rotate that key** and manually clean up your saved workflows: the previous versions stored the key inside every image and workflow export.

## Installation

### ComfyUI Manager

Search for `OpenAI API` in the `Custom Nodes Manager` and install it.

![Entry in the ComfyUI manager](res/comfymanager.png)

### Manually

On the github interface, click the green `<> Code` button and then `Download ZIP`. Extract the root folder of the zip file into your `ComfyUI/custom_nodes` directory.

Then install the dependencies with the Python environment used by ComfyUI (`openai` is not a ComfyUI dependency):

```bash
pip install -r ComfyUI/custom_nodes/<extracted folder>/requirements.txt
```

With the Windows portable build, use its embedded Python: `python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\<extracted folder>\requirements.txt`.

## Example

![Example](res/example.png)
