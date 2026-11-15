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

- `Base URL`: the OpenAI-compatible endpoint **including its API path prefix**, usually `/v1`: the part before `/chat/completions`, as requests are sent to `<Base URL>/chat/completions` and nothing is added automatically. Examples: `http://localhost:8000/v1` (vLLM, SGLang, tabbyAPI...), `http://localhost:11434/v1` (ollama), `http://localhost:1234/v1` (LM Studio), `https://api.openai.com/v1`. Some providers use another prefix; llama.cpp also answers without `/v1` (e.g. `http://localhost:8080`). A trailing slash makes no difference.
- `Max Retries` (0 to 10, default 2): how many times a failed request (connection error, timeout, HTTP 408/409/429/5xx) is retried by the OpenAI SDK, with an exponential backoff.
- `Timeout` (seconds, default 600): maximum duration of a single request attempt. Raise it for slow local models generating long answers.
- Connect one client node to as many chat completion (and unload) nodes as needed: they share its connections to the server, kept open between requests and across runs (as long as ComfyUI keeps the client node cached), instead of opening a new one per request.

### Chat completion node

- `System Prompt` (optional): sent as the first `system` message (or `developer` with the `Developer Role` option).
- `History`: chain the `History` output of a chat completion node into the `History` input of the next one to continue the conversation. The history contains the previous messages (images included) and the model answers. A non empty `System Prompt` replaces the system message of the history; if left empty, the history one is kept (its role follows the `Developer Role` option when that option is connected).
- `Force Regen`: ComfyUI caches the node output and only calls the API again when one of its inputs changes. Enable it to send a new request on every run, e.g. to get a different answer with the same prompt. If this input is linked to another node, the request is always sent again.
- If the model returns no text (for example a reasoning model that used its whole token budget), the response is an empty string and a warning with the `finish_reason` is logged in the console. A refusal or a response without any choice raises an error.
- The ComfyUI `Cancel` button (or `POST /interrupt`) stops a running request at once, instead of waiting for the answer or the client timeout. It also stops the encoding of the input images, which runs outside ComfyUI's event loop so that other nodes keep running meanwhile.
- Chat completion nodes that do not depend on each other (independent branches of the workflow) send their requests at the same time, so the run takes as long as the slowest one instead of the sum. The server must handle concurrent requests for this to help (most local servers queue or batch them depending on their settings).

If you want to customize the chat completion, you can chain options to modify the request. Most common options are available as predefined nodes but you can inject any key/value pair using the `Extra body` node.

Options nodes are available for:
- `seed`
- `temperature`
- `max_tokens`
- `max_completion_tokens`
- `top_p`
- `frequency_penalty`
- `presence_penalty`
- `reasoning_effort` (`none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`)
- `developer_role`
- `extra_body` (for any other key/value pair)

`reasoning_effort` values supported depend on the backend and the model (e.g. gpt-oss only knows `low`, `medium` and `high`): an unsupported value may be rejected or ignored by the server, any other value can be sent with the `Extra body` node. `none` is sent literally as the `"none"` string, it is not the same as not using the node (which lets the model use its default): on vLLM, SGLang and llama.cpp it disables thinking for models whose chat template supports it.

Use either `max_tokens` or `max_completion_tokens`, not both. Older OpenAI-compatible servers keep using `max_tokens`. Newer OpenAI chat models that reject `max_tokens` use `max_completion_tokens`.

### Trim History node

Limits a history to its last `Max Turns` turns. The leading system/developer message is kept. A turn starts at a user message and includes the answers that follow it, so an exchange is never split and the first kept message (after the system one) is always a user message. An empty input (`None`) is passed through as is.

## Memory (looping workflows)

The `Save Memory` and `Load Memory` nodes carry a value from one queued run to the next: an answer, an image analysis, a whole conversation history... Typical loop: generate an image, analyse it with a chat completion, save the analysis (or the history), and on the next run load it to improve the next prompt.

- `Save Memory` stores a copy of any value under a text `Key` and passes the value through. It runs on every queued prompt, even with nothing connected downstream.
- `Load Memory` returns a copy of the value stored under its `Key`, or nothing (`None`, as if the input was not connected) when the key is empty or nothing is stored yet, e.g. on the first run. A stored `None` counts as nothing stored. Connect its optional `Default` input to return a value instead in that case: needed when the output goes into a required input, like a text concatenation.
- `Load Memory` returns the value saved during the **previous** run: ComfyUI decides which nodes must run before executing any of them. `Load Memory` and the nodes downstream stay cached only while no `Save Memory` writes that key: every `Save Memory` run counts as a change, even with an identical value, so in a loop they run again on every queued prompt.
- If a `Save Memory` and a `Load Memory` use the same key but are not connected to each other (directly or not), their execution order in a run is undefined. Connect the loop as `Load Memory` -> ... -> `Save Memory`.
- `Reset` wipes the key and returns nothing (or `Default`) for this run. **Untick it afterwards**, otherwise the key is wiped on every run.
- The memory is a helper, not a database: it is kept in RAM only, shared by every prompt and workflow using the same key, and lost when ComfyUI restarts.
- Stored histories grow with every run: put a `Trim History` node before `Save Memory`.

Example workflows (API format: open them with *Workflow > Open* or drag and drop them in ComfyUI, then set the client `Base URL` and the model name):

- [`examples/memory_chat_loop_api.json`](examples/memory_chat_loop_api.json): a text only chat that remembers the conversation across queued runs. Change the prompt and queue again; tick `Reset` on the `Load Memory` node for one run to start a new conversation.
- [`examples/memory_image_feedback_loop_api.json`](examples/memory_image_feedback_loop_api.json): a prompt writer generates the image prompt, a vision model critiques the generated image, and the next run writes an improved prompt from that critique. Pick your checkpoint in the `Load Checkpoint` node and use a vision capable model.

## Unloading the model (local servers)

The `Unload Model` node asks a local inference server to unload its model, e.g. to free VRAM before an image generation. Connect the output to unload after (usually the chat completion response) to its `Value` input: the node runs once that value is ready and passes it through unchanged. Connect the same client node as your chat completion node, no change needed (keep the usual `.../v1` base URL): its API key and its connection to the server are reused, and the server root used by the native unload routes below is derived from the base URL automatically.

- `Server Type`: pick your server, there is no "try every server" mode. `Model` is the model name, as in the chat completion node: required for llama.cpp, LM Studio and ollama.
- `Timeout` (default 10 s) is specific to the unload request. The unload is never retried (the client `Max Retries` does not apply) and redirects are not followed.
- If the unload fails, a warning is logged and the value is still passed through; enable `Fail On Error` to stop the workflow instead.
- `Cancel` also stops a pending unload request at once.

| Server | Unload call (from the server root) | Reload on the next request | Server side requirements |
|---|---|---|---|
| ollama | `POST /api/generate` with `keep_alive: 0` | yes, automatic | none |
| LM Studio | `POST /api/v1/models/unload` | yes, with *Just-in-time model loading* (enabled by default) | the API token in `OPENAI_API_KEY` if authentication is enabled |
| llama.cpp | `POST /models/unload` | yes, unless started with `--no-models-autoload` | router mode only: `llama-server` started without `-m` (`--models-dir`, `--models-preset` or the cache) |
| tabbyAPI | `POST /v1/model/unload` | only with `inline_model_loading: true` in `config.yml`, and the model name of the chat completion must match a folder of its `model_dir` | an admin key in `OPENAI_API_KEY` (unload and inline loading both require it) |

vLLM (sleep mode) and SGLang (memory release) are not supported: after their unload call the next request waits or fails until an explicit wake up / resume call, which would break the next generation.

This node comes from the idea and the unload helper of [PR #11](https://github.com/hekmon/comfyui-openai-api/pull/11) by nikich340.

## API key

The client node reads the API key from the `OPENAI_API_KEY` environment variable **at execution time**. It is intentionally impossible to enter an API key inside the workflow.

**Never store a real API key in a workflow.** ComfyUI serializes every node widget value — even from bypassed nodes — into the `prompt` and `workflow` metadata chunks of generated images and exported workflow files. Anyone you share an image or a workflow with (Discord, Reddit, forum attachments...) can read the key back from those metadata. This is a ComfyUI core behavior (see the [official workflow metadata documentation](https://docs.comfy.org/development/api-development/workflow-metadata)) which no custom node can opt out of: any node exposing a key widget leaks it.

How to configure:

- **Official OpenAI API or any authenticated endpoint**: set the environment variable before starting ComfyUI, e.g. `export OPENAI_API_KEY=sk-...` (Linux/macOS) or `setx OPENAI_API_KEY sk-...` then restart your shell (Windows). In Docker, pass it with `-e OPENAI_API_KEY=sk-...`.
- **ComfyUI Desktop**: an installation can also define its own environment variables, which override the system ones: make sure `OPENAI_API_KEY` there holds the key only. The client node refuses a key with non ASCII or control characters (e.g. pasted text) with an explicit error, without showing it.
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
