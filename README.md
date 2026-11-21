# ComfyUI OpenAI API

![OpenAI API banner](res/banner.png)

ComfyUI nodes to use language models and vision language models in your workflows, through any OpenAI compatible chat completions API.

- **Any OpenAI compatible server**: local ones (llama.cpp, vLLM, SGLang, Ollama, LM Studio, tabbyAPI, Unsloth Studio...) or cloud providers (OpenAI, OpenRouter...).
- **Text and images**: send a batch of images along with the prompt to a vision model.
- **Conversations**: chain the history from node to node, or carry it across queued runs with the memory nodes to build [looping workflows](#looping-workflows).
- **Reasoning (thinking) models**: their reasoning is kept in the history and, optionally, sent back to the server on the next turns, for models that reuse their past thinking. It can also be printed in the console.
- **Responsive**: the ComfyUI `Cancel` button stops a running request at once instead of waiting for the full answer, and independent requests run in parallel.
- **Local servers friendly**: [unload the model](#unload-model) to free VRAM before an image generation, keep long conversations [bounded](#trim-history), see the [prompt cache](#prompt-cache-and-usage-stats) usage.

It is [KISS](https://en.wikipedia.org/wiki/KISS_principle) by design, for those who want these basics without importing a massive project like [LLM party](https://github.com/heshengtao/comfyui_LLM_party): the chat completions endpoint (plus the native unload routes of a few local servers) should be enough for 99% of use cases.

## Quick start

This pack is only the client: you need an OpenAI compatible **server**, either a local one with a model downloaded (e.g. [Ollama](https://ollama.com) or [LM Studio](https://lmstudio.ai)) or a cloud account (OpenAI, OpenRouter...).

1. **Install** the pack: search `OpenAI API` in the ComfyUI Manager (see [Installation](#installation)).
2. **Add the Client node**: double click the canvas, search `OpenAI API` (every node of the pack is named `OpenAI API - ...`) and add **OpenAI API - Client**.
3. **Set its `Base URL`**: the address of the server API, given in its documentation or startup log, **with its path prefix** (usually `/v1`). E.g. `http://localhost:11434/v1` for Ollama, `http://localhost:1234/v1` for LM Studio. The default, `https://api.openai.com/v1`, is OpenAI itself.
4. **API key** (cloud providers, or a local server protected by a key): set the `OPENAI_API_KEY` environment variable before starting ComfyUI. It can not be entered in the workflow, on purpose (see [API key](#api-key)). Local servers usually need none: skip this step.
5. **Add a Chat Completion node**, connect the client to its `API Client` input, and set:
   - `Model`: the exact model name the server knows, e.g. `llama3.2` for Ollama (`ollama list`). Most servers list theirs at `<Base URL>/models`: open e.g. `http://localhost:11434/v1/models` in a browser.
   - `Prompt`: your question.
6. **Show the answer**: drag from the `Response` output, pick the core **Preview Any** node, then click *Run*.

Running again with the same inputs gives the same answer without calling the server: ComfyUI cached it. Enable `Force Regen` to get a new one on every run, or feed the history of the previous run back with Load Memory to continue the conversation (see [Looping workflows](#looping-workflows)).

### Examples

Open them with *Workflow > Open* (or drag and drop the file in ComfyUI), then set the client `Base URL` and the model name. A Manager install also has them in `ComfyUI/custom_nodes/<pack folder>/examples/`.

[`examples/memory_chat_loop.json`](examples/memory_chat_loop.json): a text only chat that remembers the conversation across queued runs. Change the prompt and run again; tick `Reset` on the Load Memory node for one run to start a new conversation.

![Memory chat loop example](res/memory_chat_loop.png)

[`examples/memory_image_feedback_loop.json`](examples/memory_image_feedback_loop.json): a prompt writer generates the image prompt, a vision model critiques the generated image, and the next run writes an improved prompt from that critique. Pick your image checkpoint in the Load Checkpoint node, and use a vision capable model (e.g. Qwen2.5-VL, Gemma 3).

![Memory image feedback loop example](res/memory_image_feedback_loop.png)

## Nodes

Every node name starts with `OpenAI API - `, and they are all in the `OpenAI API` category.

| Node | What it does |
|---|---|
| [Client](#client) | The connection to the server, shared by the other nodes |
| [Chat Completion](#chat-completion) | Sends a prompt (and images, and a history), returns the answer and the updated history |
| [Options](#options) | Request parameters (temperature, max tokens, seed...) chained into the Chat Completion |
| [Trim History](#trim-history) | Keeps a history within a number of turns and images |
| [Save Memory / Load Memory](#save-memory--load-memory) | Carry any value from one queued run to the next |
| [Unload Model](#unload-model) | Asks a local server to unload its model, e.g. to free VRAM |

- **Advanced inputs**: rarely changed settings, marked *advanced* below. With the *Nodes 2.0* display, click *Show advanced inputs* at the bottom of the node to see them; the classic display shows them with the other widgets.
- **Console messages**: the log messages of this pack (usage stats, reasoning, warnings, HTTP requests...) start with `[openai-api]`, followed by the node name, or by the library name in brackets for the messages of the libraries it uses (e.g. `[openai-api] [httpx] HTTP Request: ...`).

### Client

| Input | Default | Description |
|---|---|---|
| `Base URL` | `https://api.openai.com/v1` | The server API, **including its path prefix** (usually `/v1`) |
| `Max Retries` (advanced) | 2 | Retries of a failed request (connection error, timeout, HTTP 408/409/429/5xx), with an exponential backoff. 0 to 10 |
| `Timeout` (advanced) | 600 | Maximum duration of a request attempt, in seconds. Raise it for slow local models writing long answers |

- **The Base URL is the part before `/chat/completions`**: requests go to `<Base URL>/chat/completions`, nothing is added automatically. Surrounding spaces and a trailing slash make no difference.
- **Common Base URLs**: `http://localhost:8000/v1` (vLLM, SGLang, tabbyAPI...), `http://localhost:11434/v1` (Ollama), `http://localhost:1234/v1` (LM Studio), `https://api.openai.com/v1`. Some providers use another prefix; llama.cpp also answers without `/v1` (e.g. `http://localhost:8080`).
- **`localhost` is the machine running ComfyUI**: with ComfyUI in Docker or the server on another computer, use the server IP address (or `host.docker.internal` from a container).
- **One client for many nodes**: connect it to as many Chat Completion and Unload Model nodes as needed. They share its connections to the server, kept open between requests and across runs (as long as ComfyUI keeps the Client node cached).

### Chat Completion

| Input | Default | Description |
|---|---|---|
| `API Client` | | The [Client](#client) node |
| `Model` | | The model name, as the server knows it |
| `Force Regen` | off | Sends a new request on every run (see below) |
| `Prompt` | | The user message, required |
| `System Prompt` | empty | Optional instructions, sent as the first `system` message (or `developer`, see [Options](#options)) |
| `History` | | Optional: the previous conversation, from the `History` output of another Chat Completion or from [Load Memory](#save-memory--load-memory) |
| `Options` | | Optional: request parameters, from the [option nodes](#options) |
| `Image(s)` | | Optional: an image or a batch of images, sent before the prompt. Use a vision capable model |
| `Send Back Reasoning` (advanced) | on | Sends the reasoning kept in the history back to the server (see [Reasoning models](#reasoning-models)) |
| `Show Reasoning` (advanced) | off | Prints the reasoning in the console |

| Output | Description |
|---|---|
| `Response` | The answer text |
| `History` | The history received (with the `System Prompt` applied), plus the new prompt (images included) and the answer |

- **Conversations**: chain the `History` output into the `History` input of the next Chat Completion to continue the conversation.
- **System prompt and history**: a non empty `System Prompt` replaces the system message of the history. If left empty, the history one is kept (its role follows the [Developer Role](#options) option when that option is connected).
- **Force Regen**: ComfyUI caches the node output and only calls the API again when one of its inputs changes. Enable it to get a different answer with the same inputs. If this input is linked to another node, the request is always sent again. Changing any widget (`Show Reasoning` included) also sends a new request.
- **Usage stats**: the token usage of each request is shown under the node and in the console (see [Prompt cache and usage stats](#prompt-cache-and-usage-stats)).
- **Empty answers**: if the server returns no text (e.g. a reasoning model that used its whole token budget), the response is an empty string and a warning with the `finish_reason` is logged; when a reasoning came back, the warning suggests raising the max tokens. A refusal, or a response without any choice or message, raises an error.
- **Cancel**: the ComfyUI `Cancel` button (or `POST /interrupt`) stops a running request at once, instead of waiting for the answer or the client timeout. It also stops the encoding of the input images.
- **Parallel requests**: Chat Completion nodes that do not depend on each other send their requests at the same time, so the run takes as long as the slowest one instead of the sum. The server must handle concurrent requests for this to help (most local servers queue or batch them, depending on their settings).
- **No blocking**: the images are encoded outside ComfyUI's event loop, so other nodes keep running meanwhile.

### Options

Chain option nodes (each one has an `Options` input and output) and connect the last one to the Chat Completion `Options` input. Without an option node, the server defaults apply.

| Node | Sends | Default | Notes |
|---|---|---|---|
| Seed | `seed` | 42 | 0 to 2³¹-1. Best effort reproducibility, depending on the server |
| Temperature | `temperature` | 1.0 | 0 to 2 |
| Max Tokens | `max_tokens` | 512 | Older models and most OpenAI compatible servers. Raise it for reasoning models |
| Max Completion Tokens | `max_completion_tokens` | 512 | Newer OpenAI models that reject `max_tokens` |
| Top P | `top_p` | 1.0 | 0 to 1 |
| Frequency Penalty | `frequency_penalty` | 0 | -2 to 2 |
| Presence Penalty | `presence_penalty` | 0 | -2 to 2 |
| Reasoning Effort | `reasoning_effort` | `medium` | `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max` |
| Developer Role | no field | `system` | Its `Instructions Role` toggle sets the role of the instructions message: set it to `developer` for o1 and newer OpenAI models |
| Extra Body | any key | `{}` | A JSON object merged into the request, for any other parameter, e.g. `{"top_k": 40}` |

- **Max Tokens or Max Completion Tokens, not both**: setting both (Extra Body included) raises an error.
- **512 is low for reasoning models**: the reasoning counts in the budget, and once it is spent the answer is empty. Raise it, e.g. to a few thousands.
- **Seed changes on every run by default**: its *control after generate* widget is set to *randomize*, so every run sends a new request. Set it to *fixed* to keep the seed.
- **Developer Role set to `system`** also converts a `developer` message of the history back to `system`.
- **Reasoning Effort values depend on the server and the model** (e.g. gpt-oss only knows `low`, `medium` and `high`): an unsupported value may be rejected or ignored. Any other value can be sent with Extra Body.
- **`none` is sent as is**, it is not the same as no Reasoning Effort node (which lets the model use its default): on vLLM, SGLang and llama.cpp, it disables thinking for models whose chat template supports it.
- **Extra Body keys override** the ones of the previous option nodes. A `null` value for an option key (e.g. `{"seed": null}`) removes that option from the request.
- **Chat template options** go through Extra Body. The chat template is the model specific format the server turns the messages into; some have switches, e.g. `{"chat_template_kwargs": {"enable_thinking": false}}` disables the thinking of Qwen3 models.

### Trim History

Keeps a history within a number of turns and, optionally, of images: for servers that accept only a few images per conversation, or to save context. A **turn** is a user message and the answer(s) to it.

| Input | Default | Description |
|---|---|---|
| `History` | | The history to trim |
| `Max Turns` | 10 | Up to this number of turns, no turn is dropped |
| `Max Images` | -1 (no limit) | Up to this number of images, no image is removed. 0 keeps no image at all, e.g. only the text of the image analyses |
| `Keep Turns` (advanced) | -1 (= `Max Turns`) | Turns left after a cut |
| `Keep Images` (advanced) | -1 (= `Max Images`) | Images left after a cut |

**Basic use**

- **Where**: in a loop, put a single Trim History **right after Load Memory**: Load Memory -> Trim History -> Chat Completion (`History` input) -> Save Memory. The Chat Completion outputs the trimmed history plus its new turn, so Save Memory stores a bounded window.
- **Each limit works on its own**: above `Max Turns` turns, the oldest turns are dropped. Above `Max Images` images (counted in the kept turns), the oldest images are removed. A history within `Max Turns` never loses a turn because of its images.
- **Budget**: only the history the node receives is counted, not what the next nodes add in this run. What the node keeps plus what one run adds must fit what the server accepts: e.g. a server limited to 8 images per conversation, with 1 new image per run: `Max Images` 7.
- **What is kept**: the leading system/developer message, and whole turns: the first kept message after it is always a user message (answers before the first user message are dropped). Only images are removed from the kept turns, never text: a message left without any content gets the text `[image removed]`, so that the conversation stays valid for the server. An empty input (`None`) is passed through as is.

**Keep margins (advanced)**

Local servers reuse the start of the conversation they already processed (the **prompt cache**), and only process the new part. Every cut changes that start: the server then processes the whole conversation again, images included, which can take a while.

- **With the default `Keep` values**, every run cuts once the limit is reached.
- **Lower values cut less often**, at the cost of a shorter history right after a cut: with `Max Images` 7, `Keep Images` 3 and 1 new image per run, one cut every 5 runs.
- **Out of range values**: a `Keep` value greater than its `Max` value is capped at it, and `Keep Images` is ignored when `Max Images` is -1 (no image is ever removed). No error, since the node runs in unattended loops, but a console warning (once per combination of values) says which value was capped or ignored.

### Save Memory / Load Memory

Carry a value from one queued run to the next: an answer, an image analysis, a whole conversation history... (see [Looping workflows](#looping-workflows)).

| Node | Input | Default | Description |
|---|---|---|---|
| Save Memory | `Key` | `memory` | The name the value is stored under |
| | `Value` | | Any value: a copy is stored, and the value is passed through |
| Load Memory | `Key` | `memory` | The name to read |
| | `Reset` | off | Wipes the key and returns nothing (or `Default`) for this run |
| | `Default` | | Optional: returned when nothing is stored |

- **Load Memory returns nothing** (`None`, as if the input was not connected) when the key is empty or nothing is stored yet, e.g. on the first run. A stored `None` counts as nothing stored. Connect `Default` to return a value instead: needed when the output goes into a required input, like a text concatenation.
- **`Reset`**: **untick it afterwards**, otherwise the key is wiped on every run.
- **Save Memory always runs**, on every queued prompt, even with nothing connected downstream.
- **RAM only**: the memory is a helper, not a database. It is shared by every prompt and workflow using the same key, and lost when ComfyUI restarts.

### Unload Model

Asks a local inference server to unload its model, e.g. to free VRAM before an image generation. It passes its `Value` input through unchanged, once the unload is done.

| Input | Default | Description |
|---|---|---|
| `Value` | | Any value to unload after, usually the Chat Completion `Response`: the node runs once it is ready |
| `API Client` | | The same [Client](#client) as your Chat Completion, no change needed |
| `Server Type` | `ollama` | Your server software (table below). There is no "try every server" mode |
| `Model` | empty | The model name, as in the Chat Completion. Required for llama.cpp, LM Studio, Ollama and Unsloth Studio; tabbyAPI ignores it and unloads its loaded model |
| `Timeout` (advanced) | 10 | Duration of the unload request, in seconds. It is never retried and redirects are not followed |
| `Fail On Error` (advanced) | off | Stops the workflow when the unload fails. Off, a warning is logged and the value is passed through anyway |

- **Same client**: keep the usual `.../v1` Base URL. Its API key and its connection are reused, and the server root used by the unload routes below is the Base URL without its trailing `/v1`.
- **Cancel** also stops a pending unload request at once.

| Server | Unload call (from the server root) | Reload on the next request | Server side requirements |
|---|---|---|---|
| Ollama | `POST /api/generate` with `keep_alive: 0` | yes, automatic | none |
| LM Studio | `POST /api/v1/models/unload` | yes, with *Just-in-time model loading* (enabled by default) | its API token in `OPENAI_API_KEY`, if authentication is enabled |
| llama.cpp | `POST /models/unload` | yes, unless started with `--no-models-autoload` | router mode only (1) |
| Unsloth Studio | `POST /models/unload` | only with *Model auto-switch* enabled (2) | version 0.1.907 or later, and a Studio API key (2) |
| tabbyAPI | `POST /v1/model/unload` | only with inline model loading (3) | an admin key in `OPENAI_API_KEY` |

1. `llama-server` started without `-m` (with `--models-dir`, `--models-preset` or the cache).
2. Both under *Settings > API* in Unsloth Studio (*Model auto-switch* is off by default). The key (`sk-unsloth-...`) goes in `OPENAI_API_KEY`; with keyless access, leave `OPENAI_API_KEY` unset.
3. `inline_model_loading: true` in its `config.yml`, and the Chat Completion model name must match a folder of its `model_dir`. Unload and inline loading both require an admin key.

vLLM (sleep mode) and SGLang (memory release) are not supported: after their unload call, the next request waits or fails until an explicit wake up call, which would break the next generation.

This node was first requested by @JaneSmith in [issue #6](https://github.com/hekmon/comfyui-openai-api/issues/6), and built from the unload helper of [PR #11](https://github.com/hekmon/comfyui-openai-api/pull/11) by @nikich340.

## Guides

### Looping workflows

With Save Memory and Load Memory, a workflow can reuse what the previous run produced. Typical loop: generate an image, analyse it with a Chat Completion, save the analysis (or the history), and on the next run load it to improve the next prompt.

- **Wire the loop as Load Memory -> ... -> Save Memory.** Load Memory returns the value saved during the **previous** run: ComfyUI decides which nodes must run before executing any of them.
- **Connect them, directly or not.** If a Save Memory and a Load Memory use the same key without being connected, their execution order in a run is undefined.
- **They run on every queued prompt.** Load Memory and the nodes downstream stay cached only while no Save Memory writes their key, and every Save Memory run counts as a change, even with an identical value.
- **Trim the history.** A stored history grows with every run: put a [Trim History](#trim-history) node right after Load Memory.

The two [examples](#examples) are loops:

- **Chat loop**: Trim History lets the history grow up to 10 turns (`Max Turns`), then cuts it back to the last 6 (`Keep Turns`): once full, the conversation start changes once every 5 runs instead of on every run.
- **Image feedback loop**: each run adds 2 turns (prompt, critique) and 1 image. Trim History lets the history grow up to 2 runs (`Max Turns` 4, `Max Images` 2), then cuts it back to the last run (`Keep Turns` 2, `Keep Images` 1), once every 2 runs: a request never holds more than 6 turns and 3 images.

### Skipping a request

ComfyUI only runs the nodes that an output node needs. Behind the core `If/Else Switch` node, whose inputs are lazy, a Chat Completion on the branch not taken does not run: no request is sent.

- **Every use of its outputs must go through a switch.** A Chat Completion that also feeds an output node directly (Save Memory, Unload Model, Preview Any...) always runs. Bypass or remove such previews when you want the request to be skipped.
- **In a memory loop, route the `History` through a second switch** before Save Memory: the Chat Completion `History` on the branch that runs it, the Load Memory output on the other one. The stored history is then saved back unchanged when the request is skipped.
- **Connect both branches**: a switch with an unconnected branch outputs nothing, and Save Memory would wipe the history.
- **Switch nodes without lazy inputs** (some custom node packs) run both branches, so both requests are sent.

### Reasoning models

Reasoning (thinking) models return their reasoning next to the answer: in a `reasoning_content` field (llama.cpp, SGLang, LM Studio, tabbyAPI, DeepSeek...), a `reasoning` one (recent vLLM, Ollama, OpenRouter, Groq...), or as thinking parts of the content (Mistral).

- **Kept in the `History` output**, as received. With `Send Back Reasoning` (advanced, on by default), the next Chat Completions send it back to the server as is; turn it off to send the history without it. Whether the model reuses it is up to the server and its chat template: some keep the past reasoning only with a chat template option (e.g. `preserve_thinking`, set with [Extra Body](#options)).
- **When to turn Send Back Reasoning off**: if the context grows too much, or if the server rejects the reasoning of the previous answers (some cloud providers answer HTTP 400 or 422 on the second turn).
- **`Show Reasoning`** (advanced, off by default, as a long reasoning floods the console) prints it in the console.
- **Relays** (services forwarding the requests to other providers, like OpenRouter or the Vercel AI Gateway) also return `reasoning_details` blocks: text, summary and encrypted. They are kept unmodified and in order. The content of the encrypted blocks is never shown, only their count and size.
- **The OpenAI API itself returns no reasoning** on chat completions, only its token count: its summaries and encrypted reasoning are only available through its Responses API.

### Prompt cache and usage stats

The usage stats of each request are shown under the Chat Completion and on one line in the console, e.g.:

```
Prompt tokens: 6165 (cache read: 4800, uncached: 1365), Completions tokens: 72 (reasoning: 59, content: 13)
```

- **Cache read**: the prompt tokens the server reused from its prompt cache. `cache read: 0` is a cache miss.
- **Uncached**: the prompt tokens the server had to process, cache writes included.
- **Cache write**: the tokens written to the cache, when the server reports some.
- **Reasoning and content**: the completion tokens spent reasoning, and the rest, the answer itself. Shown when the server reports reasoning tokens.
- **No cache figures at all**: the server does not report its cache. vLLM needs `--enable-prompt-tokens-details`, SGLang `--enable-cache-report`. Older llama.cpp builds are read from their `timings`.
- **Hybrid models** (linear attention or Mamba layers, e.g. Qwen3-Next): vLLM only caches whole blocks of more than a thousand tokens (the size depends on the model), so a shorter prompt always shows `cache read: 0`.
- **Relays and gateways** may change what the server reports, or the request itself (e.g. Bifrost raises any max tokens below 16 to 16).

## API key

The Client node reads the API key from the `OPENAI_API_KEY` environment variable **at execution time**. It is intentionally impossible to enter an API key inside the workflow.

**Never store a real API key in a workflow.** ComfyUI saves every node widget value, even from bypassed nodes, into the metadata of generated images and exported workflow files. Anyone you share an image or a workflow with (Discord, Reddit, forum attachments...) can read the key back from those metadata. This is a ComfyUI core behavior (see the [official workflow metadata documentation](https://docs.comfy.org/development/api-development/workflow-metadata)) that no custom node can opt out of: any node exposing a key widget leaks it.

- **Authenticated servers** (official OpenAI API, cloud providers...): set the variable before starting ComfyUI, e.g. `export OPENAI_API_KEY=sk-...` (Linux/macOS). On Windows, `setx OPENAI_API_KEY sk-...`, then start ComfyUI from a new terminal (or relaunch it). In Docker, pass it with `-e OPENAI_API_KEY=sk-...`.
- **ComfyUI Desktop**: an installation can define its own environment variables, which override the system ones: make sure `OPENAI_API_KEY` there holds the key only. A key with non ASCII or control characters (e.g. pasted text) is refused with an explicit error, without showing it. Surrounding spaces are removed.
- **Keyless servers** (vLLM, llama.cpp, some Ollama setups, Unsloth Studio with keyless access...): nothing to configure. These servers require an `Authorization` header but do not check its value: if the variable is not set (or empty), the client sends the `not-needed` placeholder (Unsloth Studio only accepts a few such dummy values).
- **Imported workflows**: the key is sent to whatever `Base URL` the Client node of the workflow uses, so check it before running a workflow you did not write.
- **Defense in depth**: if you share generated images, launch ComfyUI with `--disable-metadata` (or enable *Settings > Server-Config > Disable saving prompt metadata in files* in ComfyUI Desktop) to strip the workflow metadata from output files.

If you used a version older than 3.0.0 with a real key and shared generated images or workflow files, **rotate that key** and clean up your saved workflows: these versions stored the key inside every image and workflow export.

## Troubleshooting

- **Connection error or timeout**: check that the server is running and that the `Base URL` reaches it from the machine running ComfyUI (`localhost` is that machine, see [Client](#client)).
- **HTTP 404**: a model name the server does not know, or a `Base URL` without its API prefix (usually `/v1`; a Base URL without any path triggers a warning in the console).
- **HTTP 401**: the server needs an API key, see [API key](#api-key). Free cloud models (e.g. OpenRouter `:free` ones) need one too.
- **HTTP 400 or 422 on the second turn, mentioning `reasoning`**: the server rejects the reasoning sent back. Turn `Send Back Reasoning` off.
- **HTTP 429**: rate limited. With free cloud models, the shared quota of the upstream provider is often exhausted for a while: retry later or pick another model.
- **Empty response**: the model probably spent its token budget reasoning (the console warning says so). Raise Max Tokens (512 by default on the option node), or lower the Reasoning Effort.
- **The same answer on every run**: ComfyUI cached the node, as none of its inputs changed. Enable `Force Regen`.
- **A new request on every run**: an input changes on every run, often a Seed node left on *randomize*.
- **`cache read: 0` on every request**: see [Prompt cache and usage stats](#prompt-cache-and-usage-stats); hybrid models only cache long prompts.

## Installation

Requires ComfyUI 0.10.0 or later.

The project has a low footprint, with 3 dependencies of which only one is new to ComfyUI: `openai` (the official OpenAI API Python library), `numpy` and `Pillow` (both already used by ComfyUI).

### ComfyUI Manager

Search for `OpenAI API` in the `Custom Nodes Manager` and install it. Its page on the Comfy registry: [registry.comfy.org/nodes/openai-api](https://registry.comfy.org/nodes/openai-api).

> **This README documents the `main` branch, which can be ahead of the latest release.** The registry only serves tagged releases: to use features documented here before the next release, install from `main` manually.

### Manually

On the GitHub page, click the green `<> Code` button, then `Download ZIP`. Extract the root folder of the zip file into your `ComfyUI/custom_nodes` directory.

Then install the dependencies with the Python environment used by ComfyUI (`openai` is not a ComfyUI dependency):

```bash
pip install -r ComfyUI/custom_nodes/<extracted folder>/requirements.txt
```

With the Windows portable build, use its embedded Python: `python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\<extracted folder>\requirements.txt`.
