# ComfyUI OpenAI API

This repository contains ComfyUI nodes that integrates with the OpenAI API: it allows you to use language models and vision language models within your workflow.

It is [KISS](https://en.wikipedia.org/wiki/KISS_principle) by design and intended for those who only wants basic capabilities without having to import massive projects like [LLM party](https://github.com/heshengtao/comfyui_LLM_party): only the chat completions endpoint (with vision support) is implemented as it should be enough for 99% of use cases.

Thanks to its simplicity the project has a low footprint: it only has 1 external dependency (3 in total) !

- `openai` the official OpenAI bindings for Python
- `numpy` for computation but it is already a dependency of ComfyUI
- `Pillow` also already used by ComfyUI for image processing

The default `base_url` parameter value targets the official OpenAI API endpoint but by changing it, you can also use this project with any openai compatible servers like vLLM, TGI, etc...

## Installation

### ComfyUI Manager

Search for `OpenAI API` in the `Custom Nodes Manager` and install it.

![Entry in the ComfyUI manager](res/comfymanager.png)

### Manually

On the github interface, click the green `<> Code` button and then `Download ZIP`. Extract the root folder of the zip file into your `ComfyUI/custom_nodes` directory.

## Example

### Simple

You only need 2 nodes to get started:

- a client node that holds your API key and the base url if you are using an openai compatible server
- a chat completion node that sends a prompt to the model and returns the response as a string

![Simple Example](res/vl.png)

### Advanced

If you want to customize the chat completion, you can chain options to modify the request. Most common options are available as predefined nodes but you can inject any key/value pair using the `Extra body` node.

Also, several chat completions can be chained to share the context history of previous turns.

![Advanced Example](res/advanced.png)
