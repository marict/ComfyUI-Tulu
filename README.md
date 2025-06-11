# ComfyUI-Tulu

ComfyUI-Tulu provides two simple nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that integrate the **Tulu** language models from [AllenAI's Open-Instruct](https://github.com/allenai/open-instruct).

## Nodes

### Load Tulu Model
Downloads and loads a Tulu model using HuggingFace. The default model is `allenai/tulu-2-dpo-7b`. The node returns a text generation pipeline that runs on the user's GPU when available.

### Tulu Prompt
Takes a loaded pipeline and a prompt (system and user parts) and returns the model's response.

## Installation

1. Install the dependencies:

```bash
pip install torch==2.1.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install transformers requests huggingface_hub
```

2. Place this repository inside ComfyUI's `custom_nodes` folder.
3. Start ComfyUI and the nodes will appear in the interface.

The first run will download the Tulu weights automatically.

## Tests

Unit tests are located in the `tests/` directory and can be run with:

```bash
pytest
```
