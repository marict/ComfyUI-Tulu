import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class LoadTuluModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_id": ("STRING", {"default": "allenai/tulu-2-dpo-7b"})
            }
        }

    RETURN_TYPES = ("TULU_MODEL",)
    FUNCTION = "load"
    OUTPUT_NODE = True

    def load(self, model_id: str):
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto",
        )
        return (pipe,)

class TuluPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("TULU_MODEL",),
                "system_prompt": ("STRING", {"default": ""}),
                "user_prompt": ("STRING", {"default": ""}),
                "max_new_tokens": ("INT", {"default": 64, "min": 1, "max": 1024}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate"

    def generate(self, model, system_prompt: str, user_prompt: str, max_new_tokens: int = 64, temperature: float = 0.7):
        prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{user_prompt}\n<|assistant|>"
        outputs = model(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
        )
        text = outputs[0]["generated_text"][len(prompt):]
        return (text,)

NODE_CLASS_MAPPINGS = {
    "LoadTuluModel": LoadTuluModel,
    "TuluPrompt": TuluPrompt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadTuluModel": "Load Tulu Model",
    "TuluPrompt": "Tulu Prompt",
}
