import types
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import comfyui_tulu.nodes as nodes

class DummyPipeline:
    def __call__(self, prompt, max_new_tokens=1, temperature=0.0, do_sample=False):
        return [{"generated_text": prompt + " response"}]

def test_prompt_node():
    pipe = DummyPipeline()
    prompt_node = nodes.TuluPrompt()
    output, = prompt_node.generate(
        pipe,
        system_prompt="sys",
        user_prompt="hello",
        max_new_tokens=1,
        temperature=0.0,
    )
    assert "response" in output


def test_load_model_patch(monkeypatch):
    def fake_from_pretrained(model_id, *args, **kwargs):
        return "model"

    def fake_pipeline(task, model=None, tokenizer=None, device_map=None):
        assert model == "model"
        assert tokenizer == "model"
        return DummyPipeline()

    monkeypatch.setattr(nodes, "AutoTokenizer", types.SimpleNamespace(from_pretrained=fake_from_pretrained))
    monkeypatch.setattr(nodes, "AutoModelForCausalLM", types.SimpleNamespace(from_pretrained=fake_from_pretrained))
    monkeypatch.setattr(nodes, "pipeline", fake_pipeline)

    loader = nodes.LoadTuluModel()
    pipe, = loader.load("some-model")
    assert isinstance(pipe, DummyPipeline)
