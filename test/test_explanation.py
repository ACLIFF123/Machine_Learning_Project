import src.models.explanation as explanation_module 


class FakeChatbot:
    def __init__(self, response="This is the explanation."):
        self.response = response
        self.last_prompt = None
        self.calls = 0

    def generate_single_reply(self, prompt: str) -> str:
        self.last_prompt = prompt
        self.calls += 1
        return self.response


def test_explain(monkeypatch):
    # Arrange: patch Chatbot inside the ExplanationLLM module
    fake_llm = FakeChatbot("A clear explanation.")
    monkeypatch.setattr(
        explanation_module,
        "Chatbot",
        lambda: fake_llm
    )

    explainer = explanation_module.ExplanationLLM()

    headline = "Oil prices soar after major supply disruption"
    category = "business"
    confidence = 0.87

    # Act
    explanation = explainer.explain(headline, category, confidence)

    # Assert: output from fake LLM
    assert explanation == "A clear explanation."

    # Assert: LLM was actually called
    assert fake_llm.calls == 1

    # Prompt structure checks (behavioral testing)
    prompt = fake_llm.last_prompt
    assert "<|system|>" in prompt
    assert "You explain why it is that news categories" in prompt
    assert f'Headline: "{headline}"' in prompt
    assert f"Category: {category}" in prompt
    assert "Model confidence: 87.0%" in prompt
    assert "Explain why this fits the category" in prompt
    assert prompt.endswith("<|end|>")