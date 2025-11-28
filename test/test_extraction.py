from src.models.extraction import ExtractorLLM

class FakeChatbot:
    def __init__(self, response="A Clean Headline"):
        self.response = response
        self.last_prompt = None

    def generate_single_reply(self, prompt):
        self.last_prompt = prompt
        return self.response


def test_extract_returns_string(monkeypatch):
    fake_llm = FakeChatbot(response="Breaking News!")

    monkeypatch.setattr(
        "src.models.extraction.Chatbot",
        lambda: fake_llm
    )

    extractor = ExtractorLLM()
    result = extractor.extract("Some long article...")

    assert isinstance(result, str)
    assert result == "Breaking News!"
    assert fake_llm.last_prompt.endswith("<|end|>")