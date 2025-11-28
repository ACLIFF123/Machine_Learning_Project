import src.chatbot_user_interface as ui_module 

class FakeExtractorLLM:
    def __init__(self, response="Some Extracted Headline"):
        self.response = response
        self.called_with = None

    def extract(self, text: str) -> str:
        self.called_with = text
        return self.response


class FakeClassifier:
    last_instance = None

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.called_with = None
        FakeClassifier.last_instance = self

    def classify(self, text: str) -> str:
        self.called_with = text
        return "sports"


def test_ui_process(monkeypatch):
    fake_llm = FakeExtractorLLM("Some Extracted Headline")

  
    monkeypatch.setattr(ui_module, "extractor_llm", fake_llm)
    monkeypatch.setattr(ui_module, "NewsClassifier", FakeClassifier)

    raw_text = "Long article about the Champions League"

 
    headline, category = ui_module.ui_process(raw_text)

  
    assert fake_llm.called_with == raw_text
    assert headline == "Some Extracted Headline"
    assert FakeClassifier.last_instance.model_path == "model/model.pkl"
    assert FakeClassifier.last_instance.called_with == "Some Extracted Headline"
    assert category == "sports"