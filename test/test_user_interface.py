import src.chatbot_user_interface as ui_module  # <-- THIS is where ui_process lives


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

    # ✅ Patch the GLOBALS inside src/chatbot_user_interface.py
    monkeypatch.setattr(ui_module, "extractor_llm", fake_llm)
    monkeypatch.setattr(ui_module, "NewsClassifier", FakeClassifier)

    raw_text = "Long article about the Champions League"

    # Call the function from the same module
    headline, category = ui_module.ui_process(raw_text)

    # --- Assertions ---

    # Extractor got the raw text
    assert fake_llm.called_with == raw_text
    # ui_process returns the extracted headline
    assert headline == "Some Extracted Headline"

    # Classifier was instantiated and used correctly
    assert FakeClassifier.last_instance is not None
    assert FakeClassifier.last_instance.model_path == "model/model.pkl"
    assert FakeClassifier.last_instance.called_with == "Some Extracted Headline"
    assert category == "sports"