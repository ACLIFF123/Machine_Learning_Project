from src.models.chatbot_tiny_llama import Chatbot

class ExtractorLLM:

    def __init__(self):
        self.llm = Chatbot()
        self.system_prompt = (
            "<|system|>\n"
            "You extract news headlines. "
            "Return ONLY the cleaned headline. No explanations.\n"
            "<|end|>\n"
        )

    def extract(self, user_input: str) -> str:
        prompt = (
            self.system_prompt +
            f"<|user|>\nExtract the main news headline from this:\n{user_input}\n<|end|>"
        )

        return self.llm.generate_single_reply(prompt).strip()