from src.models.chatbot_tiny_llama import Chatbot

class ExplanationLLM:
 
    def __init__(self):
        self.llm = Chatbot()
        self.system_prompt = (
            "<|system|>\n"
            "You explain why it is that news categories. "
            "You ONLY explain why the provided category is correct.\n"
            "You keep your response clear and justify why it is that category"
            "You keep your answer to 2 setences max"
            "<|end|>\n"
        )

    def explain(self, headline: str, category: str, confidence: float=None) -> str:
        conf = f"Model confidence: {confidence*100:.1f}%.\n" if confidence else ""

        prompt = (
            self.system_prompt +
            f"<|user|>\n"
            f"Headline: \"{headline}\"\n"
            f"Category: {category}\n"
            f"{conf}\n"
            "Explain why this fits the category in 2 short sentences.\n"
            "<|end|>"
        )

        return self.llm.generate_single_reply(prompt).strip()
        
        
