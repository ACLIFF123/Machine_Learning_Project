import torch
import ollama 

class Chatbot:
    def __init__(self, model_name: str = "tinyllama"):
        print("Using Ollama backend")
        self.model_name = model_name

    def encode_prompt(self, prompt: str):
        # No tokenizer; just return the raw string for compatibility
        return prompt

    def decode(self, text: str):
        return text

    def generate_single_reply(self, prompt: str) -> str:
        # Ignore encode/decode and go straight to Ollama
        response = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": 0.8,
                "top_p": 0.95,
                "top_k": 50,
                "num_predict": 500,
            },
        )
        return response["message"]["content"].strip()
        