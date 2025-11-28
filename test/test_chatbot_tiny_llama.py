from src.models.chatbot_tiny_llama import Chatbot


def test_encode_prompt():
    bot = Chatbot()
    assert bot.encode_prompt("test") == "test"


def test_decode():
    bot = Chatbot()
    assert bot.decode("abc") == "abc"
