import pickle
import os
from src.logger import get_logger

logger = get_logger(__name__)

class NewsClassifier:
    def __init__(self, model_path="model/model.pkl"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        logger.info(f"Loading model from {model_path}")
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

    def classify(self, text: str) -> str:
        if not isinstance(text, str) or text.strip() == "":
            raise ValueError("Input text must be a non-empty string")

        prediction = self.model.predict([text])[0]
        logger.debug(f"Input text: {text}")
        logger.debug(f"Model prediction: {prediction}")

        return prediction
    
    