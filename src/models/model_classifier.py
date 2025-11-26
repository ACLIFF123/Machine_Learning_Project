from src.models.model_pipeline_builder import build_model_pipeline
from src.models.evaluation import get_metrics_for_evaluation
from src.logger import get_logger
import pickle
import os

logger = get_logger(__name__)

class ClassificationModel:

    def __init__(self, model_type="logistic"):
        self.model = build_model_pipeline(model_type)
        self.metrics = {}
    
    def train(self, X_train, y_train):
        logger.info("Training model")
        self.model.fit(X_train, y_train)

    def evaluate(self, X_val, y_val):
        logger.info("Evaluating model")
        self.metrics = get_metrics_for_evaluation(self.model, X_val, y_val)

        logger.info(f"Accuracy:  {self.metrics['accuracy']:.3f}")
        logger.info(f"\nConfusion Matrix:\n{self.metrics['cm']}")
        logger.info(f"\nClassification Report:\n{self.metrics['classification_report']}")


    def save(self, path="model/model.pkl"):
        os.makedirs("model", exist_ok= True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        logger.info(f"Model saved to {path}")


    def predict(self, text: str):
        return self.model.predict([text])[0]