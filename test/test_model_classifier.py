import pandas as pd
import os
from src.models.model_classifier import ClassificationModel

def test_classifier_train_and_predict():
    model = ClassificationModel("logistic")

    X = pd.Series(["football match today", "big science breakthrough"])
    y = pd.Series(["sports", "sci/tech"])

    model.train(X, y)

    prediction = model.predict("football team wins match")
    assert isinstance(prediction, str)



def test_classifier_evaluation_output():
    model = ClassificationModel("logistic")

    X = pd.Series(["football news", "tech update"])
    y = pd.Series(["sports", "sci/tech"])

    model.train(X, y)
    model.evaluate(X, y)

    assert "accuracy" in model.metrics
    assert "cm" in model.metrics
    assert "classification_report" in model.metrics