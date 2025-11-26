from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from src.models.model_pipeline_builder import build_model_pipeline

def test_build_model_pipeline():
    pipeline = build_model_pipeline("logistic")

    assert isinstance(pipeline, Pipeline)
    assert "tfidf" in pipeline.named_steps
    assert "clf" in pipeline.named_steps
    assert isinstance(pipeline.named_steps["clf"], LogisticRegression)

def test_pipeline_can_train():
    pipeline = build_model_pipeline("logistic")

    X = ["hello world", "machine learning is cool"]
    y = ["greeting", "statement"]

    pipeline.fit(X, y)