from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from src.models.model_pipeline_builder import build_model_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier


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

def test_build_naive_bayes_pipeline():
    pipeline = build_model_pipeline("naive_bayes")

    assert isinstance(pipeline, Pipeline)
    assert "tfidf" in pipeline.named_steps
    assert "clf" in pipeline.named_steps
    assert isinstance(pipeline.named_steps["clf"], MultinomialNB)


def test_build_random_forest_pipeline():
    pipeline = build_model_pipeline("random_forest")

    assert isinstance(pipeline, Pipeline)
    assert "tfidf" in pipeline.named_steps
    assert "clf" in pipeline.named_steps
    assert isinstance(pipeline.named_steps["clf"], RandomForestClassifier)