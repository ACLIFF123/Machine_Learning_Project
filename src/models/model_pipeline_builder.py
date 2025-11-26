from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def build_model_pipeline(model_type="logistic"):
    if model_type == "logistic":
        return Pipeline([
            ("tfidf", TfidfVectorizer(stop_words="english")),
            ("clf", LogisticRegression(max_iter=1000))
        ])
    else:
        raise ValueError(f"Unsupported model type: {model_type}")