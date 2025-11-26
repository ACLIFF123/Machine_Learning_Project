from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

def build_model_pipeline(model_type="logistic"):
    if model_type == "logistic":
        return Pipeline([
            ("tfidf", TfidfVectorizer(stop_words="english")),
            ("clf", LogisticRegression(max_iter=1000))
        ])

    elif model_type == "naive_bayes":
        return Pipeline([
            ("tfidf", TfidfVectorizer(stop_words="english")),
            ("clf", MultinomialNB())
        ])

    elif model_type == "random_forest":
        return Pipeline([
            ("tfidf", TfidfVectorizer(stop_words="english")),
            ("clf", RandomForestClassifier(n_estimators=200))
        ])

    else:
        raise ValueError(f"Unsupported model type: {model_type}")