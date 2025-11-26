import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from src.ingest import CLEANED_DATA_PATH
from pprint import pprint
from src.logger import get_logger

logger = get_logger(__name__)

def load_cleaned_data(filepath = CLEANED_DATA_PATH):
    return pd.read_csv(filepath)

def split_dataset(X,y, test_size=0.2, random_seed =42):
    X_train, X_val, y_train, y_val = train_test_split(
        X, 
        y, 
        test_size=test_size, 
        random_state=random_seed
    )
    return X_train, X_val, y_train, y_val

def build_logistic_regression_model():

    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(stop_words="english")),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    return model 

def train_model(X_train, y_train):
    model = build_logistic_regression_model()
    model.fit(X_train, y_train)
    return model 


def evaluate_model(model, X_val, y_val):

    y_pred = model.predict(X_val)

    accuracy = accuracy_score(y_val, y_pred)
    logger.info("=== Evaluation Metrics ===")
    logger.info(f"Accuracy : {accuracy:.4f}")

    cm = confusion_matrix(y_val, y_pred)
    logger.info(f"Confusion Matrix:\n{cm}")

    report = classification_report(y_val, y_pred, zero_division=0)
    logger.info(f"Classification Report:\n{report}")

def main():
    df = load_cleaned_data()
    X = df["news_text"]
    y = df["category"]
    
    X_train, X_val, y_train, y_val = split_dataset(X, y)

    # Log sizes
    logger.info(f"Train size: {len(X_train)}")
    logger.info(f"Validation size: {len(X_val)}")

    # Log example rows
    logger.info("\nExample train row:")
    logger.info(f"{X_train.iloc[0]} → {y_train.iloc[0]}")

    logger.info("\nExample validation row:")
    logger.info(f"{X_val.iloc[0]} → {y_val.iloc[0]}")
  
    model = train_model(X_train, y_train)

    evaluate_model(model,X_val,y_val)

if __name__ == "__main__":
    main()
