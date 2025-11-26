import pandas as pd
from sklearn.model_selection import train_test_split
from src.ingest import CLEANED_DATA_PATH


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
