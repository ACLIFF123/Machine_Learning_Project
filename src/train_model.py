import pandas as pd 
from sklearn.model_selection import train_test_split
from src.ingest import CLEANED_DATA_PATH
from pprint import pprint

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
    

def main():
    df = load_cleaned_data()
    X = df["news_text"]
    y = df["category"]
    
    X_train, X_val, y_train, y_val = split_dataset(X, y)
    # Print sizes
    print("Train size:", len(X_train))
    print("Validation size:", len(X_val))
    # Print examples
    print("\nExample train row:")
    print(X_train.iloc[0], "→", y_train.iloc[0])
    print("\nExample validation row:")
    print(X_val.iloc[0], "→", y_val.iloc[0])
        

if __name__ == "__main__":
    main()
