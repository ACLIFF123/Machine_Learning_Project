import pandas as pd 
from sklearn.model_selection import train_test_split
from src.ingest import CLEANED_DATA_PATH


def load_cleaned_data(filepath = CLEANED_DATA_PATH):
    return pd.read_csv(filepath)


def main():
    df = load_cleaned_data()
    print (df.head())
        

if __name__ == "__main__":
    main()
    