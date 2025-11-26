from src.train_model import load_cleaned_data, split_dataset
import pandas as pd
import os 



def test_load_dataset(tmp_path):

    
    csv_file = tmp_path / "sample.csv"
    csv_file.write_text("news_text,category\n", encoding="utf-8")

    # Call the function
    df = load_cleaned_data(csv_file)

    assert isinstance(df, pd.DataFrame)
    assert "news_text" in df.columns
    assert "category" in df.columns


def test_split_dataset():
    X = pd.Series(["a", "b", "c", "d"])
    y = pd.Series(["x", "x", "y", "y"])
    result = split_dataset(X,y, random_seed=42, test_size=0.2)
    assert len(result) == 4
    X_train, X_val, y_train, y_val = result
    assert len(X_train) == 3
    assert len(X_val) == 1
    assert len(y_train) == 3
    assert len(y_val) == 1